#!/usr/bin/env python3
"""Train the models of a launch on a machine with several GPUs, one training process per GPU. It does what cells 3 and 4
of the Colab notebooks do: the same models, settings, run names and plan files.

    python scripts/launch.py --variant base                 # a new launch on data/silver (tag v7), seeds 42,43
    python scripts/launch.py --variant large                # a new launch on data/silver_large (tag v7large), seeds 42,43
    python scripts/launch.py --variant base --launch 2026-10-09_r1   # continue that launch: finished runs are skipped
    python scripts/launch.py --variant base --n-docs 2000 --seeds 42 --no-drive   # smoke run: nothing leaves the machine
    python scripts/launch.py --variant base --status        # progress of the runs of the latest launch of the variant

Runs go to runs/<model>_<tag>_<launch>[_s<seed>], the plan of a launch to runs/launches/<launch>_<tag>.json and its
progress to runs/launches/<launch>_<tag>.state.json. A launch never overwrites a model that is already trained.

Right after training a run is delivered, while the GPU goes on with the next run:
- to Google Drive, <remote>runs/<run>/, when the rclone remote of PSEUDO_DRIVE (default `gdrive:` = MyDrive/ai-camp, set up
  by scripts/vast.sh drive) exists; a run that Drive already holds is not trained again;
- to the Hugging Face Hub as the public model <account of the token>/<run>, when a token is stored (scripts/vast.sh hf)
  and --no-publish is not given. Runs on a part of the data (--n-docs) are never published.

Two launches (base and large) can run at the same time: a GPU is given to one training process at a time.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import zoneinfo
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DATA = HERE.parent / "data"
RUNS = HERE / "runs"
NEED = ("train.parquet", "dev.parquet", "dev.jsonl")

# the two datasets of the Colab notebooks (VARIANTS of scripts/make_notebooks.py): tag, folder of data/ and of MyDrive/ai-camp, default seeds
VARIANTS = {
    "base": dict(tag="v7", folder="silver", seeds="42,43"),
    "large": dict(tag="v7large", folder="silver_large", seeds="42,43"),
}
# The models of every launch (MODELS of scripts/make_notebooks.py). wechsel_base has the settings of the earlier runs.
# modern_liberta_large is trained with windows of 2048 tokens (the model reads up to 8192): a batch of 8 such windows
# holds as many tokens as 32 windows of 512.
MODELS = [
    dict(name="wechsel_base", model="benjamin/roberta-base-wechsel-ukrainian", extra="", lr=4e-5, batch=64, accum=1, epochs=3),
    dict(name="modern_liberta_large_w2048", model="Goader/modern-liberta-large", extra="--window 2048 --stride 256", lr=3e-5,
         batch=8, accum=1, epochs=3),
]
ENV = dict(TRANSFORMERS_VERBOSITY="error", PYTHONWARNINGS="ignore", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1")
DRIVE = os.environ.get("PSEUDO_DRIVE", "gdrive:")
HF_LICENSE = "mit"  # the licence of the training corpus lang-uk/court-decisions-uk and of the wechsel base model
print_lock = threading.Lock()


def say(*a) -> None:
    with print_lock:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def file_sha1(path: Path, n: int = 12) -> str:
    """The same hash as pseudo.train.file_sha1 (not imported: that module loads torch)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def gpu_ids() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


# ---------------------------------------------------------------------------------------------- Google Drive, Hugging Face

def rclone(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args], capture_output=True, text=True)


def drive_on() -> bool:
    return bool(shutil.which("rclone")) and DRIVE.split(":")[0] + ":" in rclone("listremotes").stdout.split()


def drive_names() -> list[str]:
    """Runs and launch plans that are already on Drive (Colab launches of the same day take the same numbers)."""
    return [n for d in ("runs", "runs/launches") for n in rclone("lsf", DRIVE + d).stdout.split("\n")]


def drive_config(run: str) -> dict:
    """config.json of the run on Drive; it is uploaded last, so its presence means that the whole run is there."""
    r = rclone("cat", f"{DRIVE}runs/{run}/config.json")
    try:
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}
    except json.JSONDecodeError:
        return {}


def upload(run: str) -> str:
    """Copy the run to Drive; returns what the progress file shows."""
    dst = f"{DRIVE}runs/{run}"
    mine = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8")).get("fingerprint")
    there = drive_config(run)
    if there and there.get("fingerprint") != mine:
        say(f"{run}: another model is already saved on Drive under this name - NOT overwritten")
        return "NOT saved: the name is taken"
    for r in (rclone("copy", str(RUNS / run), dst, "--exclude", "/config.json", "--exclude", "/checkpoints/**"),
              rclone("copyto", str(RUNS / run / "config.json"), f"{dst}/config.json")):
        if r.returncode != 0:
            say(f"{run}: upload to Drive FAILED ({r.stderr.strip()[-200:]}) - run the launch again to retry")
            return "FAILED"
    say(f"{run}: saved to {dst}")
    return "saved"


def hf_token() -> str:
    """HF_TOKEN, or the token stored by scripts/vast.sh hf / `hf auth login` (the vast.ai image keeps HF_HOME in /workspace)."""
    files = [Path.home() / ".cache" / "huggingface" / "token", Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "token"]
    return os.environ.get("HF_TOKEN") or next((f.read_text().strip() for f in files if f.exists()), "")


def publish(run: str, token: str) -> str:
    """Publish the model as the public repo <account of the token>/<run>; returns its address or FAILED."""
    tag = re.search(r"_(v\d+[a-z]*)_\d{4}-", run)  # wechsel_base_v7large_2026-10-09_r1 -> v7large
    data = f"the synthetic dataset `{tag.group(1)}`" if tag else "a synthetic dataset"
    note = (f"Trained on {data}: court decisions of the Unified State Register of Court Decisions of Ukraine whose anonymised "
            "fragments are filled with generated values. The generated addresses come from the Ukrposhta directory and from "
            "OpenStreetMap (© OpenStreetMap contributors, ODbL).")
    r = subprocess.run([sys.executable, str(HERE / "scripts" / "publish.py"), "--model", str(RUNS / run), "--repo", run,
                        "--license", HF_LICENSE, "--note", note, "--message", f"Upload {run}"],
                       capture_output=True, text=True, env={**os.environ, "HF_TOKEN": token})
    if r.returncode != 0:
        say(f"{run}: publishing to Hugging Face FAILED ({(r.stderr or r.stdout).strip()[-200:]}) - run the launch again to retry")
        return "FAILED"
    say(f"{run}: published {r.stdout.strip()}")
    return r.stdout.strip()


# ------------------------------------------------------------------------------------------------------------ the plan

def top_of_history(n: int, have: list[dict]) -> list[dict]:
    """The n best other models of the past runs, by the test scores of the history (scripts/leaderboard.py writes it)."""
    sys.path.insert(0, str(HERE / "src"))
    from pseudo.data import DECODE_VERSION
    from pseudo.leaderboard import load_history, test_version, top_models

    best = top_models(load_history(RUNS), n + len(have), test_version(DATA / "test" / "court_cases.jsonl"), DECODE_VERSION)
    return [m for m in best if m["model"] not in {t["model"] for t in have}][:n]


def new_plan(a, v: dict, drive: bool) -> dict:
    tag, folder = v["tag"], v["folder"]
    missing = [n for n in NEED if not (DATA / folder / n).exists()]
    if missing:
        raise SystemExit(f"{', '.join(missing)} missing in {DATA / folder}: copy the dataset first (cell 2 of the notebook or scripts/vast.sh push)")
    names = [n.strip() for n in a.models.split(",") if n.strip()]
    unknown = set(names) - {m["name"] for m in MODELS}
    if unknown:
        raise SystemExit(f"unknown models: {', '.join(sorted(unknown))}; known: {', '.join(m['name'] for m in MODELS)}")
    top = [dict(m) for m in MODELS if not names or m["name"] in names]
    if a.top_n:
        top += top_of_history(a.top_n, top)
    seeds = [int(s) for s in (a.seeds or v["seeds"]).split(",") if s.strip()]
    now = datetime.datetime.now(zoneinfo.ZoneInfo("Europe/Kyiv"))
    date = now.strftime("%Y-%m-%d")
    seen = [p.name for p in list(RUNS.iterdir()) + list((RUNS / "launches").iterdir())] + (drive_names() if drive else [])
    taken = [int(m.group(1)) for n in seen if (m := re.search(rf"{date}_r(\d+)", n))]
    launch = f"{date}_r{max(taken, default=0) + 1}"
    jobs = [dict(run=f"{m['name']}_{tag}_{launch}" + (f"_s{s}" if s != 42 else ""), name=m["name"], model=m["model"],
                 extra=m["extra"], lr=m["lr"], batch=m["batch"], accum=m["accum"], epochs=m["epochs"], seed=s, data=folder,
                 n_docs=a.n_docs)
            for m in top for s in seeds]
    return dict(launch=launch, tag=tag, folder=folder, created=now.strftime("%Y-%m-%d %H:%M"), top=top, jobs=jobs)


def cost(job: dict) -> int:
    """The large models take several times longer: they start first, so the GPUs finish at about the same time."""
    return 3 if "large" in job["name"] else 1


def state(job: dict) -> dict:
    p = RUNS / job["run"] / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def trained(job: dict) -> bool:
    return state(job).get("status") == "done" and (RUNS / job["run"] / "model").is_dir()


def show(plan: dict) -> None:
    print(f"\nLAUNCH = {plan['launch']}  (--launch {plan['launch']} continues this launch after a restart)\n")
    for i, m in enumerate(plan["top"], 1):
        past = f"mean test macro F1 {m['score']:.4f} over {len(m['runs'])} run(s): {', '.join(m['runs'])}" if "score" in m else "from MODELS"
        print(f"{i}. {m['name']:28s} {m['model']}  lr {m['lr']} batch {m['batch']}x{m['accum']} epochs {m['epochs']} {m['extra']}\n   {past}")
    print(f"\n{len(plan['jobs'])} runs on {plan['folder']}:", *[j["run"] for j in plan["jobs"]], sep="\n  ", flush=True)


# ------------------------------------------------------------------------------------------------------------ training

class Progress:
    """runs/launches/<launch>_<tag>.state.json: what the launcher is doing with every run; --status reads it."""

    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.data = {"pid": os.getpid(), "runs": old.get("runs", {})}  # what an earlier attempt delivered is kept

    def get(self, run: str, field: str) -> str:
        return str(self.data["runs"].get(run, {}).get(field) or "")

    def mark(self, run: str, **fields) -> None:
        with self.lock:
            self.data["runs"].setdefault(run, {}).update(fields)
            self.data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)


def train(job: dict, gpu: int) -> bool:
    out = RUNS / job["run"]
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "scripts" / "train.py"), "--model", job["model"], "--out", str(out), "--epochs", str(job["epochs"]),
           "--lr", str(job["lr"]), "--batch", str(job["batch"]), "--grad-accum", str(job["accum"]), "--dev-docs", "500",
           "--seed", str(job["seed"]), *job["extra"].split()]
    if job["n_docs"]:
        cmd += ["--n-docs", str(job["n_docs"])]
    env = {**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(gpu), "PSEUDO_SILVER": str(DATA / job["data"])}
    with open(out / "console.log", "a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return rc == 0 and trained(job)


def deliver(job: dict, progress: Progress, drive: bool, token: str) -> None:
    """Drive first: it is the copy that the leaderboard and the other notebooks read."""
    run = job["run"]
    if drive and progress.get(run, "drive") != "saved":
        progress.mark(run, drive="uploading")
        progress.mark(run, drive=upload(run))
    if token and not job["n_docs"] and not progress.get(run, "hf").startswith("http"):
        progress.mark(run, hf="uploading")
        progress.mark(run, hf=publish(run, token))


def worker(gpu: int, todo: queue.Queue, total: int, failed: list[str], progress: Progress, deliveries: list, drive: bool, token: str) -> None:
    with open(f"/tmp/pseudo_gpu{gpu}.lock", "w") as lock:
        while True:
            fcntl.flock(lock, fcntl.LOCK_EX)  # another launch may be training on this GPU: wait for its run to end
            try:
                try:
                    job = todo.get_nowait()
                except queue.Empty:
                    return
                say(f"GPU {gpu}: start {job['run']} ({job['model']}, seed {job['seed']}, data {job['data']}); {todo.qsize()} of {total} waiting")
                progress.mark(job["run"], state="running", gpu=gpu, started=time.time(), finished=None)
                t0 = time.time()
                ok = train(job, gpu)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            progress.mark(job["run"], state="done" if ok else "failed", finished=time.time())
            if ok:
                c = state(job)
                say(f"GPU {gpu}: done  {job['run']} in {(time.time() - t0) / 60:.0f} min, best dev F1 {c.get('best_dev_f1') or 0:.4f} "
                    f"(epoch {c.get('best_epoch')}), fingerprint {c.get('fingerprint')}")
                t = threading.Thread(target=deliver, args=(job, progress, drive, token))  # the GPU does not wait for the uploads
                t.start()
                deliveries.append(t)
            else:
                failed.append(job["run"])
                say(f"GPU {gpu}: FAILED {job['run']}: {last_error(job['run'])} - see {RUNS / job['run'] / 'console.log'}")


# -------------------------------------------------------------------------------------------------------------- status

def log_tail(run: str, size: int = 30000) -> str:
    p = RUNS / run / "console.log"
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        f.seek(max(0, p.stat().st_size - size))
        return f.read().decode("utf-8", errors="replace")


def last_error(run: str) -> str:
    lines = [l.strip() for l in log_tail(run).replace("\r", "\n").splitlines() if l.strip()]
    errors = [l for l in lines if re.search(r"Error|error:|Killed|out of memory", l)]
    return (errors[-1] if errors else lines[-1] if lines else "no output")[:240]


def step_progress(run: str) -> dict:
    """The newest line of the training progress bar, ` 42%|████▏     | 1234/2900 [35:10<48:02,  1.71s/it]`, and the newest log line."""
    tail = log_tail(run)
    logs = re.findall(r"\d\d:\d\d:\d\d (?:INFO|WARNING|ERROR) ([^\r\n]*)", tail)
    out = {"log": logs[-1][:150] if logs else ""}
    bars = list(re.finditer(r"(\d+)%\|[^|\r\n]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+)", tail))
    if bars and bars[-1].start() > tail.rfind(" windows ("):  # a bar of this attempt: that line is logged before the first step
        pct, step, total, elapsed, left = bars[-1].groups()
        out.update(percent=int(pct), step=int(step), total=int(total), elapsed=elapsed, left=left)
    return out


def launcher_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        return "launch.py" in Path(f"/proc/{pid}/cmdline").read_text()
    except OSError:
        if Path("/proc").exists():
            return False
        try:
            os.kill(pid, 0)  # no /proc (macOS)
            return True
        except OSError:
            return False


def collect(plan: dict, plan_path: Path) -> dict:
    """The state of every run of the launch: what --status prints and --status --json returns."""
    sp = plan_path.with_suffix(".state.json")
    st = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    alive = launcher_alive(st.get("pid"))
    rows = []
    for j in plan["jobs"]:
        s, c = st.get("runs", {}).get(j["run"], {}), state(j)
        hist_file = RUNS / j["run"] / "dev_history.json"
        dev = [round(h["final_score"], 4) for h in json.loads(hist_file.read_text(encoding="utf-8"))] if hist_file.exists() else []
        row = dict(run=j["run"], gpu=s.get("gpu"), dev=dev, drive=s.get("drive"), hf=s.get("hf"))
        if trained(j) or s.get("state") == "on_drive":
            row.update(state="done", minutes=c.get("train_minutes"))
        elif s.get("state") == "running":
            p = step_progress(j["run"])
            # the launcher is gone but the run was not finished: the machine was restarted or the session was killed
            row.update(state="running" if alive else "stopped", minutes=(time.time() - s["started"]) / 60, **p)
            if p.get("total"):
                row["epoch"] = round(p["step"] / p["total"] * j["epochs"], 2)
        elif s.get("state") == "failed":
            row.update(state="FAILED", error=last_error(j["run"]))
        else:
            row["state"] = "waiting"
        rows.append(row)
    counts = {k: sum(r["state"] == k for r in rows) for k in ("done", "running", "waiting", "FAILED", "stopped")}
    delivering = sum("uploading" in (r["drive"], r["hf"]) for r in rows)
    return dict(launch=plan["launch"], tag=plan["tag"], folder=plan["folder"], created=plan["created"], alive=alive, counts=counts,
                delivering=delivering, finished=not alive and counts["done"] == len(rows), runs=rows)


def status(info: dict) -> None:
    c = info["counts"]
    print(f"launch {info['launch']} on {info['folder']} (started {info['created']}): " + ", ".join(f"{n} {k}" for k, n in c.items() if n)
          + ("; uploads in progress" if info["delivering"] else "")
          + ("; the launcher is running" if info["alive"] else "; FINISHED" if info["finished"] else "; the launcher is NOT running"))
    for r in info["runs"]:
        dev = " ".join(f"e{i} {s:.4f}" for i, s in enumerate(r["dev"], 1))
        where = f"GPU {r['gpu']}" if r["state"] in ("running", "stopped") and r["gpu"] is not None else ""
        if r["state"] == "done":
            note = f"dev F1 {dev or '-'}; {r.get('minutes') or 0:.0f} min" + "".join(f"; {k} {r[n]}" for k, n in (("Drive", "drive"), ("HF", "hf")) if r[n])
        elif r["state"] in ("running", "stopped"):
            note = (f"{r['percent']:3d}% step {r['step']}/{r['total']}, epoch {r['epoch']}, {r['elapsed']} done, ~{r['left']} left"
                    if "percent" in r else f"preparing, {r['minutes']:.0f} min: {r.get('log', '')}") + (f" | dev F1 {dev}" if dev else "")
        else:
            note = r.get("error", "")
        print(f"  {r['state']:8s} {where:6s} {r['run']:58s} {note}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS), help="the dataset: base = data/silver, large = data/silver_large")
    ap.add_argument("--launch", default="", help="continue this launch (e.g. 2026-10-09_r1); empty = a new launch")
    ap.add_argument("--models", default="", help="comma-separated names from MODELS (default: all)")
    ap.add_argument("--top-n", type=int, default=0, help="besides MODELS, add this many of the best models of the test history")
    ap.add_argument("--seeds", default="", help="comma-separated (default: 42,43)")
    ap.add_argument("--n-docs", type=int, default=0, help="train documents per run (0 = all; e.g. 2000 for a smoke run)")
    ap.add_argument("--gpus", default="", help="comma-separated GPU indices (default: all)")
    ap.add_argument("--no-publish", action="store_true", help="do not publish the trained models to the Hugging Face Hub")
    ap.add_argument("--no-drive", action="store_true", help="do not read or write Google Drive (a smoke run that should leave no trace there)")
    ap.add_argument("--plan-only", action="store_true", help="make (or read) the plan of the launch, print it and stop")
    ap.add_argument("--status", action="store_true", help="print the progress of the runs of --launch (default: the latest launch of the variant)")
    ap.add_argument("--json", action="store_true", help="with --status: the progress as JSON")
    a = ap.parse_args()

    v = VARIANTS[a.variant]
    launches = RUNS / "launches"
    launches.mkdir(parents=True, exist_ok=True)
    plan_file = lambda launch: launches / f"{launch}_{v['tag']}.json"
    if a.status:
        plans = sorted(launches.glob(f"*_{v['tag']}.json"), key=lambda p: p.stat().st_mtime)
        p = plan_file(a.launch) if a.launch else plans[-1] if plans else None
        if p is None or not p.exists():
            return print("{}" if a.json else f"no launch of {v['tag']} yet")
        info = collect(json.loads(p.read_text(encoding="utf-8")), p)
        return print(json.dumps(info, ensure_ascii=False)) if a.json else status(info)
    drive = not a.no_drive and drive_on()
    if a.launch:
        if not plan_file(a.launch).exists() and drive:  # a launch that was started elsewhere, e.g. in Colab
            rclone("copy", f"{DRIVE}runs/launches/{plan_file(a.launch).name}", str(launches))
        if not plan_file(a.launch).exists():
            raise SystemExit(f"no launch {a.launch} of {v['tag']}: " + ", ".join(sorted(p.stem for p in launches.glob(f"*_{v['tag']}.json"))))
        plan = json.loads(plan_file(a.launch).read_text(encoding="utf-8"))  # the models of a launch stay the ones chosen when it started
        say(f"continuing launch {a.launch} (started {plan['created']}); --models, --top-n, --seeds and --n-docs are ignored")
    else:
        plan = new_plan(a, v, drive)
        plan_file(plan["launch"]).write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
        say(f"new launch {plan['launch']} -> {plan_file(plan['launch'])}")
    if drive:
        rclone("copy", str(plan_file(plan["launch"])), f"{DRIVE}runs/launches")
    show(plan)
    if a.plan_only:
        return
    token = "" if a.no_publish else hf_token()
    progress = Progress(plan_file(plan["launch"]).with_suffix(".state.json"))
    jobs = sorted(plan["jobs"], key=cost, reverse=True)
    sha = file_sha1(DATA / plan["folder"] / "train.parquet")
    todo: queue.Queue = queue.Queue()
    deliveries: list[threading.Thread] = []
    for j in jobs:
        there = drive_config(j["run"]) if drive and not trained(j) else {}
        if there.get("status") == "done":
            say(f"{j['run']}: already trained and saved on Drive - skipped")
            progress.mark(j["run"], state="on_drive", drive="saved")
        elif not trained(j):
            progress.mark(j["run"], state="waiting", gpu=None)
            todo.put(j)
        elif state(j).get("data_version", {}).get("train.parquet") != sha:
            # the data changed since the run was trained: the saved model is kept
            say(f"{j['run']}: trained on other data ({j['data']} changed) - NOT overwritten; start a new launch to train on the new data")
            progress.mark(j["run"], state="done")
        else:
            say(f"{j['run']}: already trained - skipped")
            progress.mark(j["run"], state="done")
            deliveries.append(threading.Thread(target=deliver, args=(j, progress, drive, token)))  # what an earlier attempt did not deliver
            deliveries[-1].start()
    gpus = [int(g) for g in a.gpus.split(",") if g.strip()] or gpu_ids()
    if not gpus:
        raise SystemExit("no GPU found (nvidia-smi)")
    full = not any(j["n_docs"] for j in jobs)
    say(f"launch {plan['launch']}: {todo.qsize()} of {len(jobs)} runs to train on GPUs {gpus}; "
        + (f"every run is copied to {DRIVE}runs/" if drive else "Google Drive is NOT connected: the runs stay on this machine") + "; "
        + ("every model is published to the Hugging Face Hub" if token and full else "nothing is published to Hugging Face"
           + ("" if full else " (runs on a part of the data)") + ("" if token or a.no_publish else " (no token: scripts/vast.sh hf)")))
    total, failed, t0 = todo.qsize(), [], time.time()
    threads = [threading.Thread(target=worker, args=(g, todo, total, failed, progress, deliveries, drive, token)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for t in deliveries:
        t.join()
    say(f"launch {plan['launch']}: {total - len(failed)} of {total} runs trained in {(time.time() - t0) / 60:.0f} min"
        + (f"; FAILED: {', '.join(failed)} - run again with --launch {plan['launch']}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
