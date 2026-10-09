#!/usr/bin/env python3
"""Score runs on the organizers' test set and rewrite the Markdown leaderboard with the history of runs.

    uv run scripts/leaderboard.py --drive /content/drive/MyDrive/ai-camp --name leaderboard --runs wechsel_base,xlmr_base

A run is predicted once: `<drive>/runs/<run>/test_result.json` is reused until the run is retrained, the test file
changes or the decoder changes. Every new evaluation is appended to `<drive>/runs/test_history.jsonl`;
`<drive>/runs/<name>.md` is rebuilt from that history.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from pseudo import DATA  # noqa: E402
from pseudo.data import DECODE_VERSION  # noqa: E402
from pseudo.leaderboard import append_history, load_history, record, render, test_version  # noqa: E402


def evaluate(test: Path, sub: Path, *flags: str) -> str:
    return subprocess.run([sys.executable, str(DATA / "evaluate.py"), str(test), str(sub), *flags], capture_output=True, text=True).stdout


def run_key(cfg: dict) -> str:
    """Identifies the trained model a test result belongs to."""
    return str(cfg.get("fingerprint") or cfg.get("train_minutes"))


def mtime(path: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", required=True, help="folder that holds runs/ (MyDrive/ai-camp)")
    ap.add_argument("--name", default="leaderboard", help="the Markdown file is <drive>/runs/<name>.md")
    ap.add_argument("--runs", default="", help="comma-separated run names for the leaderboard table")
    ap.add_argument("--all", action="store_true", help="add every run that has a trained model on Drive")
    ap.add_argument("--baseline", default=None, help="run to show the difference to")
    ap.add_argument("--local", default=str(HERE / "runs"), help="runs trained in this session")
    ap.add_argument("--test", default=str(DATA / "test" / "court_cases.jsonl"))
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    remote, local, test = Path(a.drive) / "runs", Path(a.local), Path(a.test)
    if not test.exists():
        raise SystemExit(f"test set missing: {test}")
    remote.mkdir(parents=True, exist_ok=True)
    test_id = test_version(test)
    eval_key = lambda key: f"{key}:{test_id}:d{DECODE_VERSION}"  # the model, the test file and the decoder it was scored with
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    if a.all:
        runs = sorted(d.name for d in remote.iterdir() if (d / "model").is_dir()) + runs
    runs = list(dict.fromkeys(runs))
    history = load_history(remote)
    for run in runs:
        src = remote / run
        # a run trained in this session is under the local runs/, an older one only on Drive
        loc = next((d for d in (local / run, src) if (d / "model").is_dir() and (d / "config.json").exists()), None)
        if loc is None:
            print(f"{run}: no trained model", flush=True)
            continue
        cfg = json.loads((loc / "config.json").read_text(encoding="utf-8"))
        key = run_key(cfg)
        cached = src / "test_result.json"
        res = json.loads(cached.read_text(encoding="utf-8")) if cached.exists() else {}
        ts = mtime(cached) if cached.exists() else None
        if res.get("eval_key") != eval_key(key):  # not scored yet, retrained, another test file or decoder: predict and keep the result on Drive
            print(f"{run}: predicting the test set with {loc / 'model'}", flush=True)
            sub = local / run / "test_sub.jsonl"
            sub.parent.mkdir(parents=True, exist_ok=True)
            src.mkdir(parents=True, exist_ok=True)
            sub.unlink(missing_ok=True)
            subprocess.run([sys.executable, str(HERE / "scripts" / "predict.py"), "--model", str(loc / "model"), "--input", str(test),
                            "--output", str(sub), "--batch", str(a.batch)])
            if not sub.exists():
                print(f"{run}: prediction failed", flush=True)
                continue
            res = {**json.loads(evaluate(test, sub, "--json").strip().splitlines()[-1]), "run_fingerprint": key,
                   "eval_key": eval_key(key), "test": test_id, "decode": DECODE_VERSION}
            (src / "test_sub.jsonl").write_bytes(sub.read_bytes())
            (src / "test_result.txt").write_text(evaluate(test, sub), encoding="utf-8")
            cached.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            ts = time.strftime("%Y-%m-%d %H:%M")
        if not any(h["run"] == run and h.get("key") == res["eval_key"] for h in history):
            rec = record(run, cfg, res, res["eval_key"], ts)
            append_history(remote, rec)
            history.append(rec)
        print(f"{run}: test macro F1 {res['final_score']:.4f}", flush=True)
    # runs outside this leaderboard that were scored earlier still belong to the history
    for cached in sorted(remote.glob("*/test_result.json")):
        run, cfg_path = cached.parent.name, cached.parent / "config.json"
        if run in runs or not cfg_path.exists():
            continue
        cfg, res = json.loads(cfg_path.read_text(encoding="utf-8")), json.loads(cached.read_text(encoding="utf-8"))
        key = res.get("eval_key") or run_key(cfg)  # results cached before the test file changed carry the fingerprint only
        if res.get("run_fingerprint") == run_key(cfg) and not any(h["run"] == run and h.get("key") == key for h in history):
            rec = record(run, cfg, res, key, mtime(cached))
            append_history(remote, rec)
            history.append(rec)
    history.sort(key=lambda r: r["ts"])
    cpu = {}
    for p in sorted(remote.glob("cpu_timing_*.json")):
        cpu.update(json.loads(p.read_text(encoding="utf-8")))
    out = remote / f"{a.name}.md"
    out.write_text(render(history, runs, a.name, a.baseline, cpu, test=test_id, decode=DECODE_VERSION), encoding="utf-8")
    print(f"saved {out}; history in {remote / 'test_history.jsonl'}")


if __name__ == "__main__":
    main()
