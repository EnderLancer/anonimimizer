"""Test-set leaderboard with run history.

Every evaluation of a run on the organizers' test set is one line of `runs/test_history.jsonl` (append-only).
The Markdown file is a view of that history: the current leaderboard, how the best score moved, each run across
its retrains, every evaluation with what changed, and per-type details.
"""
from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

from pseudo import TYPES

HISTORY = "test_history.jsonl"
LEGACY_LOG = "test_runs.log"
PARAMS = ("epochs", "lr", "batch", "grad_accum", "window", "stride", "use_masks", "oversample_number", "n_docs")
BAR_LO, BAR_HI, BAR_W = 0.5, 1.0, 20
_EIGHTHS = " ▏▎▍▌▋▊▉"
_LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) (\S+) (.*?)(\d\.\d{4})\s*$")


def record(run: str, cfg: dict, res: dict, key: str, ts: str) -> dict:
    """One history line: what was evaluated (run config) and how it scored (`evaluate.py --json` output)."""
    rel, strict = res["ner_relaxed"], res["ner_strict"]
    return {
        "ts": ts, "run": run, "model": cfg.get("model"), "key": key, "data": cfg.get("data_version"),
        "params": {k: cfg.get(k) for k in PARAMS}, "train_minutes": cfg.get("train_minutes"),
        "score": res["final_score"], "strict": strict["macro"]["f1"], "precision": rel["macro"]["precision"],
        "recall": rel["macro"]["recall"], "risk": res["deanonymization_risk_by_name"], "b3": res["consistency_b3"]["f1"],
        "per_type": {t: {**{k: rel["per_type"][t][k] for k in ("f1", "precision", "recall", "tp", "fp", "fn")},
                         "strict_f1": strict["per_type"][t]["f1"]} for t in TYPES},
    }


def parse_legacy_log(text: str) -> list[dict]:
    """`test_runs.log` lines (date, run, optional model and data version, macro F1 last) -> history records without per-type metrics."""
    out = []
    for line in text.splitlines():
        m = _LOG_LINE.match(line.strip())
        if not m:
            continue
        ts, run, rest, score = m.groups()
        words = rest.split()
        d = re.search(r"data=(\{.*?\})", rest)
        try:
            data = ast.literal_eval(d.group(1)) if d else None
        except (ValueError, SyntaxError):
            data = None
        out.append({"ts": ts, "run": run, "model": words[0] if words and "/" in words[0] else None, "key": None,
                    "data": data, "score": float(score)})
    return out


def load_history(runs_dir: Path) -> list[dict]:
    """All evaluations, oldest first. The first call imports the old `test_runs.log` so earlier results are kept."""
    path = runs_dir / HISTORY
    if not path.exists():
        log = runs_dir / LEGACY_LOG
        rows = parse_legacy_log(log.read_text(encoding="utf-8")) if log.exists() else []
        if rows:
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return rows
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted(rows, key=lambda r: r["ts"])


def append_history(runs_dir: Path, rec: dict) -> None:
    with open(runs_dir / HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _same_evaluation(a: dict, b: dict) -> bool:
    if a["run"] != b["run"] or round(a["score"], 4) != round(b["score"], 4):
        return False
    if a.get("key") and b.get("key"):
        return a["key"] == b["key"]
    return a.get("data") is None or b.get("data") is None or a["data"] == b["data"]


def collapse(history: list[dict]) -> list[dict]:
    """Repeated evaluations of the same trained model count once: the earliest date with the richest metrics."""
    out, last = [], {}
    for r in history:
        prev = last.get(r["run"])
        if prev is not None and _same_evaluation(prev, r):
            if "per_type" not in prev and "per_type" in r:
                prev.update({k: v for k, v in r.items() if k != "ts"})
            continue
        r = dict(r)
        out.append(r)
        last[r["run"]] = r
    return out


def _data_key(data: dict | None) -> str | None:
    return json.dumps(data, sort_keys=True) if data else None


def bar(score: float) -> str:
    x = max(0.0, min(1.0, (score - BAR_LO) / (BAR_HI - BAR_LO))) * BAR_W
    full, part = int(x), int((x - int(x)) * 8)
    blocks = "█" * full + (_EIGHTHS[part] if part else "")
    return "`" + blocks + "░" * (BAR_W - len(blocks)) + "`"


def delta(new: float, old: float | None) -> str:
    if old is None:
        return "-"
    d = new - old
    if abs(d) < 5e-5:
        return "= 0.0000"
    return f"▲ +{d:.4f}" if d > 0 else f"▼ -{-d:.4f}"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def _bold_best(values: list[float | None], fmt: str, lowest: bool = False) -> list[str]:
    known = [v for v in values if v is not None]
    top = (min(known) if lowest else max(known)) if known else None
    return ["-" if v is None else f"**{v:{fmt}}**" if v == top and len(known) > 1 else f"{v:{fmt}}" for v in values]


def _model(r: dict) -> str:
    return r["model"].split("/")[-1] if r.get("model") else "-"


def _changes(r: dict, prev: dict | None, label) -> str:
    """What differs from the previous evaluation of the same run: data version and training settings."""
    if prev is None:
        return "first evaluation"
    out = []
    if label(prev) != label(r):
        out.append(f"data {label(prev)} → {label(r)}")
    a, b = prev.get("params") or {}, r.get("params") or {}
    out += [f"{k} {a[k]} → {b[k]}" for k in PARAMS if k in a and k in b and a[k] != b[k]]
    return "; ".join(out) or "retrained"


def render(history: list[dict], runs: list[str], name: str, baseline: str | None = None, cpu: dict | None = None) -> str:
    """Markdown view of the history. `runs` are the rows of the leaderboard; the history sections cover every run."""
    cpu = cpu or {}
    h = collapse(history)
    labels: dict[str, str] = {}
    for r in h:
        k = _data_key(r.get("data"))
        if k and k not in labels:
            labels[k] = f"D{len(labels) + 1}"
    label = lambda r: labels.get(_data_key(r.get("data")), "?")
    by_run: dict[str, list[dict]] = {}
    best = None
    for r in h:
        seq = by_run.setdefault(r["run"], [])
        r["_prev"] = seq[-1] if seq else None
        r["_gain"] = None if best is None else r["score"] - best
        r["_record"] = best is None or r["score"] > best
        best = r["score"] if r["_record"] else best
        seq.append(r)
    runs = list(dict.fromkeys(runs))
    current = sorted((by_run[x][-1] for x in runs if x in by_run), key=lambda r: -r["score"])
    f1 = lambda r, t: r["per_type"][t]["f1"] if "per_type" in r else None
    top = max(h, key=lambda r: r["score"]) if h else None

    L = [f"# {name} - organizers' test set", ""]
    if top:
        L += [f"_updated {time.strftime('%Y-%m-%d %H:%M')} · {len(by_run)} runs · {len(h)} evaluations · "
              f"best **{top['score']:.4f}** `{top['run']}`_", ""]
    L += ["Score = test macro F1 (official `evaluate.py`, IoU ≥ 0.5). Bars run from 0.50 to 1.00. ▲ / ▼ = change of the score. "
          "D1, D2, … = data versions, listed at the end.", ""]

    L += ["## Leaderboard", "", "Latest evaluation of each run, best first; the best value of a column is bold.", ""]
    base = by_run[baseline][-1] if baseline in by_run else None
    head = ["#", "run", "model", "test macro F1", "0.50 … 1.00", "vs previous"] + ([f"vs {baseline}"] if base else []) \
        + [t[:3] for t in TYPES] + ["data", "evaluated"]
    cols = [_bold_best([r["score"] for r in current], ".4f")] + [_bold_best([f1(r, t) for r in current], ".3f") for t in TYPES]
    rows = []
    for i, r in enumerate(current):
        rows.append([str(i + 1), r["run"], _model(r), cols[0][i], bar(r["score"]), delta(r["score"], r["_prev"] and r["_prev"]["score"])]
                    + ([delta(r["score"], base["score"]) if r is not base else "baseline"] if base else [])
                    + [c[i] for c in cols[1:]] + [label(r), r["ts"]])
    rows += [["-", x, "no result"] + ["-"] * (len(head) - 3) for x in runs if x not in by_run]
    L += _table(head, rows)

    L += ["## Progress", "", "Every evaluation that set a new best score, oldest first.", ""]
    L += _table(["date", "run", "model", "test macro F1", "0.50 … 1.00", "gain"],
                [[r["ts"], r["run"], _model(r), f"**{r['score']:.4f}**", bar(r["score"]),
                  "first" if r["_gain"] is None else delta(r["score"], r["score"] - r["_gain"])] for r in h if r["_record"]])

    L += ["## Runs over time", "", "Each run across its retrains (new data or settings), oldest → latest; the last 6 evaluations.", ""]
    rows = []
    for seq in sorted(by_run.values(), key=lambda s: -s[-1]["score"]):
        pts = [f"{p['score']:.4f} ({label(p)})" for p in seq[-6:]]
        pts[-1] = f"**{pts[-1]}**"
        rows.append([seq[-1]["run"], _model(seq[-1]), str(len(seq)), ("… → " if len(seq) > 6 else "") + " → ".join(pts),
                     delta(seq[-1]["score"], seq[0]["score"]) if len(seq) > 1 else "-"])
    L += _table(["run", "model", "evaluations", "test macro F1 (data)", "first → latest"], rows)

    L += ["## History", "", "Every evaluation, newest first. ★ = best score at that moment.", ""]
    rows = []
    for r in reversed(h):
        rows.append([r["ts"], r["run"], label(r), f"{r['score']:.4f}", delta(r["score"], r["_prev"] and r["_prev"]["score"])]
                    + ["-" if f1(r, t) is None else f"{f1(r, t):.3f}" for t in TYPES]
                    + [f"{r['strict']:.4f}" if "strict" in r else "-",
                       ("★ " if r["_record"] else "") + _changes(r, r["_prev"], label)])
    L += _table(["date", "run", "data", "test macro F1", "vs previous"] + [t[:3] for t in TYPES] + ["strict", "changed since the run's previous evaluation"], rows)

    full = [r for r in current if "per_type" in r]
    if full:
        L += ["## Details of the leaderboard runs", ""]
        cols = [_bold_best([r[k] for r in full], fmt, lowest) for k, fmt, lowest in
                (("strict", ".4f", False), ("precision", ".3f", False), ("recall", ".3f", False), ("risk", ".3f", True), ("b3", ".3f", False))]
        L += _table(["run", "strict F1", "precision", "recall", "deanon risk", "consistency B³", "CPU s/doc", "train min"],
                    [[r["run"]] + [c[i] for c in cols]
                     + [f"{cpu[r['run']]:.2f}" if r["run"] in cpu else "-", f"{r['train_minutes']:.0f}" if r.get("train_minutes") else "-"]
                     for i, r in enumerate(full)])
        L += ["Precision / recall per type:", ""]
        L += _table(["run"] + list(TYPES),
                    [[r["run"]] + [f"{r['per_type'][t]['precision']:.3f} / {r['per_type'][t]['recall']:.3f}" for t in TYPES] for r in full])

    if labels:
        L += ["## Data versions", ""]
        rows = []
        for k, lab in labels.items():
            seen = [r for r in h if _data_key(r.get("data")) == k]
            d = json.loads(k)
            rows.append([lab, f"`{d.get('train.parquet', '-')}`", f"`{d.get('dev.parquet', '-')}`", seen[0]["ts"], str(len(seen))])
        L += _table(["data", "train.parquet", "dev.parquet", "first evaluated", "evaluations"], rows)
    return "\n".join(L)
