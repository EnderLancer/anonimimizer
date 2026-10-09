#!/usr/bin/env python3
"""Build a Markdown (+ HTML) report over runs/*: data version, leaderboard, per-type metrics, loss curves,
error samples on dev, log tails. Meant to be read by a human or pasted to an assistant for analysis.

    uv run scripts/report.py --runs runs --out runs/report.md [--errors-run wechsel_base]
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pseudo import DATA, SILVER, TYPES  # noqa: E402

sys.path.insert(0, str(DATA))


def load(run: Path) -> dict | None:
    cfg = run / "config.json"
    if not cfg.exists():
        return None
    d = {"name": run.name, "config": json.loads(cfg.read_text(encoding="utf-8"))}
    hist = run / "dev_history.json"
    d["history"] = json.loads(hist.read_text(encoding="utf-8")) if hist.exists() else []
    d["best"] = max(d["history"], key=lambda h: h["final_score"]) if d["history"] else None
    tl = run / "train_log.json"
    d["train_log"] = json.loads(tl.read_text(encoding="utf-8")) if tl.exists() else []
    log = run / "train.log"
    d["log_tail"] = log.read_text(encoding="utf-8").splitlines()[-25:] if log.exists() else []
    return d


def errors_on_dev(run: Path, best_epoch: int, per_type: int = 10) -> dict:
    import evaluate as ev
    from pseudo.train import file_sha1

    gold_path = SILVER / "dev.jsonl"
    sub = run / f"dev_submission_epoch{best_epoch}.jsonl"
    if not gold_path.exists() or not sub.exists():
        return {}
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    run_dev = (cfg.get("data_version") or {}).get("dev.parquet")
    if (SILVER / "dev.parquet").exists() and run_dev and run_dev != file_sha1(SILVER / "dev.parquet"):
        return {"_note": f"dev data changed since this run (run dev {run_dev}, current {file_sha1(SILVER / 'dev.parquet')}); error dump skipped"}
    gold = {str(r["id"]): r for r in ev.load_jsonl(str(gold_path))}
    pred = {str(r["id"]): r for r in ev.load_jsonl(str(sub))}
    fn, fp = collections.defaultdict(list), collections.defaultdict(list)
    for k, g in gold.items():
        if k not in pred:
            continue
        o = g["original"]
        G = ev.spans_from_annotations(g["annotations"], len(o), k)
        try:
            P = ev.spans_from_annotations(pred[k]["annotations"], len(o), k)
        except ev.FormatError:
            continue  # stale submission for a different dev text
        pairs = ev.match(G, P, False)
        mg, mp = {i for i, _ in pairs}, {j for _, j in pairs}
        ctx = lambda s: (o[max(0, s.start - 45): s.start] + "[[" + o[s.start:s.end] + "]]" + o[s.end: s.end + 30]).replace("\n", " ")
        for i, s in enumerate(G):
            if i not in mg:
                fn[s.type].append(ctx(s))
        for j, s in enumerate(P):
            if j not in mp:
                fp[s.type].append(ctx(s))
    return {t: {"fn": len(fn[t]), "fp": len(fp[t]), "fn_examples": fn[t][:per_type], "fp_examples": fp[t][:per_type]} for t in TYPES}


def build(runs_dir: Path, errors_run: str | None) -> str:
    runs = [d for d in (load(p) for p in sorted(runs_dir.iterdir()) if p.is_dir()) if d]
    runs.sort(key=lambda d: -(d["best"]["final_score"] if d["best"] else -1))
    L = [f"# Pseudonymizer — training report", "", f"_generated {time.strftime('%Y-%m-%d %H:%M')}_", ""]
    dvs = {json.dumps(d["config"].get("data_version"), sort_keys=True) for d in runs}
    L += ["## Data", ""]
    for dv in sorted(dvs):
        L.append(f"- data version: `{dv}`")
    L += ["", "## Leaderboard (dev, official `evaluate.py`, IoU ≥ 0.5)", "",
          "| run | model | status | best ep | **macro F1** | " + " | ".join(t for t in TYPES) + " | strict | risk | B³ | loss | min | fingerprint |",
          "|" + "---|" * 14]
    for d in runs:
        c, b = d["config"], d["best"]
        if not b:
            L.append(f"| {d['name']} | {c.get('model')} | {c.get('status')} | — | — |" + " — |" * 4 + " — | — | — | — | — | {c.get('fingerprint','')} |")
            continue
        m = b["metrics"]
        per = m["ner_relaxed"]["per_type"]
        L.append(f"| {d['name']} | {c.get('model')} | {c.get('status')} | {int(b['epoch'])} | **{m['final_score']:.4f}** | "
                 + " | ".join(f"{per[t]['f1']:.3f}" for t in TYPES)
                 + f" | {m['ner_strict']['macro']['f1']:.4f} | {m['deanonymization_risk_by_name']:.3f} | {m['consistency_b3']['f1']:.3f}"
                 f" | {c.get('train_loss', float('nan')):.4f} | {c.get('train_minutes', float('nan')):.0f} | `{c.get('fingerprint', '')}` |")
    L += ["", "## Runs", ""]
    for d in runs:
        c = d["config"]
        L += [f"### {d['name']}", "",
              f"- model `{c.get('model')}`, lr {c.get('lr')}, batch {c.get('batch')} × accum {c.get('grad_accum')}, epochs {c.get('epochs')}, "
              f"window {c.get('window')}/{c.get('stride')}, masks {c.get('use_masks')}, oversample НОМЕР {c.get('oversample_number')}",
              f"- docs {c.get('n_docs_used')}, windows {c.get('windows')}, device {c.get('gpu', c.get('device'))}, "
              f"{c.get('train_minutes', 0):.1f} min, started {c.get('started')}, finished {c.get('finished')}", ""]
        if d["history"]:
            L += ["| epoch | macro F1 | " + " | ".join(t[:3] for t in TYPES) + " | strict | P | R | risk |", "|" + "---|" * 10]
            for h in d["history"]:
                m = h["metrics"]
                per = m["ner_relaxed"]["per_type"]
                L.append(f"| {int(h['epoch'])} | {m['final_score']:.4f} | " + " | ".join(f"{per[t]['f1']:.3f}" for t in TYPES)
                         + f" | {m['ner_strict']['macro']['f1']:.4f} | {m['ner_relaxed']['macro']['precision']:.3f} | {m['ner_relaxed']['macro']['recall']:.3f} | {m['deanonymization_risk_by_name']:.3f} |")
            L.append("")
            per = d["best"]["metrics"]["ner_relaxed"]["per_type"]
            L += ["Best epoch, per type (TP / FP / FN): " + "; ".join(f"{t} {per[t]['tp']}/{per[t]['fp']}/{per[t]['fn']}" for t in TYPES), ""]
        if d["train_log"]:
            losses = [(x["epoch"], x["loss"]) for x in d["train_log"] if "loss" in x]
            step = max(1, len(losses) // 12)
            L += ["Loss curve (epoch: loss): " + ", ".join(f"{e:.2f}: {l:.4f}" for e, l in losses[::step]), ""]
        if d["log_tail"]:
            L += ["<details><summary>train.log tail</summary>", "", "```", *d["log_tail"], "```", "", "</details>", ""]
    target = errors_run or (runs[0]["name"] if runs and runs[0]["best"] else None)
    if target:
        d = next((r for r in runs if r["name"] == target), None)
        if d and d["best"]:
            err = errors_on_dev(runs_dir / target, int(d["best"]["epoch"]))
            if err.get("_note"):
                L += [f"## Error analysis on dev — `{target}`", "", f"_{err['_note']}_", ""]
            elif err:
                L += [f"## Error analysis on dev — `{target}` (best epoch {int(d['best']['epoch'])})", ""]
                for t in TYPES:
                    e = err[t]
                    L += [f"### {t}: FN {e['fn']}, FP {e['fp']}", ""]
                    L += [f"- FN `{x}`" for x in e["fn_examples"]]
                    L += [f"- FP `{x}`" for x in e["fp_examples"]]
                    L.append("")
    L += ["## How to read", "",
          "- macro F1 = mean of the four per-type F1 (IoU ≥ 0.5); this is the competition score.",
          "- high precision + low recall on a type ⇒ the model does not recognise formats absent from the training data (fix the generator);",
          "  low precision ⇒ the model over-predicts (fix labels / add negatives / post-processing).",
          "- dev is synthetic from the same generator as train, so it is optimistic; compare with a logged test run before concluding.",
          "- fingerprint = sha1(model, hyper-parameters, data version): identical fingerprint ⇒ identical experiment, skipped on rerun."]
    return "\n".join(L) + "\n"


def to_html(md: str) -> str:
    try:
        import markdown  # type: ignore

        body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    except ImportError:
        body = "<pre>" + html.escape(md) + "</pre>"
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Pseudonymizer training report</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.45}"
            "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:3px 6px}th{background:#f3f3f3}"
            "code{background:#f6f6f6;padding:1px 3px}pre{background:#f6f6f6;padding:8px;overflow-x:auto}</style></head><body>"
            + body + "</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/report.md")
    ap.add_argument("--errors-run", default=None, help="run to dump dev errors for (default: best)")
    a = ap.parse_args()
    md = build(Path(a.runs), a.errors_run)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    out.with_suffix(".html").write_text(to_html(md), encoding="utf-8")
    print(md[:3000])
    print(f"\n... written {out} and {out.with_suffix('.html')}")


if __name__ == "__main__":
    main()
