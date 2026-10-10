#!/usr/bin/env python3
"""Оцінка на OOD-наборі (data/ood/ood_cases.jsonl) з розбивкою за джерелом.

    python evaluate_ood.py ood/ood_cases.jsonl results/ood_full.jsonl
    python evaluate_ood.py ood/ood_cases.jsonl results/ood_model.jsonl results/ood_full.jsonl --names model,full
    python evaluate_ood.py ood/ood_cases.jsonl results/ood_full.jsonl --by subset --bootstrap 500 --json out.json

Метрики ті самі, що в evaluate.py (NER macro F1 з IoU >= 0,5, strict F1, deanonymization risk by name, B³),
порахувані на всьому наборі та окремо на кожній групі (поле gold-рядка --by: source за замовчуванням).
Додатково: хибні спрацювання на 1 000 символів у документах, де gold не має сутностей цього типу (надмірне
спрацювання на чистому тексті), recall за підвидом сутності (поле kind у gold) і бутстреп-інтервал macro F1.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402

TYPES = ev.TYPES


def _preds_for(pred_rows: list[dict], ids: set[str]) -> list[dict]:
    return [r for r in pred_rows if str(r["id"]) in ids]


def groups_of(gold: list[dict], by: str) -> list[str]:
    seen = []
    for g in gold:
        v = str(g.get(by, "—"))
        if v not in seen:
            seen.append(v)
    return seen


def false_positives(gold: list[dict], preds: dict[str, dict]) -> dict[str, dict]:
    """Хибні спрацювання за типом на документах без gold-сутностей цього типу, на 1 000 символів."""
    chars = Counter()
    fps = Counter()
    docs = Counter()
    for g in gold:
        gs = ev.spans_from_annotations(g["annotations"], len(g["original"]), g["id"])
        p = preds.get(str(g["id"]))
        ps = ev.spans_from_annotations(p["annotations"], len(g["original"]), g["id"]) if p and "annotations" in p else []
        for t in TYPES:
            if any(s.type == t for s in gs):
                continue
            docs[t] += 1
            chars[t] += len(g["original"])
            fps[t] += sum(1 for s in ps if s.type == t)
    return {t: {"docs": docs[t], "false_positives": fps[t], "per_1k_chars": 1000 * fps[t] / chars[t] if chars[t] else 0.0}
            for t in TYPES}


def recall_by_kind(gold: list[dict], preds: dict[str, dict]) -> dict[str, dict]:
    """Частка знайдених gold-фрагментів (IoU >= 0,5, той самий тип) за підвидом kind."""
    tp, total = Counter(), Counter()
    for g in gold:
        gs = ev.spans_from_annotations(g["annotations"], len(g["original"]), g["id"])
        p = preds.get(str(g["id"]))
        ps = ev.spans_from_annotations(p["annotations"], len(g["original"]), g["id"]) if p and "annotations" in p else []
        matched = {i for i, _ in ev.match(gs, ps, strict=False)}
        for i, (a, s) in enumerate(zip(g["annotations"], gs)):
            key = f"{s.type}/{a.get('kind') or '—'}"
            total[key] += 1
            tp[key] += i in matched
    return {k: {"total": total[k], "found": tp[k], "recall": tp[k] / total[k]} for k in sorted(total)}


def bootstrap(gold: list[dict], pred_rows: list[dict], n: int, seed: int) -> tuple[float, float]:
    """95 % інтервал macro F1 за бутстрепом по документах."""
    rng = random.Random(seed)
    scores = []
    for _ in range(n):
        rows = [rng.choice(gold) for _ in gold]
        scores.append(ev.evaluate(rows, pred_rows)["final_score"])
    scores.sort()
    return scores[int(0.025 * n)], scores[min(n - 1, int(0.975 * n))]


def evaluate_all(gold: list[dict], submissions: dict[str, list[dict]], by: str, n_boot: int = 0, seed: int = 0) -> dict:
    out = {"by": by, "groups": groups_of(gold, by), "submissions": {}}
    for name, pred_rows in submissions.items():
        preds = {str(r["id"]): r for r in pred_rows}
        res = {"overall": ev.evaluate(gold, pred_rows), "by_group": {}}
        for grp in out["groups"]:
            g_rows = [g for g in gold if str(g.get(by, "—")) == grp]
            res["by_group"][grp] = ev.evaluate(g_rows, _preds_for(pred_rows, {str(g["id"]) for g in g_rows}))
        res["false_positives"] = false_positives(gold, preds)
        res["recall_by_kind"] = recall_by_kind(gold, preds)
        if n_boot:
            res["macro_f1_ci95"] = bootstrap(gold, pred_rows, n_boot, seed)
        out["submissions"][name] = res
    return out


def _f(x: float) -> str:
    return f"{x:.3f}".lstrip("0") if x < 1 else "1.0"


def _cell(per: dict) -> str:
    return f"{_f(per['f1'])} ({_f(per['precision'])} / {_f(per['recall'])})"


def _row(label: str, r: dict) -> str:
    rel = r["ner_relaxed"]
    return (f"| {label} | {r['n_docs']} | " + " | ".join(_cell(rel["per_type"][t]) for t in TYPES) +
            f" | **{_f(rel['macro']['f1'])}** | {_f(r['ner_strict']['macro']['f1'])} | "
            f"{_f(r['deanonymization_risk_by_name'])} ({r['persons_leaked']}/{r['persons_total']}) | {_f(r['consistency_b3']['f1'])} |")


def render(result: dict) -> str:
    lines = []
    head = ("| Група | Док. | " + " | ".join(TYPES) + " | **Середнє F1** | Strict F1 | Ризик (by name) | B³ |\n"
            "|---|---|" + "---|" * (len(TYPES) + 4))
    for name, res in result["submissions"].items():
        lines += [f"### {name}", "", head]
        for grp in result["groups"]:
            lines.append(_row(grp, res["by_group"][grp]))
        lines.append(_row("усього", res["overall"]))
        if "macro_f1_ci95" in res:
            lo, hi = res["macro_f1_ci95"]
            lines.append(f"\n95 % бутстреп-інтервал середнього F1: {_f(lo)}–{_f(hi)}.")
        fp = res["false_positives"]
        lines += ["", "Хибні спрацювання на документах без сутностей типу (на 1 000 символів): " +
                  ", ".join(f"{t} {fp[t]['per_1k_chars']:.2f} ({fp[t]['false_positives']} у {fp[t]['docs']} док.)" for t in TYPES) + "."]
        rk = res["recall_by_kind"]
        lines += ["", "Recall за підвидом: " + ", ".join(f"{k} {_f(v['recall'])} ({v['found']}/{v['total']})" for k, v in rk.items()) + ".", ""]
        for w in res["overall"].get("warnings", []):
            lines.append(f"Увага: {w}")
    if len(result["submissions"]) > 1:
        lines += ["### Порівняння (середнє F1)", "", "| Submission | " + " | ".join(result["groups"]) + " | усього |",
                  "|---|" + "---|" * (len(result["groups"]) + 1)]
        for name, res in result["submissions"].items():
            lines.append(f"| {name} | " + " | ".join(_f(res["by_group"][g]["final_score"]) for g in result["groups"]) +
                         f" | **{_f(res['overall']['final_score'])}** |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold")
    ap.add_argument("submissions", nargs="+")
    ap.add_argument("--names", default=None, help="назви submission через кому (за замовчуванням — назви файлів)")
    ap.add_argument("--by", default="source", help="поле gold-рядка для груп (source, subset, origin)")
    ap.add_argument("--bootstrap", type=int, default=0, help="кількість бутстреп-вибірок для інтервалу macro F1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="записати повний результат у файл")
    a = ap.parse_args()
    gold = ev.load_jsonl(a.gold)
    names = a.names.split(",") if a.names else [os.path.splitext(os.path.basename(p))[0] for p in a.submissions]
    if len(names) != len(a.submissions):
        raise SystemExit("--names: кількість назв не збігається з кількістю submission")
    subs = {n: ev.load_jsonl(p) for n, p in zip(names, a.submissions)}
    result = evaluate_all(gold, subs, a.by, a.bootstrap, a.seed)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    print(render(result))


if __name__ == "__main__":
    main()
