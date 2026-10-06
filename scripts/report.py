#!/usr/bin/env python3
"""Перетворює JSON-результат data/evaluate.py на відсоток і Markdown-звіт.

Запуск:
    python scripts/report.py result.json [--min-score 50] [--compare RegEx=regex.json --compare Model=model.json]

Звіт друкується в stdout (у CI його дописують у $GITHUB_STEP_SUMMARY).
З --min-score скрипт завершується з помилкою, якщо результат нижчий за поріг у відсотках.
--compare додає таблицю з результатами окремих етапів поруч із підсумковим.
"""
import argparse
import json
import sys

TYPES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--min-score", type=float, default=None, help="мінімальний результат у відсотках")
    ap.add_argument("--compare", action="append", default=[], metavar="NAME=result.json",
                    help="результат окремого етапу для порівняння")
    args = ap.parse_args()
    res = load(args.result)

    score = 100 * res["final_score"]
    relaxed, strict = res["ner_relaxed"], res["ner_strict"]
    lines = [f"## Результат: **{score:.2f}%**", ""]
    if args.compare:
        runs = [(name, load(path)) for name, path in (c.split("=", 1) for c in args.compare)] + [("Разом", res)]
        lines += ["| Етап | Бал | " + " | ".join(f"{t} F1" for t in TYPES) + " | Deanon. risk |",
                  "|---|---|" + "---|" * len(TYPES) + "---|"]
        for name, r in runs:
            per = r["ner_relaxed"]["per_type"]
            lines.append(f"| {name} | {pct(r['final_score'])} | " + " | ".join(pct(per[t]["f1"]) for t in TYPES)
                         + f" | {pct(r['deanonymization_risk_by_name'])} |")
        lines += ["", "### Разом", ""]
    lines += [
        "| Метрика | Значення |",
        "|---|---|",
        f"| NER macro F1 (IoU ≥ 0.5) — підсумковий бал | {pct(relaxed['macro']['f1'])} |",
        f"| NER macro precision / recall | {pct(relaxed['macro']['precision'])} / {pct(relaxed['macro']['recall'])} |",
        f"| NER macro F1 (strict) | {pct(strict['macro']['f1'])} |",
        f"| Deanonymization risk (by name) | {pct(res['deanonymization_risk_by_name'])} "
        f"({res['persons_leaked']} з {res['persons_total']} осіб) |",
        f"| Consistency B³ F1 | {pct(res['consistency_b3']['f1'])} |",
        f"| Рішень у тестовому наборі | {res['n_docs']} |",
        "",
        "| Тип | P | R | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in TYPES:
        m = relaxed["per_type"][t]
        lines.append(f"| {t} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {m['tp']} | {m['fp']} | {m['fn']} |")
    if res["warnings"]:
        lines += ["", "<details><summary>Попередження</summary>", ""]
        lines += [f"- {w}" for w in res["warnings"][:50]]
        lines += ["", "</details>"]
    print("\n".join(lines))

    if args.min_score is not None and score < args.min_score:
        print(f"Результат {score:.2f}% нижчий за поріг {args.min_score:.2f}%", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
