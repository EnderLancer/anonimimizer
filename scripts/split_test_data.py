#!/usr/bin/env python3
"""Ділить тестовий набір на вхідні дані для рішення та очікуваний результат.

Запуск:
    python scripts/split_test_data.py data/test/court_cases.jsonl inputs.jsonl

inputs.jsonl містить лише id та original, тож рішення не бачить правильних анотацій.
Очікуваний результат — сам файл data/test/court_cases.jsonl, його читає data/evaluate.py.
"""
import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    n = 0
    with open(src, encoding="utf-8-sig") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            fout.write(json.dumps({"id": row["id"], "original": row["original"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} рішень записано у {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
