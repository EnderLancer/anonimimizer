#!/usr/bin/env python3
"""Етап RegEx: документи -> анотації.

Запуск:
    python solution/regex_annotate.py inputs.jsonl regex.jsonl

Вхід — JSON Lines з полями id та original. Вихід — анотації у форматі anonymizer/annotations.py,
їх можна одразу оцінити: python data/evaluate.py data/test/court_cases.jsonl regex.jsonl
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from anonymizer import mapping, read_jsonl, regex_stage, write_jsonl  # noqa: E402
from anonymizer.annotations import document_row  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    t = time.time()
    rows = []
    for doc in read_jsonl(src):
        hidden = regex_stage.find(doc["original"])
        rows.append(document_row(doc["id"], doc["original"], hidden, mapping.assign(hidden), {"stage": "regex"}))
    write_jsonl(dst, rows)
    print(f"RegEx: {len(rows)} документів, {sum(len(r['annotations']) for r in rows)} анотацій за {time.time() - t:.0f} с",
          file=sys.stderr)


if __name__ == "__main__":
    main()
