#!/usr/bin/env python3
"""Етап RegEx: документи -> анотації.

Запуск:
    python solution/regex_annotate.py inputs.jsonl regex.jsonl

Вхід — JSON Lines з полями id та original. Вихід — анотації без номерів псевдонімів
у форматі anonymizer/annotations.py, далі їх обробляє solution/link.py.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from anonymizer import read_jsonl, regex_stage, write_jsonl  # noqa: E402
from anonymizer.annotations import document_row  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    t = time.time()
    rows = []
    for doc in read_jsonl(src):
        rows.append(document_row(doc["id"], doc["original"], regex_stage.find(doc["original"]), {"stage": "regex"}))
    write_jsonl(dst, rows)
    print(f"RegEx: {len(rows)} документів, {sum(len(r['annotations']) for r in rows)} анотацій за {time.time() - t:.0f} с",
          file=sys.stderr)


if __name__ == "__main__":
    main()
