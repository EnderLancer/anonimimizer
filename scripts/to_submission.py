#!/usr/bin/env python3
"""Лишає з результату solution/link.py лише те, що читає data/evaluate.py.

Запуск:
    python scripts/to_submission.py linked.jsonl submission.jsonl

Вихід: {"id": ..., "annotations": [{"start": ..., "end": ..., "pseudonym": "ОСОБА_1"}, ...]}
"""
import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    with open(src, encoding="utf-8-sig") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            anns = [{"start": a["start"], "end": a["end"], "pseudonym": a["pseudonym"]} for a in row["annotations"]]
            fout.write(json.dumps({"id": row["id"], "annotations": anns}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
