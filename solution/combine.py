#!/usr/bin/env python3
"""Етап злиття: кандидати -> анотації з псевдонімами (ОСОБА_1, НОМЕР_2 …) і очищений документ.

Запуск:
    python solution/combine.py candidates.jsonl linked.jsonl

Політика злиття й нумерації — pseudonymizer.pipeline.combine. Вихідні анотації зберігають type, text і metadata
кандидата й отримують pseudonym; рядок документа отримує pseudonymized. Для data/evaluate.py результат
проганяють через scripts/to_submission.py.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pseudonymizer.pipeline import combine  # noqa: E402


def render(text, annotations):
    parts, pos = [], 0
    for a in annotations:
        parts += [text[pos:a["start"]], a["pseudonym"]]
        pos = a["end"]
    parts.append(text[pos:])
    return "".join(parts)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    n, by_source = 0, Counter()
    with open(src, encoding="utf-8-sig") as fin, open(dst, "w", encoding="utf-8") as out:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            anns = combine(row["original"], row["annotations"])
            by_source.update(a["metadata"].get("source") for a in anns)
            out.write(json.dumps({"id": row["id"], "original": row["original"],
                                  "metadata": {**row.get("metadata", {}), "stage": "combine"},
                                  "annotations": anns, "pseudonymized": render(row["original"], anns)},
                                 ensure_ascii=False) + "\n")
            n += 1
    print(f"combine: {n} рішень, анотацій за джерелом: {dict(by_source)}", file=sys.stderr)


if __name__ == "__main__":
    main()
