#!/usr/bin/env python3
"""Hidden values at positions -> Mapping values -> Link results -> Cleaned document.

Запуск:
    python solution/link.py submission.jsonl regex.jsonl model.jsonl

Об'єднує анотації етапів (за перетину перемагає файл, указаний раніше), дає однакові псевдоніми
однаковим значенням і пише анотації разом з очищеним документом (поле pseudonymized).
Документ, якого немає в одному з файлів, береться з решти.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from anonymizer import Hidden, mapping, pipeline, read_jsonl, write_jsonl  # noqa: E402
from anonymizer.annotations import document_row  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    dst, sources = sys.argv[1], sys.argv[2:]
    stages = [{r["id"]: r for r in read_jsonl(p)} for p in sources]
    priority = tuple(next(iter(s.values()))["metadata"]["stage"] for s in stages if s)

    ids = list(dict.fromkeys(i for s in stages for i in s))
    rows = []
    for doc_id in ids:
        rows_of_doc = [s[doc_id] for s in stages if doc_id in s]
        original = rows_of_doc[0]["original"]
        if any(r["original"] != original for r in rows_of_doc):
            sys.exit(f"id={doc_id}: етапи працювали з різними версіями документа")
        found = [[Hidden.from_dict(a, original) for a in r["annotations"]] for r in rows_of_doc]
        hidden = pipeline.merge(*found, priority=priority)
        labels = mapping.assign(hidden)
        row = document_row(doc_id, original, hidden, labels, {"stage": "link", "sources": list(priority)})
        row["pseudonymized"] = pipeline.link(original, labels)
        rows.append(row)
    write_jsonl(dst, rows)
    by_source = {}
    for r in rows:
        for a in r["annotations"]:
            by_source[a["metadata"].get("source")] = by_source.get(a["metadata"].get("source"), 0) + 1
    print(f"Link: {len(rows)} документів, анотацій за джерелом: {by_source}", file=sys.stderr)


if __name__ == "__main__":
    main()
