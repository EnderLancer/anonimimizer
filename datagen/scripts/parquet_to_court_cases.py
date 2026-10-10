"""Convert a silver train .parquet into the court_cases.jsonl test schema.

Usage: python datagen/scripts/parquet_to_court_cases.py IN.parquet OUT.jsonl
"""
from __future__ import annotations

import json
import sys

import pyarrow.parquet as pq


def convert(row: dict) -> dict:
    text = row["text"]
    anns = [{"start": a["start"], "end": a["end"], "entity": text[a["start"]: a["end"]], "pseudonym": a["pseudonym"]}
            for a in row["annotations"]]
    by_pseudonym: dict[str, list[dict]] = {}
    for a in anns:
        by_pseudonym.setdefault(a["pseudonym"], []).append({k: a[k] for k in ("start", "end", "entity")})
    return {"id": row["id"], "pseudonymized": row["text_pseudonymized"], "original": text,
            "annotations": anns, "annotations_by_pseudonym": by_pseudonym}


def main(src: str, dst: str) -> None:
    pf = pq.ParquetFile(src)
    n = 0
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for batch in pf.iter_batches(batch_size=500):
            for row in batch.to_pylist():
                f.write(json.dumps(convert(row), ensure_ascii=False) + "\n")
                n += 1
    print(f"{src} -> {dst}: {n} docs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
