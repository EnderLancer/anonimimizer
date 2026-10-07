"""Reading the ЄДРСР parquet and reading/writing intermediate JSONL files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[3]  # repo root (datagen/src/silver/io.py)
DATA = ROOT / "data"
SILVER = DATA / "silver"
RESOURCES = DATA / "resources"
TRAIN_PARQUET = DATA / "train" / "2024-court-decisions.parquet"


def read_parquet_texts(path: Path = TRAIN_PARQUET, limit: int | None = None) -> Iterator[tuple[str, str]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(batch_size=2000, columns=["id", "text"]):
        ids = batch.column("id").to_pylist()
        texts = batch.column("text").to_pylist()
        for i, t in zip(ids, texts):
            yield str(i), t
            n += 1
            if limit and n >= limit:
                return


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def step_path(step: str) -> Path:
    return SILVER / f"{step}.jsonl"
