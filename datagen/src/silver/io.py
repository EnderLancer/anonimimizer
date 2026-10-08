"""Reading the ЄДРСР parquet and reading/writing intermediate JSONL files."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[3]  # repo root (datagen/src/silver/io.py)
DATA = ROOT / "data"
# A dataset variant has its own output folder: «base» -> data/silver, «large» -> data/silver_large. It also selects
# the corpus mix (filter.KEEP) and how many times a Supreme Court decision is filled (fill.run.VS_FILLS).
# make_silver.py --variant sets the environment variable before this module is imported.
VARIANT = os.environ.get("SILVER_VARIANT", "base")
SILVER = DATA / ("silver" if VARIANT == "base" else f"silver_{VARIANT}")
RESOURCES = DATA / "resources"
TRAIN_PARQUET = DATA / "train" / "2024-court-decisions.parquet"
# Supreme Court decisions of 2024 from the same Hugging Face dataset (lang-uk/court-decisions-uk); it contains the
# 250 decisions of TEST_POOL_PARQUET, which silver.exclude keeps out of the training data
VS_PARQUET = DATA / "train" / "2024-5K-supreme-court-decisions-deduplicated.parquet"
TEST_POOL_PARQUET = DATA / "train" / "250-deanonymized-court-cases.parquet"
SOURCES = (("vs5k", VS_PARQUET), ("edrsr", TRAIN_PARQUET))


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


def labeled_step() -> str:
    """03_ner when the NER layer was run on the current labels, otherwise 02_labeled."""
    ner, labeled = step_path("03_ner"), step_path("02_labeled")
    return "03_ner" if ner.exists() and ner.stat().st_mtime >= labeled.stat().st_mtime else "02_labeled"
