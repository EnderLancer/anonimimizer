#!/usr/bin/env python3
"""Silver training-data pipeline. Each step reads the previous step's JSONL and writes its own.

    uv run scripts/make_silver.py filter [--limit N]
    uv run scripts/make_silver.py fill   [--seed 42]
    uv run scripts/make_silver.py label
    uv run scripts/make_silver.py ner
    uv run scripts/make_silver.py excerpt
    uv run scripts/make_silver.py audit | samples | split | export | report
    uv run scripts/make_silver.py all    [--limit N] [--skip-ner] [--variant large]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["filter", "fill", "label", "ner", "excerpt", "audit", "samples", "split", "export", "report", "smoke", "all"])
    ap.add_argument("--limit", type=int, default=None, help="process only the first N parquet rows (debug)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-ner", action="store_true", help="'all' without the NER layer")
    ap.add_argument("--negatives", type=float, default=0.03, help="share of placeholder-free documents to keep")
    ap.add_argument("--score", action="store_true", help="samples: score the ok/bad marks instead of generating")
    ap.add_argument("--batch", type=int, default=8, help="ner: spaCy batch size")
    ap.add_argument("--variant", choices=["base", "large"], default="base",
                    help="large: more of the general corpus and every Supreme Court decision filled twice; written to data/silver_large/")
    a = ap.parse_args()
    os.environ["SILVER_VARIANT"] = a.variant  # silver.io reads it on import
    from silver import pipeline

    pipeline.run(a)


if __name__ == "__main__":
    main()
