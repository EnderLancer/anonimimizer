"""Step 1: select usable documents from the parquet and normalize their text."""
from __future__ import annotations

import logging
import random
from collections import Counter

from silver.io import read_parquet_texts, step_path, write_jsonl
from silver.normalize import MOJIBAKE, ODD_PLACEHOLDER, RUSSIAN, find_placeholders, normalize

log = logging.getLogger("silver")

MIN_LEN, MAX_LEN = 1500, 10000
DEDUP_PREFIX = 300


def _reject_reason(text: str, n_ph: int, seen: set) -> str | None:
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return "length"
    if len(RUSSIAN.findall(text)) > 3:
        return "russian"
    if ODD_PLACEHOLDER.search(text):
        return "odd_placeholder"
    if MOJIBAKE.search(text):
        return "mojibake"
    key = text[:DEDUP_PREFIX]
    if key in seen:
        return "duplicate"
    seen.add(key)
    return None


def run_filter(limit: int | None = None, negatives: float = 0.10) -> None:
    rng = random.Random(0)
    seen: set[str] = set()
    reasons: Counter = Counter()
    kept, kept_neg = 0, 0

    def gen():
        nonlocal kept, kept_neg
        for doc_id, raw in read_parquet_texts(limit=limit):
            text = normalize(raw)
            placeholders = find_placeholders(text)
            reason = _reject_reason(text, len(placeholders), seen)
            if reason:
                reasons[reason] += 1
                continue
            is_negative = not placeholders
            if is_negative:
                # keep only a sample of placeholder-free documents, proportional to the positives seen so far
                if kept_neg >= max(1, int(negatives * kept)) or rng.random() > 0.5:
                    reasons["negative_sampled_out"] += 1
                    continue
                kept_neg += 1
            kept += 1
            yield {"id": doc_id, "text": text, "placeholders": placeholders, "is_negative": is_negative}

    n = write_jsonl(step_path("00_filtered"), gen())
    log.info("kept %d documents (%d negatives); rejected: %s", n, kept_neg, dict(reasons))
