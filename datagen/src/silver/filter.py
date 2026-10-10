"""Step 1: select usable documents from the source parquet files and normalize their text."""
from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import Counter

from silver.exclude import ExcludedPool, case_number
from silver.io import SOURCES, VARIANT, read_parquet_texts, step_path, write_jsonl
from silver.normalize import ODD_PLACEHOLDER, find_placeholders, normalize

log = logging.getLogger("silver")

MIN_LEN, MAX_LEN = 1500, 10000
# The Hugging Face files and the test set hold decisions of at most 10 000 characters; the registry corpus (vs_edrsr)
# has no such cap, and «large» keeps its longer decisions too (training reads sliding windows anyway)
MAX_LEN_REGISTRY = {"base": 10000, "large": 30000}[VARIANT]
DEDUP_PREFIX = 300
_CODES = {
    "КУпАП": re.compile(r"КУпАП|адміністративн\w+\s+правопорушенн"),
    "КПК": re.compile(r"\bКПК\b|кримінальн\w+\s+провадженн"),
    "КАС": re.compile(r"\bКАС\b|адміністративн\w+\s+судочинств"),
    "ЦПК": re.compile(r"\bЦПК\b|цивільн\w+\s+(?:справ|судочинств)"),
    "ГПК": re.compile(r"\bГПК\b|господарськ\w+\s+суд"),
}
_MILITARY = re.compile(r"військов\w+\s+частин\w*\s+(?:№\s*)?НОМЕР_\d+")
# Share of the general ЄДРСР corpus that is kept, by kind of decision. The test set is Supreme Court decisions, so
# those and the kinds with rare entity types stay whole; petty-offence rulings (a fifth of the corpus) nearly all go.
# «large» keeps every decision with a rare entity type except most petty-offence rulings.
KEEP = {
    "base": {"ВС": 1.0, "військові": 1.0, "КАС": 1.0, "ЦПК": 0.55, "ГПК": 0.5, "інше": 0.5, "КПК": 0.3, "КУпАП": 0.04},
    "large": {"ВС": 1.0, "військові": 1.0, "КАС": 1.0, "ЦПК": 1.0, "ГПК": 1.0, "інше": 1.0, "КПК": 1.0, "КУпАП": 0.2},
}[VARIANT]
# multiplier for documents whose only placeholders are ОСОБА (not for the Supreme Court)
KEEP_PERSONS_ONLY = {"base": 0.15, "large": 0.3}[VARIANT]


def doc_kind(text: str) -> str:
    """Supreme Court / military-unit case / the procedural code the decision cites most."""
    if "Верховн" in text[:600]:
        return "ВС"
    if _MILITARY.search(text):
        return "військові"
    score = {k: len(rx.findall(text)) for k, rx in _CODES.items()}
    return max(score, key=score.get) if max(score.values()) else "інше"


def _keep(doc_id: str, share: float) -> bool:
    return int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % 10000 < share * 10000


def _reject_reason(text: str, n_ph: int, seen: set, max_len: int = MAX_LEN) -> str | None:
    if not (MIN_LEN <= len(text) <= max_len):
        return "length"
    if ODD_PLACEHOLDER.search(text):
        return "odd_placeholder"
    key = text[:DEDUP_PREFIX]
    if key in seen:
        return "duplicate"
    seen.add(key)
    return None


def run_filter(limit: int | None = None, negatives: float = 0.03) -> None:
    rng = random.Random(0)
    pool = ExcludedPool()
    seen: set[str] = set()
    seen_ids: set[str] = set()
    reasons: Counter = Counter()
    kinds: Counter = Counter()
    kept, kept_neg = 0, 0

    def gen():
        nonlocal kept, kept_neg
        for corpus, path in SOURCES:
            for doc_id, raw in read_parquet_texts(path, limit=limit):
                if doc_id in seen_ids:
                    reasons["same_id"] += 1
                    continue
                seen_ids.add(doc_id)
                text = normalize(raw)
                placeholders = find_placeholders(text)
                max_len = MAX_LEN_REGISTRY if corpus == "vs_edrsr" else MAX_LEN
                reason = _reject_reason(text, len(placeholders), seen, max_len) or pool.reason(doc_id, text)
                if reason:
                    reasons[reason] += 1
                    continue
                is_negative = not placeholders
                supreme = corpus in ("vs5k", "vs_edrsr")
                kind = "ВС" if supreme else doc_kind(text)
                if not supreme:
                    rare = kind == "ВС" or bool({p["type"] for p in placeholders} - {"ОСОБА"})
                    share = KEEP[kind] * (1.0 if rare else KEEP_PERSONS_ONLY)
                    if not is_negative and not _keep(doc_id, share):
                        reasons["mix"] += 1
                        continue
                if is_negative:
                    # keep only a sample of placeholder-free documents, proportional to the positives seen so far
                    if kept_neg >= max(1, int(negatives * kept)) or rng.random() > 0.5:
                        reasons["negative_sampled_out"] += 1
                        continue
                    kept_neg += 1
                kept += 1
                kinds[f"{corpus}:{kind}"] += 1
                yield {"id": doc_id, "corpus": corpus, "kind": kind, "case": case_number(text), "text": text,
                       "placeholders": placeholders, "is_negative": is_negative}

    n = write_jsonl(step_path("00_filtered"), gen())
    log.info("kept %d documents (%d negatives): %s; rejected: %s", n, kept_neg, dict(kinds.most_common()), dict(reasons))
