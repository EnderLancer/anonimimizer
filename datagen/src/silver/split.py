"""Step 6: deterministic train/dev/dev_judges split by source document id."""
from __future__ import annotations

import hashlib
import logging

from silver.io import read_jsonl, step_path, write_jsonl

log = logging.getLogger("silver")
TYPES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
DEV_N, JUDGES_N = 500, 200
OPEN_NAME_SOURCES = {"role_regex", "header", "signature", "ini_any", "full_pib", "in_person_of", "ner"}


def _bucket(doc_id: str) -> int:
    return int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % 100


def run_split(src: str | None = None) -> None:
    src = src or ("03_ner" if step_path("03_ner").exists() else "02_labeled")
    docs = list(read_jsonl(step_path(src)))
    docs.sort(key=lambda d: _bucket(d["id"]) * 10**12 + int(d["id"]))
    dev, judges, train = [], [], []
    type_counts = {t: 0 for t in TYPES}
    # dev candidates: 5 % of the hash space, positives only; rarer types first so that every type is represented
    cand = [d for d in docs if _bucket(d["id"]) < 5 and not d.get("is_negative")]
    cand.sort(key=lambda d: -len({l["type"] for l in d["labels"]} - {"ОСОБА"}))
    dev = cand[:DEV_N]
    dev_ids = {d["id"] for d in dev}
    for d in dev:
        for t in {l["type"] for l in d["labels"]}:
            type_counts[t] += 1
    for d in docs:
        if d["id"] in dev_ids:
            continue
        has_open = any(l["source"] in OPEN_NAME_SOURCES for l in d["labels"])
        if len(judges) < JUDGES_N and has_open and 5 <= _bucket(d["id"]) < 8:
            judges.append(d)
            continue
        train.append(d)
    for name, rows in (("split_train", train), ("split_dev", dev), ("split_dev_judges", judges)):
        write_jsonl(step_path(name), rows)
    log.info("split: train %d, dev %d (types %s), dev_judges %d", len(train), len(dev), type_counts, len(judges))
