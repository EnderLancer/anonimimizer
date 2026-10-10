"""Step 6: deterministic train/dev/dev_judges split by court case, so decisions of one case never land on both sides."""
from __future__ import annotations

import hashlib
import logging

from silver.io import read_jsonl, step_path, write_jsonl

log = logging.getLogger("silver")
TYPES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
DEV_N, JUDGES_N = 500, 200
DEV_BUCKETS = 12  # % of the Supreme Court cases that dev is drawn from
OPEN_NAME_SOURCES = {"role_regex", "header", "signature", "ini_any", "full_pib", "in_person_of", "ner"}


def _group(d: dict) -> str:
    return d.get("case") or d["id"]


def _bucket(key: str) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100


def split_docs(docs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """(train, dev, dev_judges). Held-out sets take one fill per decision; other fills and other decisions of a
    held-out case are dropped, not trained on."""
    docs = sorted(docs, key=lambda d: (_bucket(_group(d)), _group(d), int(d["id"]), d.get("fill", 1)))
    first = [d for d in docs if d.get("fill", 1) == 1]
    # dev: Supreme Court decisions only (the genre of the test set), positives only; rarer types first so that
    # every type is represented
    cand = [d for d in first if d.get("corpus") == "vs5k" and _bucket(_group(d)) < DEV_BUCKETS and not d.get("is_negative")]
    cand.sort(key=lambda d: -len({l["type"] for l in d["labels"]} - {"ОСОБА"}))
    dev = cand[:DEV_N]
    dev_ids = {d["id"] for d in dev}
    judges = []
    for d in first:
        has_open = any(l["source"] in OPEN_NAME_SOURCES for l in d["labels"])
        if len(judges) < JUDGES_N and d["id"] not in dev_ids and d.get("corpus") != "vs5k" and has_open and DEV_BUCKETS <= _bucket(_group(d)) < DEV_BUCKETS + 6:
            judges.append(d)
    held_ids = dev_ids | {d["id"] for d in judges}
    held_cases = {_group(d) for d in dev + judges}
    train = [d for d in docs if d["id"] not in held_ids and _group(d) not in held_cases]
    return train, dev, judges


def run_split(src: str = "04_excerpt") -> None:
    docs = list(read_jsonl(step_path(src)))
    train, dev, judges = split_docs(docs)
    type_counts = {t: 0 for t in TYPES}
    for d in dev:
        for t in {l["type"] for l in d["labels"]}:
            type_counts[t] += 1
    for name, rows in (("split_train", train), ("split_dev", dev), ("split_dev_judges", judges)):
        write_jsonl(step_path(name), rows)
    log.info("split: train %d, dev %d (types %s), dev_judges %d, dropped with held-out cases %d",
             len(train), len(dev), type_counts, len(judges), len(docs) - len(train) - len(dev) - len(judges))
