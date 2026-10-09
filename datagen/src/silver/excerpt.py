"""Step 5: cut every decision to its reasoning part. The test set holds excerpts: the text after «ВСТАНОВИВ:»
without the signatures; only a minority of them keep the heading with the panel of judges or the signature block."""
from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import Counter

from silver.io import labeled_step, read_jsonl, step_path, write_jsonl

log = logging.getLogger("silver")

_SP = r"[ \t]?"
# ВСТАНОВИВ: / УСТАНОВИЛА / встановив: / В С Т А Н О В И В :
_BODY = re.compile(rf"^[ \t]*(?:[ВУ]{_SP}С{_SP}Т{_SP}А{_SP}Н{_SP}О{_SP}В{_SP}И{_SP}(?:В|Л{_SP}А|Л{_SP}И)|[ву]станови(?:в|ла|ли))[ \t]*:?[ \t]*$", re.M)
_SIGNATURE = re.compile(r"^[ \t]*(?:Судд[яі]|Головуюч(?:ий|а)|Суддя-доповідач)\b", re.M)
P_KEEP_HEAD, P_KEEP_SIGNATURE = 0.05, 0.13  # decisions without «ВСТАНОВИВ:» (a fifth) keep their heading anyway
TAIL = 700      # the signature block is looked for in the last characters
MIN_LEN = 1000  # a shorter excerpt is not cut at all


def cut_points(text: str) -> tuple[int, int]:
    """(start of the reasoning part, start of the signature block); 0 / len(text) when a boundary is not found."""
    m = _BODY.search(text, 0, int(len(text) * 0.6))
    start = m.end() if m else 0
    m = _SIGNATURE.search(text, max(start, len(text) - TAIL))
    return start, m.start() if m else len(text)


def excerpt(doc: dict) -> dict:
    text = doc["text"]
    key = doc["id"] + (f":{doc['fill']}" if doc.get("fill", 1) > 1 else "")
    rng = random.Random(int(hashlib.sha1(key.encode()).hexdigest(), 16))
    body, sign = cut_points(text)
    a = 0 if rng.random() < P_KEEP_HEAD else body
    b = len(text) if rng.random() < P_KEEP_SIGNATURE else sign
    spans = doc["labels"] + doc["masks"]
    a = min([a] + [s["start"] for s in spans if s["start"] < a < s["end"]])  # never cut through a labelled span
    b = max([b] + [s["end"] for s in spans if s["start"] < b < s["end"]])
    a += len(text[a:b]) - len(text[a:b].lstrip())
    b -= len(text[a:b]) - len(text[a:b].rstrip())
    if b - a < MIN_LEN:
        a, b = 0, len(text)

    def shift(items: list[dict]) -> list[dict]:
        return [{**s, "start": s["start"] - a, "end": s["end"] - a} for s in items if a <= s["start"] and s["end"] <= b]

    return {**doc, "text": text[a:b], "labels": shift(doc["labels"]), "masks": shift(doc["masks"]),
            "cut": {"head": a > 0, "signature": b < len(text)}}


def run_excerpt(src: str | None = None, dst: str = "04_excerpt") -> None:
    stats: Counter = Counter()

    def gen():
        for doc in read_jsonl(step_path(src or labeled_step())):
            out = excerpt(doc)
            stats["documents"] += 1
            stats["head_cut"] += out["cut"]["head"]
            stats["signature_cut"] += out["cut"]["signature"]
            stats["labels_dropped"] += len(doc["labels"]) - len(out["labels"])
            stats["chars_in"] += len(doc["text"])
            stats["chars_out"] += len(out["text"])
            yield out

    write_jsonl(step_path(dst), gen())
    log.info("excerpts: %s", dict(stats))
