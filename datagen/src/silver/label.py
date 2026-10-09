"""Step 3: add labels for PD that ЄДРСР left open; text is not modified."""
from __future__ import annotations

import logging
from collections import Counter

from silver.io import read_jsonl, step_path, write_jsonl
from silver.masks import label_masks
from silver.merge import any_overlap, merge
from silver.numbers import label_numbers
from silver.persons import label_persons

log = logging.getLogger("silver")


def label_doc(doc: dict) -> dict:
    text = doc["text"]
    labels = list(doc["labels"])
    masks = list(doc["masks"])
    # A value that the fill step masked on purpose (it could not tell what the clerk hid) stays a mask: merge() lets a
    # label win over a mask, so a name or a number that a regex layer finds inside such a value is dropped here.
    filled = list(doc["masks"])
    labels += [l for l in label_persons(text, labels) if not any_overlap(l, filled)]
    nl, nm, stats = label_numbers(text, labels, masks)
    labels += [l for l in nl if not any_overlap(l, filled)]
    masks += nm
    nl, nm = label_masks(text, labels, masks)
    labels += nl
    masks += nm
    labels, masks = merge(text, labels, masks)
    return {**doc, "labels": labels, "masks": masks, "iban_stats": stats}


def run_label(src: str = "01_filled", dst: str = "02_labeled") -> None:
    src_c: Counter = Counter()
    mask_c: Counter = Counter()
    iban: Counter = Counter()

    def gen():
        for doc in read_jsonl(step_path(src)):
            out = label_doc(doc)
            src_c.update(l["source"] for l in out["labels"])
            mask_c.update(m["reason"] for m in out["masks"])
            iban.update(out.pop("iban_stats"))
            yield out

    n = write_jsonl(step_path(dst), gen())
    log.info("labeled %d docs; labels by source: %s", n, dict(src_c.most_common()))
    log.info("masks by reason: %s; iban: %s", dict(mask_c.most_common()), dict(iban))
