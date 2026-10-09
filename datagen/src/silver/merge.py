"""Span bookkeeping: overlap checks, priority merge and validation."""
from __future__ import annotations

PRIORITY = {"placeholder": 3, "role_regex": 2, "header": 2, "signature": 2, "in_person_of": 2, "full_pib": 2,
            "rnokpp": 2, "passport": 2, "id_card": 2, "unzr": 2, "iban": 2, "card": 2, "email": 2, "dob": 2,
            "address": 2, "ini_any": 2, "ner": 1}


def overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def any_overlap(span: dict, spans: list[dict]) -> bool:
    return any(overlaps(span, s) for s in spans)


def strip_span(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def merge(text: str, labels: list[dict], masks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Resolve overlaps: higher priority wins; same type overlaps are unioned; losers of a
    different type become masks (reason=conflict). Labels always beat masks."""
    out: list[dict] = []
    for lab in sorted(labels, key=lambda x: (-PRIORITY.get(x["source"], 0), x["start"])):
        s, e = strip_span(text, lab["start"], lab["end"])
        if e <= s:
            continue
        lab = {**lab, "start": s, "end": e}
        clash = [o for o in out if overlaps(lab, o)]
        if not clash:
            out.append(lab)
            continue
        if all(o["type"] == lab["type"] for o in clash):
            for o in clash:
                out.remove(o)
            s = min([lab["start"]] + [o["start"] for o in clash])
            e = max([lab["end"]] + [o["end"] for o in clash])
            best = max(clash + [lab], key=lambda x: (PRIORITY.get(x["source"], 0), x["end"] - x["start"]))
            out.append({**best, "start": s, "end": e})
        else:
            masks.append({"start": lab["start"], "end": lab["end"], "reason": "conflict"})
    out.sort(key=lambda x: x["start"])
    kept_masks = []
    for m in masks:
        if m["end"] <= m["start"]:
            continue
        if any_overlap(m, out):
            continue
        kept_masks.append(m)
    kept_masks.sort(key=lambda x: x["start"])
    n = len(text)
    for sp in out + kept_masks:
        assert 0 <= sp["start"] < sp["end"] <= n, sp
    return out, kept_masks
