"""Open the state bodies in a court_cases-schema JSONL: drop every ІНФОРМАЦІЯ annotation that overlaps a
STATE_BODY match (predict_trf.py), the same rule as data/test/court_cases_fixed_state_bodies_open.jsonl.

Art. 7 of the Law on access to court decisions and item 5 of the DSA codifier protect only natural persons, and the
gold masks the same bodies inconsistently. Remaining pseudonyms keep their numbers; `pseudonymized` and
`annotations_by_pseudonym` are rebuilt from the remaining annotations.

Usage: python datagen/scripts/open_state_bodies.py IN.jsonl OUT.jsonl
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from predict_trf import state_body_spans  # noqa: E402


def render(text: str, annotations: list[dict]) -> str:
    out, pos = [], 0
    for a in sorted(annotations, key=lambda x: x["start"]):
        if a["start"] < pos:
            continue
        out += [text[pos: a["start"]], a["pseudonym"]]
        pos = a["end"]
    out.append(text[pos:])
    return "".join(out)


def open_bodies(doc: dict) -> tuple[dict, int]:
    text = doc["original"]
    bodies = state_body_spans(text)
    keep, dropped = [], 0
    for a in doc["annotations"]:
        if a["pseudonym"].startswith("ІНФОРМАЦІЯ") and any(a["start"] < e and s < a["end"] for s, e in bodies):
            dropped += 1
        else:
            keep.append(a)
    if not dropped:
        return doc, 0
    by_ps: dict[str, list[dict]] = {}
    for a in keep:
        by_ps.setdefault(a["pseudonym"], []).append({k: a[k] for k in ("start", "end", "entity")})
    return {**doc, "pseudonymized": render(text, keep), "annotations": keep, "annotations_by_pseudonym": by_ps}, dropped


def main(src: str, dst: str) -> None:
    n = docs = dropped = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            if not line.strip():
                continue
            d, k = open_bodies(json.loads(line))
            g.write(json.dumps(d, ensure_ascii=False) + "\n")
            n, docs, dropped = n + 1, docs + bool(k), dropped + k
    print(f"{src} -> {dst}: {n} docs, {dropped} masks removed in {docs} docs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
