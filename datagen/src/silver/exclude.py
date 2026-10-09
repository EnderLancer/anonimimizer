"""Documents that must never reach the training data: the 250 Supreme Court decisions the organizers' test set was
cut from, other decisions of the same cases, and near-copies of those decisions."""
from __future__ import annotations

import re
from collections import Counter

from silver.io import TEST_POOL_PARQUET

_CASE = re.compile(r"[Сс]прав[аі]\s*№\s*([0-9A-Za-zА-Яа-яІЇЄҐіїєґ/\-]+)")
_WORD = re.compile(r"[^\W\d_]+|\d+")
_PLACEHOLDER_NO = re.compile(r"(ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_\d+")
HEADING = 1500
SHINGLE = 5
NEAR_COPY = 0.5  # share of the shorter document's word 5-grams found in the other one


def case_number(text: str) -> str | None:
    """«справа № 296/5017/18» from the heading, in a form that survives typing differences."""
    m = _CASE.search(text[:HEADING])
    if not m:
        return None
    return m.group(1).strip("/-").lower().replace("–", "-").translate(str.maketrans("асеікморх", "aceikmopx")) or None


def shingles(text: str) -> set[int]:
    words = _WORD.findall(_PLACEHOLDER_NO.sub(r"\1", text).lower())
    return {hash(" ".join(words[i: i + SHINGLE])) for i in range(len(words) - SHINGLE + 1)}


class ExcludedPool:
    def __init__(self, docs: list[tuple[str, str]] | None = None):
        """`docs` are (id, registry text) pairs; by default the 250-deanonymized-court-cases file."""
        if docs is None:
            if not TEST_POOL_PARQUET.exists():
                raise FileNotFoundError(f"{TEST_POOL_PARQUET} is missing: run scripts/fetch_resources.py")
            import pyarrow.parquet as pq

            t = pq.read_table(TEST_POOL_PARQUET, columns=["id", "original_text"])
            docs = [(str(i), x) for i, x in zip(t.column("id").to_pylist(), t.column("original_text").to_pylist())]
        self.ids = {i for i, _ in docs}
        self.cases = {c for c in (case_number(x) for _, x in docs) if c}
        self.sizes = []
        self.index: dict[int, list[int]] = {}
        for n, (_, x) in enumerate(docs):
            sh = shingles(x)
            self.sizes.append(len(sh))
            for h in sh:
                self.index.setdefault(h, []).append(n)

    def reason(self, doc_id: str, text: str) -> str | None:
        """Why the document is excluded, or None when it may be used."""
        if doc_id in self.ids:
            return "test_pool"
        if case_number(text) in self.cases:
            return "test_case"
        sh = shingles(text)
        hits = Counter(n for h in sh for n in self.index.get(h, ()))
        if any(c >= NEAR_COPY * min(len(sh), self.sizes[n]) for n, c in hits.items()):
            return "test_near_copy"
        return None
