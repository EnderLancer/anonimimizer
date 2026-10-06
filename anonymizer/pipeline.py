"""Flow: document -> RegEx -> RegExed document -> Model -> Cleaned document -> Link results -> Cleaned document.

RegEx і Model збирають приховані значення з позиціями в оригіналі, Mapping values дає їм
псевдоніми, Link results підставляє псевдоніми в оригінал.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import mapping, regex_stage
from .document import Hidden, MaskedDocument


@dataclass
class Result:
    cleaned: str               # оригінал з псевдонімами замість персональних даних
    annotations: list[dict]    # [{"start", "end", "pseudonym"}] у координатах оригіналу
    hidden: list[Hidden]


def link(original: str, labels: dict[Hidden, str]) -> Result:
    hidden = sorted(labels, key=lambda h: h.start)
    parts, pos = [], 0
    for h in hidden:
        parts += [original[pos:h.start], labels[h]]
        pos = h.end
    parts.append(original[pos:])
    anns = [{"start": h.start, "end": h.end, "pseudonym": labels[h]} for h in hidden]
    return Result("".join(parts), anns, hidden)


class Pipeline:
    def __init__(self, model=None):
        """model — об'єкт з методом find_batch(list[MaskedDocument]) -> list[list[Hidden]] або None (лише RegEx)."""
        self.model = model

    def run_batch(self, documents: list[str]) -> list[Result]:
        regexed = [MaskedDocument(doc, regex_stage.find(doc)) for doc in documents]
        model_found = self.model.find_batch(regexed) if self.model else [[] for _ in regexed]
        results = []
        for doc, found in zip(regexed, model_found):
            hidden = doc.hidden + [h for h in found if not any(h.start < r.end and r.start < h.end for r in doc.hidden)]
            results.append(link(doc.original, mapping.assign(hidden)))
        return results

    def run(self, document: str) -> Result:
        return self.run_batch([document])[0]
