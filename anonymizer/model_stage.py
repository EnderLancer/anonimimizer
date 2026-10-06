"""Етап Model: NER-модель з Hugging Face шукає персональні дані в RegExed document."""
from __future__ import annotations

import os

from .document import Hidden, MaskedDocument

# мітка моделі -> тип псевдоніма (None — не приховуємо).
# ORG вимкнено: загальна NER-модель позначає здебільшого суди й держоргани, а це не персональні дані
DEFAULT_LABELS = {"PER": "ОСОБА", "ORG": None, "LOC": None, "MISC": None}


class NerModel:
    def __init__(self, name: str = "ukr-models/uk-ner", labels: dict[str, str | None] | None = None,
                 min_score: float = 0.8, batch_size: int = 8):
        from transformers import AutoTokenizer, pipeline

        self.labels = labels or DEFAULT_LABELS
        self.min_score = min_score
        self.batch_size = batch_size
        # довгі документи ріжемо на вікна по 512 токенів з перекриттям stride
        tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=512)
        self.ner = pipeline("token-classification", model=name, tokenizer=tokenizer, aggregation_strategy="first",
                            stride=128, device=int(os.environ.get("NER_DEVICE", "-1")))

    def _keep(self, ent: dict, text: str) -> str | None:
        kind = self.labels.get(ent["entity_group"])
        if kind is None or ent["score"] < self.min_score:
            return None
        value = text[ent["start"]:ent["end"]]
        if kind == "ОСОБА" and len(value.split()) < 2:
            return None  # одне слово — найчастіше не особа, а частина назви
        return kind

    def find_batch(self, docs: list[MaskedDocument]) -> list[list[Hidden]]:
        outputs = self.ner([d.text for d in docs], batch_size=self.batch_size)
        found = []
        for doc, ents in zip(docs, outputs):
            hidden = []
            for ent in ents:
                kind = self._keep(ent, doc.text)
                pos = doc.to_original(ent["start"], ent["end"]) if kind else None
                if pos:
                    s, e = pos
                    while e > s and doc.original[e - 1] in " .,;:":
                        e -= 1
                    if e > s:
                        hidden.append(Hidden(s, e, kind, doc.original[s:e], "model"))
            found.append(hidden)
        return found
