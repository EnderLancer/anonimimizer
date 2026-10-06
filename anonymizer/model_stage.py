"""Етап Model: NER-модель з Hugging Face шукає персональні дані в оригінальному документі."""
from __future__ import annotations

import os

from .annotations import Hidden

# мітка моделі -> тип псевдоніма (None — не приховуємо).
# ORG вимкнено: загальна NER-модель позначає здебільшого суди й держоргани, а це не персональні дані
DEFAULT_LABELS = {"PER": "ОСОБА", "ORG": None, "LOC": None, "MISC": None}


class NerModel:
    def __init__(self, name: str = "ukr-models/uk-ner", labels: dict[str, str | None] | None = None,
                 min_score: float = 0.8, batch_size: int = 8):
        from transformers import AutoTokenizer, pipeline

        self.name = name
        self.labels = labels or DEFAULT_LABELS
        self.min_score = min_score
        self.batch_size = batch_size
        # довгі документи ріжемо на вікна по 512 токенів з перекриттям stride
        tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=512)
        self.ner = pipeline("token-classification", model=name, tokenizer=tokenizer, aggregation_strategy="first",
                            stride=128, device=int(os.environ.get("NER_DEVICE", "-1")))

    @property
    def metadata(self) -> dict:
        return {"stage": "model", "model": self.name, "min_score": self.min_score,
                "labels": {k: v for k, v in self.labels.items() if v}}

    def _keep(self, ent: dict, text: str) -> str | None:
        kind = self.labels.get(ent["entity_group"])
        if kind is None or ent["score"] < self.min_score:
            return None
        if kind == "ОСОБА" and len(text[ent["start"]:ent["end"]].split()) < 2:
            return None  # одне слово — найчастіше не особа, а частина назви
        return kind

    def find_batch(self, texts: list[str]) -> list[list[Hidden]]:
        outputs = self.ner(texts, batch_size=self.batch_size)
        found = []
        for text, ents in zip(texts, outputs):
            hidden = []
            for ent in ents:
                kind = self._keep(ent, text)
                if not kind:
                    continue
                s, e = ent["start"], ent["end"]
                while s < e and text[s].isspace():
                    s += 1
                while e > s and text[e - 1] in " \n.,;:":
                    e -= 1
                if e > s:
                    meta = {"source": "model", "label": ent["entity_group"], "score": round(float(ent["score"]), 4)}
                    hidden.append(Hidden(s, e, kind, text[s:e], meta))
            found.append(hidden)
        return found
