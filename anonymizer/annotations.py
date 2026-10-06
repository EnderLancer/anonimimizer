"""Спільний формат анотацій для всіх етапів.

Кожен етап пише JSON Lines, один рядок на документ:

    {"id": 116075957,
     "original": "У травні 2018 року Іваненко Іван Іванович звернувся ...",
     "metadata": {"stage": "regex"},
     "annotations": [
        {"start": 19, "end": 42, "pseudonym": "ОСОБА_1", "type": "ОСОБА",
         "text": "Іваненко Іван Іванович", "metadata": {"source": "regex", "rule": "person_full_name"}}
     ]}

start, end, pseudonym — те, що читає data/evaluate.py, тож вихід кожного етапу можна оцінити окремо.
Решта полів — для наступних етапів і для аналізу помилок.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

TYPES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")


@dataclass(eq=False)
class Hidden:
    """Прихований фрагмент original[start:end] певного типу."""
    start: int
    end: int
    type: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    def overlaps(self, other: "Hidden") -> bool:
        return self.start < other.end and other.start < self.end

    def to_dict(self, pseudonym: str) -> dict:
        return {"start": self.start, "end": self.end, "pseudonym": pseudonym, "type": self.type,
                "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict, original: str) -> "Hidden":
        kind = d.get("type") or d["pseudonym"].strip().rsplit("_", 1)[0]
        return cls(d["start"], d["end"], kind, d.get("text", original[d["start"]:d["end"]]), d.get("metadata", {}))


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def document_row(doc_id, original: str, hidden: list[Hidden], labels: dict[Hidden, str], metadata: dict) -> dict:
    hidden = sorted(hidden, key=lambda h: h.start)
    return {"id": doc_id, "original": original, "metadata": metadata,
            "annotations": [h.to_dict(labels[h]) for h in hidden]}
