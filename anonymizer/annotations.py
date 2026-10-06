"""Спільний формат анотацій для всіх етапів.

RegEx і Model пишуть JSON Lines, один рядок на документ, без номерів псевдонімів:

    {"id": 116075957,
     "original": "У травні 2018 року Іваненко Іван Іванович звернувся ...",
     "metadata": {"stage": "regex"},
     "annotations": [
        {"start": 19, "end": 42, "type": "ОСОБА", "text": "Іваненко Іван Іванович",
         "metadata": {"source": "regex", "rule": "person_full_name"}}
     ]}

solution/link.py об'єднує їх і дописує кожній анотації pseudonym з номером ("ОСОБА_1").
scripts/to_submission.py лишає з цього тільки id та start, end, pseudonym — формат data/evaluate.py.
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

    def to_dict(self, pseudonym: str | None = None) -> dict:
        d = {"start": self.start, "end": self.end, "type": self.type, "text": self.text, "metadata": self.metadata}
        if pseudonym:
            d["pseudonym"] = pseudonym
        return d

    @classmethod
    def from_dict(cls, d: dict, original: str) -> "Hidden":
        return cls(d["start"], d["end"], d["type"], d.get("text", original[d["start"]:d["end"]]), d.get("metadata", {}))


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def document_row(doc_id, original: str, hidden: list[Hidden], metadata: dict,
                 labels: dict[Hidden, str] | None = None) -> dict:
    hidden = sorted(hidden, key=lambda h: h.start)
    return {"id": doc_id, "original": original, "metadata": metadata,
            "annotations": [h.to_dict(labels[h] if labels else None) for h in hidden]}
