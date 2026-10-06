#!/usr/bin/env python3
"""Базове рішення: знаходить персональні дані регулярними виразами.

Запуск:
    python solution/predict.py inputs.jsonl submission.jsonl

Вхід — JSON Lines з полями id та original. Вихід — JSON Lines у форматі з annotations
(див. data/evaluate.py). Замініть цей файл своїм рішенням, зберігши той самий інтерфейс.
"""
import json
import re
import sys

UP = "А-ЯІЇЄҐ"
LO = "а-яіїєґ'’ʼ"
SURNAME = rf"[{UP}][{LO}]+(?:-[{UP}][{LO}]+)?"
PATTERNS = [
    # Прізвище Ім'я По батькові
    ("ОСОБА", re.compile(rf"\b{SURNAME}\s+[{UP}][{LO}]+\s+[{UP}][{LO}]+(?:вич|вна|ович|івна|ївна|овича|івни|ївни|овичу|івні|ївні|овичем|івною|ївною)\b")),
    # Прізвище І.Б.
    ("ОСОБА", re.compile(rf"\b{SURNAME}\s+[{UP}]\.\s?[{UP}]\.")),
    # І.Б. Прізвище
    ("ОСОБА", re.compile(rf"\b[{UP}]\.\s?[{UP}]\.\s?{SURNAME}")),
    # Адреса: вулиця і номер будинку
    ("АДРЕСА", re.compile(rf"\b(?:вул|вулиця|просп|проспект|пров|провулок|бульв|бульвар)\.?\s+[{UP}0-9][^,;\n]{{1,40}},\s*(?:буд\.?\s*)?\d+[{LO}А-Я]?(?:,\s*кв\.?\s*\d+)?")),
    # Телефони, РНОКПП, номери документів
    ("НОМЕР", re.compile(r"(?<![\d/])(?:\+?38\s?)?\(?0\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)")),
    ("НОМЕР", re.compile(r"(?<![\d/-])\d{10}(?![\d/-])")),
    ("НОМЕР", re.compile(rf"\b[{UP}]{{2}}\s?\d{{6}}\b")),
]


def predict(text: str) -> list[dict]:
    found = []
    for kind, pattern in PATTERNS:
        for m in pattern.finditer(text):
            if not any(m.start() < e and s < m.end() for s, e, _ in found):
                found.append((m.start(), m.end(), kind))
    found.sort()
    # однаковий текст — однаковий псевдонім
    labels: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}
    anns = []
    for s, e, kind in found:
        key = (kind, text[s:e])
        if key not in labels:
            counts[kind] = counts.get(kind, 0) + 1
            labels[key] = f"{kind}_{counts[kind]}"
        anns.append({"start": s, "end": e, "pseudonym": labels[key]})
    return anns


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    with open(src, encoding="utf-8-sig") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip():
                row = json.loads(line)
                fout.write(json.dumps({"id": row["id"], "annotations": predict(row["original"])}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
