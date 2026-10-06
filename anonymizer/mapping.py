"""Етап Mapping values: однакові значення отримують однаковий псевдонім."""
from __future__ import annotations

import re

from .annotations import Hidden

WORD = re.compile(r"[^\W\d_][\w'’ʼ-]*")
PATRONYMIC = re.compile(r"(?:ович|евич|євич|івн|ївн|інічн|ичн)", re.I)


def person_key(value: str) -> tuple:
    """«Савченка Артема Євгеновича», «Савченко А. Є.» і «А.Є. Савченко» -> ("савче", "А", "Є").

    Прізвище порівнюємо за початком, щоб не залежати від відмінка, ім'я та по батькові — за ініціалами.
    """
    words = WORD.findall(value)
    full = [w for w in words if len(w) > 1]
    initials = [w[0].upper() for w in words if len(w) == 1]
    if len(full) >= 3:
        # Ім'я По батькові Прізвище, якщо по батькові стоїть другим
        if PATRONYMIC.search(full[1]) and not PATRONYMIC.search(full[2]):
            full = [full[2], full[0], full[1]]
        surname, initials = full[0], [w[0].upper() for w in full[1:3]]
    elif full:
        surname = full[0]
    else:
        return ("", *initials)
    s = surname.lower()
    return (s[:max(3, min(5, len(s) - 1))], *initials[:2])


def value_key(h: Hidden) -> tuple:
    if h.type == "ОСОБА":
        return (h.type, *person_key(h.text))
    return (h.type, re.sub(r"\W+", " ", h.text.lower()).strip())


def assign(hidden: list[Hidden]) -> dict[Hidden, str]:
    """Псевдоніми нумеруємо за першою появою в тексті: ОСОБА_1, ОСОБА_2, …"""
    labels: dict[tuple, str] = {}
    counts: dict[str, int] = {}
    out = {}
    for h in sorted(hidden, key=lambda h: h.start):
        key = value_key(h)
        if key not in labels:
            counts[h.type] = counts.get(h.type, 0) + 1
            labels[key] = f"{h.type}_{counts[h.type]}"
        out[h] = labels[key]
    return out
