"""Етап RegEx: знаходить персональні дані з передбачуваною формою."""
from __future__ import annotations

import re

from .document import Hidden

UP = "А-ЯІЇЄҐ"
LO = "а-яіїєґ'’ʼ"
NAME = rf"[{UP}][{LO}]+"
SURNAME = rf"{NAME}(?:-{NAME})?"
PATRONYMIC = rf"[{UP}][{LO}]*(?:ович|евич|євич|івн|ївн|інічн|ичн)[{LO}]*"
INITIALS = rf"[{UP}]\.\s?[{UP}]\.?"

PERSON = [
    re.compile(rf"\b{SURNAME}\s+{NAME}\s+{PATRONYMIC}\b"),   # Прізвище Ім'я По батькові
    re.compile(rf"\b{NAME}\s+{PATRONYMIC}\s+{SURNAME}\b"),   # Ім'я По батькові Прізвище
    re.compile(rf"\b{SURNAME}\s+{INITIALS}"),                # Прізвище І.Б.
    re.compile(rf"\b{INITIALS}\s?{SURNAME}\b"),              # І.Б. Прізвище
]

NUMBER = [
    re.compile(r"\bUA\d{27}\b"),                                                    # IBAN
    re.compile(rf"\b[{UP}A-Z]{{2}}\s?\d{{4}}\s?[{UP}A-Z]{{2}}\b"),                  # номерний знак
    re.compile(rf"\b[{UP}]{{2}}\s?№?\s?\d{{6}}\b"),                                 # паспорт
    re.compile(r"\b[АA]\d{4,5}\b"),                                                 # військова частина
    re.compile(r"(?<![\d/.-])(?:\+?38\s?)?\(?0\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?![\d/-])"),  # телефон
    re.compile(r"(?<![\d/.-])\d{9,10}(?![\d/.-])"),                                 # РНОКПП, ID-картка
]

MONTHS = "січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня"
INFO = [
    re.compile(r"\b(?:19|20)\d\d-[01]\d-[0-3]\d\b"),                                # 2011-04-23
    re.compile(rf"\b\d{{1,2}}(?:\.\d{{2}}\.|\s+(?:{MONTHS})\s+)\d{{4}}(?:\s+року)?(?=\s+(?:року\s+)?народження)"),
]

# Адреса: ланцюжок частин через кому, у якому є вулиця
STREET = re.compile(rf"(?:\b(?:по|на)\s+)?\b(?:вул\.|вулиц[яіюе]|просп\.|проспект[ауі]?|пров\.|провулк?[ауі]?|бульв\.|"
                    rf"бульвар[ауі]?|пл\.|площ[аіу]|шосе|узвіз|набережна)\s*[{UP}0-9][{LO}\w'’ʼ. -]*$", re.I)
PLACE = rf"[{UP}][{LO}-]+(?:\s[{UP}][{LO}-]+)?"
ADDRESS_PART = (
    rf"(?:\d{{5}}|(?:буд\.|будинок|будинку|б\.|кв\.|квартира|квартири|корп\.|корпус|офіс|приміщення|гараж|секція)\s*№?\s*\d+[{LO}А-Я/-]?\d*"
    rf"|\d+[{LO}А-Я]?(?:/\d+)?"
    rf"|(?:м\.|місто|міста|місті|с\.|село|села|селі|смт\.?|селище|сел\.)\s*{PLACE}"
    rf"|[{UP}][{LO}-]+\s+(?:область|області|обл\.|район|району|р-н|р-ну)"
    rf"|Україна|Автономна Республіка Крим)")
PART_HEAD = re.compile(ADDRESS_PART)             # частина на початку шматка (праворуч від вулиці)
PART_TAIL = re.compile(ADDRESS_PART + r"\s*$")  # частина в кінці шматка (ліворуч від вулиці)
CHUNK = re.compile(r"[^,;:\n()«»]+")


def _addresses(text: str) -> list[tuple[int, int]]:
    """Знаходить шматок із вулицею й розширює його сусідніми частинами адреси через кому.
    Шматок, у якому частина адреси займає не весь текст, обрізаємо й далі не йдемо."""
    chunks = [(m.start(), m.end()) for m in CHUNK.finditer(text)]

    def adjacent(i: int, j: int) -> bool:
        return text[chunks[i][1]:chunks[j][0]].strip() == ","

    def strip(s: int, e: int) -> tuple[int, int]:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1] in " \t.":
            e -= 1
        return s, e

    out = []
    for i, (cs, ce) in enumerate(chunks):
        m = STREET.search(text, cs, ce)
        if not m or ce - m.start() > 60:
            continue
        start, end = strip(m.start(), ce)
        grown = False
        j = i
        while j > 0 and adjacent(j - 1, j):
            ps, pe = strip(*chunks[j - 1])
            pm = PART_TAIL.search(text, ps, pe)
            if not pm:
                break
            start, grown, j = pm.start(), True, j - 1
            if pm.start() > ps:
                break
        j = i
        while j + 1 < len(chunks) and adjacent(j, j + 1):
            ns, ne = strip(*chunks[j + 1])
            nm = PART_HEAD.match(text, ns, ne)
            if not nm:
                break
            end, grown, j = nm.end(), True, j + 1
            if nm.end() < ne:
                break
        if grown:  # сама лише назва вулиці без номера будинку — не адреса
            out.append((start, end))
    return out


def find(text: str) -> list[Hidden]:
    found: list[Hidden] = []

    def add(s: int, e: int, kind: str) -> None:
        if not any(s < h.end and h.start < e for h in found):
            found.append(Hidden(s, e, kind, text[s:e], "regex"))

    for s, e in _addresses(text):
        add(s, e, "АДРЕСА")
    for kind, patterns in (("ОСОБА", PERSON), ("НОМЕР", NUMBER), ("ІНФОРМАЦІЯ", INFO)):
        for p in patterns:
            for m in p.finditer(text):
                add(m.start(), m.end(), kind)
    return sorted(found, key=lambda h: h.start)
