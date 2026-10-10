"""Layer 1: label person names that ЄДРСР left open (judges, lawyers, representatives, …)."""
from __future__ import annotations

import json

from silver import patterns as P
from silver.io import RESOURCES
from silver.merge import any_overlap

_names = json.loads((RESOURCES / "names.json").read_text(encoding="utf-8"))
FIRST_NAMES = {n for k in ("male_first", "female_first") for n, _ in _names[k]}

STOP_FIRST = {"Україна", "України", "Кодекс", "Кодексу", "Закон", "Закону", "Верховний", "Верховного", "Суд", "Суду",
              "Про", "Постанова", "Постанови", "Ухвала", "Рішення", "Європейський", "Європейського", "Конституція",
              "Конституції", "Цивільний", "Цивільного", "Кримінальний", "Кримінального", "Господарський", "Велика",
              "Великої", "Палата", "Палати", "Касаційний", "Касаційного", "Апеляційний", "Апеляційного", "Суддя", "Судді",
              "Справа", "Сторінка", "Секретар", "Головуючий", "Головуюча", "Прокурор", "Адвокат", "Форма", "Довідка", "Наказ",
              "Указ", "Протокол", "Висновок", "Акт", "Серія", "Том", "Код", "Стаття", "Пункт", "Частина", "Додаток", "Банк",
              "Місто", "Село", "Вулиця", "Область", "Район", "Київ", "Львів", "Одеса", "Харків", "Дніпро"}


def _ok(text: str, s: int, e: int) -> bool:
    left = text[max(0, s - 30): s]
    if P.NOT_PERSON_LEFT.search(left):
        return False
    # an opening quote right before the name: a person only when the quote closes right after it
    # («суддю «Соколову В. В.»»); otherwise it is the start of a company name («"Дока Україна ..."»).
    # A closing quote followed by a space («"Центр-кредит" Кравчук С.П.») does not block.
    if left[-1:] in ("«", '"', "“", "„") and text[e: e + 1] not in ("»", '"', "”", "“"):
        return False
    words = text[s:e].split()
    first = words[0].strip(".")
    if first in STOP_FIRST:
        return False
    # «Ім'я ПРІЗВИЩЕ» / «ПРІЗВИЩЕ Ім'я По батькові»: accept only with a dictionary first name
    caps = [w for w in words if len(w) >= 3 and w.isalpha() and w.isupper()]
    if caps:
        lowers = [w for w in words if not w.isupper()]
        return bool(lowers) and lowers[0].strip(",") in FIRST_NAMES
    return True


def _add(found: list[dict], existing: list[dict], text: str, s: int, e: int, source: str) -> None:
    span = {"start": s, "end": e, "type": "ОСОБА", "pseudonym": None, "source": source}
    if e <= s or not _ok(text, s, e) or any_overlap(span, existing) or any_overlap(span, found):
        return
    found.append(span)


def label_persons(text: str, existing: list[dict]) -> list[dict]:
    found: list[dict] = []
    for m in P.HEADER.finditer(text):
        for nm in P.NAME_ANY.finditer(m.group(1)):
            _add(found, existing, text, m.start(1) + nm.start(), m.start(1) + nm.end(), "header")
    for m in P.ROLE_NAME.finditer(text):
        _add(found, existing, text, m.start(1), m.end(1), "role_regex")
    for m in P.IN_PERSON_OF.finditer(text):
        _add(found, existing, text, m.start(1), m.end(1), "in_person_of")
    for m in P.SIGNATURE.finditer(text):
        _add(found, existing, text, m.start(1), m.end(1), "signature")
    for m in P.FULL_ANY.finditer(text):
        _add(found, existing, text, m.start(), m.end(), "full_pib")
    # «Прізвище І.І.» / «І.І. Прізвище» anywhere: in court texts this shape is a person in practice
    for m in P.INI_ANY.finditer(text):
        _add(found, existing, text, m.start(), m.end(), "ini_any")
    return found
