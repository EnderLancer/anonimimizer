"""Inflection of Ukrainian personal names (surname / first name / patronymic) by grammatical case.

pymorphy3 with the Ukrainian dictionary covers most names; a suffix table is the fallback for
surnames it does not know. Cases: nomn, gent, datv, accs, ablt (locative is rare in court texts).
"""
from __future__ import annotations

from functools import lru_cache

import pymorphy3

CASES = ("nomn", "gent", "datv", "accs", "ablt")
_morph = pymorphy3.MorphAnalyzer(lang="uk")


def _restore_case(src: str, out: str) -> str:
    if src.isupper():
        return out.upper()
    if src[:1].isupper():
        return out[:1].upper() + out[1:]
    return out


@lru_cache(maxsize=200_000)
def _pymorphy_inflect(word: str, gender: str, kind: str, case: str) -> str | None:
    parses = [
        p for p in _morph.parse(word)
        if kind in p.tag.grammemes and gender in p.tag.grammemes and "nomn" in p.tag.grammemes
    ]
    if not parses:
        return None
    f = parses[0].inflect({case})
    return _restore_case(word, f.word) if f else None


# ---- rule-based fallback -------------------------------------------------------------------

_MASC_SURNAME_RULES: list[tuple[str, dict[str, str]]] = [
    # (suffix, {case: replacement}) — applied to the stem after stripping the suffix
    ("ський", {"gent": "ського", "datv": "ському", "accs": "ського", "ablt": "ським"}),
    ("цький", {"gent": "цького", "datv": "цькому", "accs": "цького", "ablt": "цьким"}),
    ("зький", {"gent": "зького", "datv": "зькому", "accs": "зького", "ablt": "зьким"}),
    ("ий", {"gent": "ого", "datv": "ому", "accs": "ого", "ablt": "им"}),
    ("ій", {"gent": "ього", "datv": "ьому", "accs": "ього", "ablt": "ім"}),
    ("ов", {"gent": "ова", "datv": "ову", "accs": "ова", "ablt": "овим"}),
    ("ев", {"gent": "ева", "datv": "еву", "accs": "ева", "ablt": "евим"}),
    ("єв", {"gent": "єва", "datv": "єву", "accs": "єва", "ablt": "євим"}),
    ("ін", {"gent": "іна", "datv": "іну", "accs": "іна", "ablt": "іним"}),
    ("ин", {"gent": "ина", "datv": "ину", "accs": "ина", "ablt": "иним"}),
    ("ко", {"gent": "ка", "datv": "ку", "accs": "ка", "ablt": "ком"}),
    ("ло", {"gent": "ла", "datv": "лу", "accs": "ла", "ablt": "лом"}),
    ("а", {"gent": "и", "datv": "і", "accs": "у", "ablt": "ою"}),
    ("я", {"gent": "і", "datv": "і", "accs": "ю", "ablt": "ею"}),
    ("ь", {"gent": "я", "datv": "ю", "accs": "я", "ablt": "ем"}),
    ("й", {"gent": "я", "datv": "ю", "accs": "я", "ablt": "єм"}),
    ("", {"gent": "а", "datv": "у", "accs": "а", "ablt": "ом"}),  # consonant-final
]
_FEMN_SURNAME_RULES: list[tuple[str, dict[str, str]]] = [
    ("ська", {"gent": "ської", "datv": "ській", "accs": "ську", "ablt": "ською"}),
    ("цька", {"gent": "цької", "datv": "цькій", "accs": "цьку", "ablt": "цькою"}),
    ("зька", {"gent": "зької", "datv": "зькій", "accs": "зьку", "ablt": "зькою"}),
    ("ова", {"gent": "ової", "datv": "овій", "accs": "ову", "ablt": "овою"}),
    ("ева", {"gent": "евої", "datv": "евій", "accs": "еву", "ablt": "евою"}),
    ("єва", {"gent": "євої", "datv": "євій", "accs": "єву", "ablt": "євою"}),
    ("іна", {"gent": "іної", "datv": "іній", "accs": "іну", "ablt": "іною"}),
    ("ина", {"gent": "иної", "datv": "иній", "accs": "ину", "ablt": "иною"}),
    ("ка", {"gent": "ки", "datv": "ці", "accs": "ку", "ablt": "кою"}),
    ("га", {"gent": "ги", "datv": "зі", "accs": "гу", "ablt": "гою"}),
    ("ха", {"gent": "хи", "datv": "сі", "accs": "ху", "ablt": "хою"}),
    ("а", {"gent": "и", "datv": "і", "accs": "у", "ablt": "ою"}),
    ("я", {"gent": "і", "datv": "і", "accs": "ю", "ablt": "ею"}),
    # feminine surnames ending in a consonant, -о, -ко, -ук, -ич are indeclinable
]
_FIRST_NAME_RULES = {
    "masc": [
        ("ій", {"gent": "ія", "datv": "ію", "accs": "ія", "ablt": "ієм"}),
        ("й", {"gent": "я", "datv": "ю", "accs": "я", "ablt": "єм"}),
        ("ь", {"gent": "я", "datv": "ю", "accs": "я", "ablt": "ем"}),
        ("а", {"gent": "и", "datv": "і", "accs": "у", "ablt": "ою"}),
        ("я", {"gent": "і", "datv": "і", "accs": "ю", "ablt": "ею"}),
        ("о", {"gent": "а", "datv": "у", "accs": "а", "ablt": "ом"}),
        ("", {"gent": "а", "datv": "у", "accs": "а", "ablt": "ом"}),
    ],
    "femn": [
        ("ія", {"gent": "ії", "datv": "ії", "accs": "ію", "ablt": "ією"}),
        ("ка", {"gent": "ки", "datv": "ці", "accs": "ку", "ablt": "кою"}),
        ("га", {"gent": "ги", "datv": "зі", "accs": "гу", "ablt": "гою"}),
        ("а", {"gent": "и", "datv": "і", "accs": "у", "ablt": "ою"}),
        ("я", {"gent": "і", "datv": "і", "accs": "ю", "ablt": "ею"}),
        ("", {}),  # consonant-final female names (Любов, Нінель) are indeclinable here
    ],
}
_PATR_RULES = {
    "masc": [("ич", {"gent": "ича", "datv": "ичу", "accs": "ича", "ablt": "ичем"})],
    "femn": [("на", {"gent": "ни", "datv": "ні", "accs": "ну", "ablt": "ною"})],
}


def _rule_inflect(word: str, rules, case: str) -> str:
    low = word.lower()
    for suf, table in rules:
        if low.endswith(suf):
            if case not in table:
                return word
            stem = word[: len(word) - len(suf)] if suf else word
            return _restore_case(word, stem + table[case]) if suf else stem + table[case]
    return word


def inflect(word: str, gender: str, kind: str, case: str) -> str:
    """kind: 'Surn' | 'Name' | 'Patr'; gender: 'masc' | 'femn'; case in CASES."""
    if case == "nomn" or not word:
        return word
    if "-" in word:  # double surnames: inflect both parts
        return "-".join(inflect(p, gender, kind, case) for p in word.split("-"))
    got = _pymorphy_inflect(word, gender, kind, case)
    if got and got.lower() != word.lower() or (got and kind == "Surn" and gender == "femn"):
        return got
    if kind == "Surn":
        if gender == "femn":
            return _rule_inflect(word, _FEMN_SURNAME_RULES, case)
        return _rule_inflect(word, _MASC_SURNAME_RULES, case)
    if kind == "Name":
        return _rule_inflect(word, _FIRST_NAME_RULES[gender], case)
    return _rule_inflect(word, _PATR_RULES[gender], case)


def inflect_full(surname: str, first: str, patr: str, gender: str, case: str) -> tuple[str, str, str]:
    return (
        inflect(surname, gender, "Surn", case),
        inflect(first, gender, "Name", case),
        inflect(patr, gender, "Patr", case),
    )
