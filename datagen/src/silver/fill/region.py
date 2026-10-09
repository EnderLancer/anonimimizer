"""Which region and settlements a decision talks about, so that synthetic addresses stay in the same area."""
from __future__ import annotations

import re
from collections import Counter

KYIV = "Київ"  # the city is a region of its own in the Ukrposhta directory
# region -> inflected forms of its centre
CENTRES = {
    "Вінницька": ("Вінниця", "Вінниц(?:я|і|ю|ею)"),
    "Волинська": ("Луцьк", "Луцьк(?:а|у|ом)?"),
    "Дніпропетровська": ("Дніпро", "Дніпр(?:о|а|і|у|ом)|Дніпропетровськ(?:а|у|ом)?"),
    "Донецька": ("Донецьк", "Донецьк(?:а|у|ом)?"),
    "Житомирська": ("Житомир", "Житомир(?:а|і|у|ом)?"),
    "Закарпатська": ("Ужгород", "Ужгород(?:а|і|у|ом)?"),
    "Запорізька": ("Запоріжжя", "Запоріжж(?:я|і|ю|ям)"),
    "Івано-Франківська": ("Івано-Франківськ", "Івано-Франківськ(?:а|у|ом)?"),
    "Київська": (None, None),
    "Кіровоградська": ("Кропивницький", "Кропивницьк(?:ий|ого|ому|им)|Кіровоград(?:а|і|у|ом)?"),
    "Луганська": ("Луганськ", "Луганськ(?:а|у|ом)?"),
    "Львівська": ("Львів", "Льв(?:ів|ова|ові|ову|овом)"),
    "Миколаївська": ("Миколаїв", "Микола(?:їв|єва|єві|єву|євом)"),
    "Одеська": ("Одеса", "Одес(?:а|и|і|у|ою)"),
    "Полтавська": ("Полтава", "Полтав(?:а|и|і|у|ою)"),
    "Рівненська": ("Рівне", "Рівн(?:е|ого|ому|им)"),
    "Сумська": ("Суми", "Сум(?:и|ах|ам|ами)?"),
    "Тернопільська": ("Тернопіль", "Терноп(?:іль|оля|олі|олю|олем)"),
    "Харківська": ("Харків", "Харк(?:ів|ова|ові|ову|овом)"),
    "Херсонська": ("Херсон", "Херсон(?:а|і|у|ом)?"),
    "Хмельницька": ("Хмельницький", "Хмельницьк(?:ий|ого|ому|им)"),
    "Черкаська": ("Черкаси", "Черкас(?:и|ах|ам|ами)?"),
    "Чернівецька": ("Чернівці", "Чернівц(?:і|ів|ях|ям|ями)"),
    "Чернігівська": ("Чернігів", "Черніг(?:ів|ова|ові|ову|овом)"),
    KYIV: ("Київ", "Ки(?:їв|єва|єві|єву|євом)"),
}
REGIONS = [r for r in CENTRES if r != KYIV]

_STEM = {r[:-1]: r for r in REGIONS}  # «Вінницьк» -> «Вінницька»
_STEMS = "|".join(re.escape(s) for s in sorted(_STEM, key=len, reverse=True))
_OBLAST = re.compile(rf"(?<![\w-])({_STEMS})(?:а|ої|ій|у|ою)\s+обл(?:аст|\.)")
_COURT = re.compile(rf"(?<![\w-])({_STEMS})(?:ий|ого|ому|им)\s+(?:апеляційн|окружн|обласн)")
_CITY_LEFT = r"(?:(?<!\w)м\.\s?|(?<!\w)міст(?:а|о|і|у|ом)\s+)"
_CITY_FORMS = [(re.compile(forms), r) for r, (_, forms) in CENTRES.items() if forms]
_CITY = re.compile(rf"{_CITY_LEFT}({'|'.join(forms for _, forms in CENTRES.values() if forms)})(?![\w-])")
_PLACE = re.compile(r"(?:(?<!\w)(?:м|с|смт|с-ще|сел)\.?\s?|(?<!\w)(?:міст|сел|селищ)(?:а|о|і|у|ом|е)\s+)"
                    r"([А-ЯІЇЄҐ][а-яіїєґ'’ʼ-]+(?:[ -][А-ЯІЇЄҐ][а-яіїєґ'’ʼ-]+)?)")


def doc_region(text: str) -> str | None:
    """The region named most often: «… області», an appeal / district administrative court, or the region's centre."""
    score: Counter = Counter(_STEM[m.group(1)] for m in _OBLAST.finditer(text))
    for m in _COURT.finditer(text):
        r = _STEM[m.group(1)]
        score[KYIV if r == "Київська" else r] += 1
    for m in _CITY.finditer(text):
        score[next(r for rx, r in _CITY_FORMS if rx.fullmatch(m.group(1)))] += 1
    if "Верховн" in text and score[KYIV]:
        score[KYIV] -= 1  # the Supreme Court sits in Kyiv: «м. Київ» in its heading says nothing about the case
    best = score.most_common(1)
    return best[0][0] if best and best[0][1] > 0 else None


def doc_places(text: str) -> Counter:
    """Settlement names as written after «м.», «с.», «смт», «місто», «село», «селище» (any grammatical case)."""
    return Counter(m.group(1) for m in _PLACE.finditer(text))
