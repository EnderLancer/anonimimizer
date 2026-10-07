"""Synthetic persons: one consistent identity per placeholder index, varied surface forms per mention."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from silver.fill.morph import inflect_full
from silver.io import RESOURCES

_FEM_SURNAME = [
    ("ський", "ська"), ("цький", "цька"), ("зький", "зька"), ("ий", "а"), ("ій", "я"),
    ("ов", "ова"), ("ев", "ева"), ("єв", "єва"), ("ін", "іна"), ("ин", "ина"), ("їн", "їна"),
]
_DATV_COURT = [("ові", "у"), ("еві", "ю"), ("єві", "ю")]
CASE_PRIOR = (("nomn", 45), ("gent", 40), ("datv", 7), ("accs", 5), ("ablt", 3))


@dataclass
class Person:
    gender: str
    surname: str
    first: str
    patr: str

    def initials(self, space: bool) -> str:
        sep = " " if space else ""
        return f"{self.first[0]}.{sep}{self.patr[0]}."


class PersonGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        d = json.loads((RESOURCES / "names.json").read_text(encoding="utf-8"))
        self.pools = {k: ([n for n, _ in v], [w for _, w in v]) for k, v in d.items()}

    def _pick(self, key: str) -> str:
        names, weights = self.pools[key]
        return self.rng.choices(names, weights)[0]

    def new_person(self, gender: str | None = None) -> Person:
        gender = gender or ("masc" if self.rng.random() < 0.52 else "femn")
        surname = self._pick("surname")
        if gender == "femn":
            for m, f in _FEM_SURNAME:
                if surname.endswith(m):
                    surname = surname[: -len(m)] + f
                    break
        first = self._pick("male_first" if gender == "masc" else "female_first")
        patr = self._pick("male_patr" if gender == "masc" else "female_patr")
        return Person(gender, surname, first, patr)

    def choose_case(self, guessed: str | None) -> str:
        if guessed:
            return guessed
        return self.rng.choices([c for c, _ in CASE_PRIOR], [w for _, w in CASE_PRIOR])[0]

    def _court_dative(self, parts: tuple[str, str, str]) -> tuple[str, str, str]:
        if self.rng.random() > 0.7:
            return parts
        out = []
        for w in parts:
            for a, b in _DATV_COURT:
                if w.endswith(a):
                    w = w[: -len(a)] + b
                    break
            out.append(w)
        return tuple(out)  # type: ignore[return-value]

    def render(self, p: Person, case: str, first_mention: bool) -> str:
        r = self.rng.random()
        s, f, pa = inflect_full(p.surname, p.first, p.patr, p.gender, case)
        if case == "datv" and p.gender == "masc":
            s, f, pa = self._court_dative((s, f, pa))
        if first_mention or r < 0.82:
            return f"{s} {f} {pa}"
        if r < 0.92:
            return f"{s} {p.initials(space=self.rng.random() < 0.5)}"
        if r < 0.95:
            return f"{s} {p.first[0]}."
        if r < 0.975:
            return f"{f} {s}"
        return f"{p.initials(space=self.rng.random() < 0.5)} {s}"


def translit(word: str) -> str:
    table = {
        "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ie", "ж": "zh", "з": "z", "и": "y",
        "і": "i", "ї": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
        "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ь": "", "ю": "iu",
        "я": "ia", "'": "", "’": "", "-": "",
    }
    return "".join(table.get(ch, ch) for ch in word.lower())
