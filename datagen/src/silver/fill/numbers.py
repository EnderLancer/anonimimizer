"""Synthetic personal identifiers for НОМЕР placeholders, chosen by context sub-type."""
from __future__ import annotations

import random
import re

CYR = "АВЕІКМНОРСТХ"
LAT = "ABEIKMHOPCTX"  # the same plate letters typed in Latin
VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # a VIN has no I, O or Q
PASSPORT_SERIES = ["АА", "АВ", "АЕ", "АК", "АМ", "АН", "АР", "АС", "ВА", "ВВ", "ВК", "ВМ", "ВР", "ЕА", "ЕЕ", "КА",
                   "КВ", "КЕ", "КМ", "КН", "КО", "КР", "КС", "МА", "МЕ", "МК", "МН", "МР", "МС", "МТ", "НА", "НЕ",
                   "НК", "НМ", "СА", "СВ", "СК", "СМ", "СН", "СО", "СР", "СС", "СТ", "ТТ", "FP", "FE"]
_SERIES_LEFT = re.compile(r"(?:№|номер\w*|сері[яїю])\s*$", re.I)
BANK_CODES = ["305299", "322001", "300346", "351005", "320649", "380805", "328209", "300528", "322313", "321842", "305749"]


class NumberGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def _digits(self, n: int, first_nonzero: bool = True) -> str:
        r = self.rng
        s = str(r.randint(1, 9)) if first_nonzero else str(r.randint(0, 9))
        return s + "".join(str(r.randint(0, 9)) for _ in range(n - 1))

    def generate(self, subtype: str, left: str = "") -> str:
        """`left` is the text right before the placeholder: «паспорт серії НОМЕР_1» must not get «серії» twice."""
        r = self.rng
        if subtype == "rnokpp":
            return str(r.randint(1, 3)) + self._digits(9, first_nonzero=False)
        if subtype == "iban":
            iban = f"UA{r.randint(10, 99)}{r.choice(BANK_CODES)}0000026{self._digits(12, first_nonzero=False)}"
            if r.random() < 0.15:  # grouped by 4
                iban = " ".join(iban[i: i + 4] for i in range(0, len(iban), 4))
            return iban
        if subtype == "card":
            return " ".join(self._digits(4, first_nonzero=False) for _ in range(4))
        if subtype == "phone":
            op = r.choice(["050", "066", "095", "099", "067", "068", "096", "097", "098", "063", "073", "093"])
            a, b, c = self._digits(3, False), self._digits(2, False), self._digits(2, False)
            return r.choice([f"+38{op}{a}{b}{c}", f"+38 ({op}) {a}-{b}-{c}", f"+38 {op} {a} {b} {c}", f"{op}{a}{b}{c}",
                             f"{op}-{a}-{b}-{c}", f"{op} {a} {b} {c}", f"({op}) {a}-{b}-{c}", f"38{op}{a}{b}{c}"])
        if subtype == "passport":
            plain = bool(_SERIES_LEFT.search(left))
            if r.random() < 0.65:
                series = r.choice(PASSPORT_SERIES)
                forms = [f"{series}{self._digits(6, False)}", f"{series} {self._digits(6, False)}", f"{series} № {self._digits(6, False)}"]
                if not plain:
                    forms += [f"серії {series} № {self._digits(6, False)}", f"серія {series} номер {self._digits(6, False)}"]
                return r.choice(forms)
            forms = [self._digits(9), f"{self._digits(9)} (УНЗР {self._digits(8, False)}-{self._digits(5, False)})"]
            return r.choice(forms if plain else forms + [f"№ {self._digits(9)}"])
        if subtype == "imei":
            return self._digits(15)
        if subtype == "svidotstvo":
            return r.choice(PASSPORT_SERIES) + r.choice(["", " ", "-"]) + self._digits(6, False)
        if subtype == "military":
            return r.choice(["А", "А", "А", "В", "Т", "АІ", "A"]) + self._digits(4, False)
        if subtype == "vin":  # 17 characters: manufacturer and model codes, then a 6-digit serial number
            return r.choice(VIN_CHARS[:23]) + "".join(r.choice(VIN_CHARS) for _ in range(10)) + self._digits(6, False)
        if subtype == "vehicle":
            abc = LAT if r.random() < 0.3 else CYR
            l1 = "".join(r.choice(abc) for _ in range(2))
            l2 = "".join(r.choice(abc) for _ in range(2))
            d = self._digits(4, False)
            return r.choice([f"{l1} {d} {l2}", f"{l1}{d}{l2}", f"{l1}-{d}-{l2}"])
        if subtype == "provadzhennia":
            # enforcement proceedings carry 8 digits in the train corpus (829 of 854 open numbers); longer ones and
            # «АВ/1234» are the forms of the number templates described in the thesis on the test generator
            form = r.choices(["8", "9-10", "letters"], [55, 25, 20])[0]
            if form == "letters":
                return "".join(r.choice(CYR) for _ in range(2)) + "/" + self._digits(4)
            return self._digits(8 if form == "8" else r.choice([9, 10]))
        # generic: spread over the realistic sub-types, plus a few serial-like codes
        choice = r.choices(["rnokpp", "passport", "iban", "phone", "serial", "code8"], [25, 20, 10, 10, 20, 15])[0]
        if choice == "serial":
            return r.choice(PASSPORT_SERIES) + r.choice(["", "-"]) + self._digits(6, False)
        if choice == "code8":
            return self._digits(8, False)
        return self.generate(choice)
