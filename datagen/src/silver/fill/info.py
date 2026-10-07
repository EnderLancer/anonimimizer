"""Values for ІНФОРМАЦІЯ placeholders: birth/death dates, e-mails, URLs, nicknames, unit / institution names."""
from __future__ import annotations

import datetime as dt
import json
import random

from silver.fill.orgs import OrgGenerator
from silver.fill.persons import Person, translit
from silver.io import RESOURCES

MONTHS_GEN = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня",
              "листопада", "грудня"]
MAIL_DOMAINS = ["gmail.com", "gmail.com", "gmail.com", "ukr.net", "ukr.net", "i.ua", "meta.ua", "outlook.com", "icloud.com", "email.ua"]


class InfoGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        d = json.loads((RESOURCES / "orgs.json").read_text(encoding="utf-8"))
        self.orgs = d["orgs"]
        self.military_names = d["military_names"]
        self.org_gen = OrgGenerator(rng)

    def _date(self, y0: int, y1: int) -> dt.date:
        y = self.rng.randint(y0, y1)
        return dt.date(y, self.rng.randint(1, 12), self.rng.randint(1, 28))

    def render_date(self, d: dt.date, style: str) -> str:
        if style == "numeric":
            return d.strftime("%d.%m.%Y")
        if style == "iso":
            return d.isoformat()
        if style == "words":
            return f"{d.day} {MONTHS_GEN[d.month - 1]} {d.year}"
        if style == "words_roku":
            return f"{d.day} {MONTHS_GEN[d.month - 1]} {d.year} року"
        return f"{d.day} {MONTHS_GEN[d.month - 1]} {d.year} року народження"

    def generate(self, subtype: str, hints: dict, person: Person | None) -> str:
        r = self.rng
        if subtype == "other":
            # unknown context: ЄДРСР marked PD here; most such placeholders are institution / unit names, the rest dates
            return self.org_gen.institution() if r.random() < 0.6 else self.generate("dob", {}, person)
        if subtype in ("dob", "death"):
            y0, y1 = (1945, 2012) if subtype == "dob" else (1995, 2024)
            d = self._date(y0, y1)
            if hints.get("style") == "numeric":
                return self.render_date(d, "numeric")
            if hints.get("no_roku"):
                return self.render_date(d, r.choice(["words", "words", "numeric"]))
            style = r.choices(["numeric", "iso", "words_roku", "words_narodzh" if subtype == "dob" else "words_roku", "words"],
                              [40, 15, 20, 15, 10])[0]
            return self.render_date(d, style)
        if subtype == "email":
            if person and r.random() < 0.7:
                local = r.choice([f"{translit(person.first)}.{translit(person.surname)}",
                                  f"{translit(person.surname)}{r.randint(1960, 2005)}",
                                  f"{translit(person.first)[0]}{translit(person.surname)}{r.randint(1, 99)}"])
            else:
                local = f"user{r.randint(1000, 99999)}"
            return f"{local}@{r.choice(MAIL_DOMAINS)}"
        if subtype == "url":
            return r.choice([f"https://www.facebook.com/profile.php?id={r.randint(10**9, 10**10)}",
                             f"https://t.me/{translit(person.surname) if person else 'user'}{r.randint(1, 999)}",
                             f"https://instagram.com/{translit(person.first) if person else 'user'}_{r.randint(1, 9999)}",
                             f"https://www.olx.ua/uk/list/user/{r.randint(10**5, 10**7)}/"])
        if subtype == "nick":
            base = translit(person.surname) if person else "user"
            return r.choice([f"@{base}{r.randint(1, 999)}", f"{base}_{r.randint(10, 99)}", f"@{base}.official"])
        if subtype == "military_name":
            return self.org_gen.unit_name()
        if subtype == "org":
            return self.org_gen.institution()
        return self.render_date(self._date(1950, 2010), "numeric")
