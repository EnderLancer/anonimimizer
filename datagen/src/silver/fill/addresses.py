"""Synthetic residential addresses: many surface formats, real street / settlement names mined from ЄДРСР."""
from __future__ import annotations

import json
import random

import pymorphy3

from silver.io import RESOURCES

_morph = pymorphy3.MorphAnalyzer(lang="uk")


def _street_locative(name: str) -> str:
    """«по вулиці Центральній» — inflect adjectival street names; genitive-of-person names stay («по вулиці Пушкіна»)."""
    words = name.split()
    out = []
    for w in words:
        parses = [p for p in _morph.parse(w) if "ADJF" in p.tag and "nomn" in p.tag and "femn" in p.tag]
        if parses:
            f = parses[0].inflect({"loct"})
            out.append((f.word[:1].upper() + f.word[1:]) if f else w)
        else:
            out.append(w)
    return " ".join(out)


class AddressGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        d = json.loads((RESOURCES / "addresses.json").read_text(encoding="utf-8"))
        self.regions = d["regions"]
        self.cities = d["cities"]
        self.villages = d["villages"]
        self.streets = d["streets"]
        self.street_types = [("вул.", "вулиця", "вулиці"), ("вул.", "вулиця", "вулиці"), ("вул.", "вулиця", "вулиці"),
                             ("просп.", "проспект", "проспекті"), ("пров.", "провулок", "провулку"), ("бульв.", "бульвар", "бульварі")]

    def _parts(self) -> dict:
        r = self.rng
        city, region = r.choice(self.cities)
        st_abbr, st_full, st_loc = r.choice(self.street_types)
        house = str(r.randint(1, 180))
        if r.random() < 0.2:
            house += r.choice(["-А", "-Б", "/2", "/1", "А", "Б", "-В"])
        return {
            "city": city, "region": region or r.choice(self.regions), "street": r.choice(self.streets),
            "st_abbr": st_abbr, "st_full": st_full, "st_loc": st_loc,
            "house": house, "apt": str(r.randint(1, 240)), "corp": str(r.randint(1, 6)),
            "zip": f"{r.randint(1, 99):02d}{r.randint(0, 999):03d}", "village": r.choice(self.villages),
            "district": r.choice(self.cities)[0], "settlement_type": r.choice(["село", "село", "с.", "смт", "селище"]),
        }

    def new_address(self) -> dict:
        p = self._parts()
        p["template"] = self.rng.randrange(14)
        p["with_apt"] = self.rng.random() < 0.7
        return p

    def render(self, p: dict, variant: bool = False) -> str:
        r = self.rng
        t = (p["template"] + (1 if variant else 0)) % 14
        apt_l = f", кв. {p['apt']}" if p["with_apt"] else ""
        apt_f = f", квартира {p['apt']}" if p["with_apt"] else ""
        st_l = f"{p['st_abbr']} {p['street']}"
        st_f = f"{p['st_full']} {p['street']}"
        reg = f"{p['region']} область"
        reg_s = f"{p['region']} обл."
        if t == 0:
            return f"м. {p['city']}, {st_l}, буд. {p['house']}{apt_l}"
        if t == 1:
            return f"{st_f}, будинок {p['house']}{apt_f}, місто {p['city']}, {reg}, {p['zip']}"
        if t == 2:
            return f"{p['zip']}, {reg}, м. {p['city']}, {st_f}, будинок {p['house']}{apt_f}"
        if t == 3:
            return f"{p['settlement_type']} {p['village']}, {p['district']}ський район, {reg}"
        if t == 4:
            return f"{st_l}, буд. {p['house']}{apt_l}, м. {p['city']}, {reg}, {p['zip']}"
        if t == 5:
            return f"{reg}, м. {p['city']}, {st_l}, {p['house']}{apt_l}"
        if t == 6:
            return f"по {p['st_loc']} {_street_locative(p['street'])}, буд. {p['house']}{apt_l}, м. {p['city']}, {reg}"
        if t == 7:
            return f"м. {p['city']}, {st_f}, {p['house']}{(', кв. ' + p['apt']) if p['with_apt'] else ''}"
        if t == 8:
            return f"{p['settlement_type']} {p['village']}, {st_l}, {p['house']}, {p['district']}ський район, {reg_s}"
        if t == 9:
            return f"{p['zip']}, м. {p['city']}, {st_l}, буд. {p['house']}, корп. {p['corp']}{apt_l}"
        if t == 10:
            return f"місто {p['city']}, {st_f}, будинок {p['house']}{apt_f}"
        if t == 11:
            return f"{reg_s}, {p['district']}ський р-н, с. {p['village']}, {st_l}, {p['house']}"
        if t == 12:
            return f"м. {p['city']}, {reg}, {st_l}, буд. {p['house']}{apt_l}, {p['zip']}"
        return f"{st_f}, {p['house']}{apt_f}, м. {p['city']}, {p['zip']}"
