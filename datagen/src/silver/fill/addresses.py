"""Synthetic residential addresses."""
from __future__ import annotations

import json
import random

from silver.io import RESOURCES


class AddressGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        d = json.loads((RESOURCES / "addresses.json").read_text(encoding="utf-8"))
        self.regions = d["regions"]
        self.cities = d["cities"]
        self.villages = d["villages"]
        self.streets = d["streets"]

    def _parts(self) -> dict:
        r = self.rng
        city, region = r.choice(self.cities)
        return {
            "city": city,
            "region": region or r.choice(self.regions),
            "street": r.choice(self.streets),
            "house": str(r.randint(1, 180)) + (r.choice(["", "", "", "-А", "/2", "Б"]) if r.random() < 0.15 else ""),
            "apt": str(r.randint(1, 240)),
            "zip": f"{r.randint(1, 99):02d}{r.randint(0, 999):03d}",
            "village": r.choice(self.villages),
            "district": r.choice(self.cities)[0],
        }

    def new_address(self) -> dict:
        p = self._parts()
        p["template"] = self.rng.choices(range(6), [30, 20, 15, 10, 15, 10])[0]
        p["with_apt"] = self.rng.random() < 0.75
        return p

    def render(self, p: dict, variant: bool = False) -> str:
        t = p["template"]
        apt_l = f", кв. {p['apt']}" if p["with_apt"] else ""
        apt_f = f", квартира {p['apt']}" if p["with_apt"] else ""
        if variant:
            t = (t + 1) % 6
        if t == 0:
            return f"м. {p['city']}, вул. {p['street']}, буд. {p['house']}{apt_l}"
        if t == 1:
            return f"вулиця {p['street']}, будинок {p['house']}{apt_f}, місто {p['city']}, {p['region']} область, {p['zip']}"
        if t == 2:
            return f"{p['zip']}, {p['region']} область, м. {p['city']}, вулиця {p['street']}, будинок {p['house']}{apt_f}"
        if t == 3:
            return f"село {p['village']}, {p['district']}ський район, {p['region']} область"
        if t == 4:
            return f"вул. {p['street']}, буд. {p['house']}{apt_l}, м. {p['city']}, {p['region']} область, {p['zip']}"
        return f"{p['region']} область, м. {p['city']}, вул. {p['street']}, {p['house']}{apt_l}"
