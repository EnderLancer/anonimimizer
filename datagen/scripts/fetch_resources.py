#!/usr/bin/env python3
"""Download/build the resources used by the placeholder filler into data/resources/.

- names.json      : first names, patronymics (by gender) and surnames with flattened frequencies
                    from lang-uk/name_freq_dict_uk
- addresses.json  : regions, cities, villages, streets (Overpass when reachable, embedded fallback)
- orgs.json       : institution names / military unit name templates for ІНФОРМАЦІЯ

Re-running is a no-op for files that already exist (use --force to rebuild).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silver.io import RESOURCES  # noqa: E402
from silver.fill.static_lists import CITIES, REGIONS, STREETS, VILLAGES, ORGS, MILITARY_UNIT_NAMES  # noqa: E402

RAW = "https://raw.githubusercontent.com/lang-uk/name_freq_dict_uk/master/generated/"
FILES = {
    "male_first": "male_fname_freq_dict.csv",
    "female_first": "female_fname_freq_dict.csv",
    "male_patr": "male_pname_freq_dict.csv",
    "female_patr": "female_pname_freq_dict.csv",
    "surname": "lname_freq_dict.csv",
}
BAD_CHARS = set("*_[]()/\\\"0123456789")


def _load_csv(url: str) -> list[tuple[str, float]]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out = []
    for row in csv.DictReader(io.StringIO(r.text)):
        name = row["name"].strip()
        if not name or set(name) & BAD_CHARS or not name[0].isupper() or len(name) < 3:
            continue
        out.append((name, float(row["freq_in_corpus"])))
    return out


def _flatten(items: list[tuple[str, float]], top: int) -> list[tuple[str, float]]:
    """sqrt-flatten the frequencies so that a few politicians' surnames do not dominate."""
    items = sorted(items, key=lambda x: -x[1])[:top]
    return [(n, math.sqrt(f) + 1e-6) for n, f in items]


def build_names(force: bool) -> None:
    path = RESOURCES / "names.json"
    if path.exists() and not force:
        print(f"skip {path.name}")
        return
    data = {}
    for key, fn in FILES.items():
        items = _load_csv(RAW + fn)
        top = 2500 if key == "surname" else 400
        data[key] = _flatten(items, top)
        print(f"{key}: {len(items)} raw -> {len(data[key])} kept")
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _overpass_streets(city: str) -> list[str]:
    q = f"""[out:json][timeout:25];area["name"="{city}"]["place"~"city|town"]->.a;
    way(area.a)["highway"]["name"];out tags;"""
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=40)
    r.raise_for_status()
    names = {el["tags"]["name"] for el in r.json().get("elements", []) if "name" in el.get("tags", {})}
    return sorted(n for n in names if 4 <= len(n) <= 40 and not any(ch.isdigit() for ch in n))


def build_addresses(force: bool, use_overpass: bool) -> None:
    path = RESOURCES / "addresses.json"
    if path.exists() and not force:
        print(f"skip {path.name}")
        return
    streets = set(STREETS)
    if use_overpass:
        for city in ["Київ", "Львів", "Одеса", "Харків", "Дніпро", "Вінниця", "Полтава", "Чернігів"]:
            try:
                got = _overpass_streets(city)
                streets.update(got)
                print(f"overpass {city}: {len(got)} streets")
            except Exception as e:  # noqa: BLE001
                print(f"overpass {city} failed: {e}")
    data = {"regions": REGIONS, "cities": CITIES, "villages": VILLAGES, "streets": sorted(streets)}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"addresses: {len(data['streets'])} streets, {len(CITIES)} cities")


def build_orgs(force: bool) -> None:
    path = RESOURCES / "orgs.json"
    if path.exists() and not force:
        print(f"skip {path.name}")
        return
    path.write_text(json.dumps({"orgs": ORGS, "military_names": MILITARY_UNIT_NAMES}, ensure_ascii=False), encoding="utf-8")
    print(f"orgs: {len(ORGS)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-overpass", action="store_true")
    a = ap.parse_args()
    RESOURCES.mkdir(parents=True, exist_ok=True)
    build_names(a.force)
    build_addresses(a.force, not a.no_overpass)
    build_orgs(a.force)


if __name__ == "__main__":
    main()
