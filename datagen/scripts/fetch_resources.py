#!/usr/bin/env python3
"""Download/build the resources used by the placeholder filler into data/resources/.

- names.json      : first names, patronymics (by gender) and surnames with flattened frequencies
                    from lang-uk/name_freq_dict_uk
- ukrposhta.parquet : the Ukrposhta address directory (settlement, street, index, house numbers) used by the
                    address generator; ukrposhta.meta.json records the checksum of the downloaded file
- orgs.json       : institution names / military unit name templates for ІНФОРМАЦІЯ
- data/train/     : the Supreme Court decisions and the 250-decision pool of the test set (exclusion list) from
                    the Hugging Face dataset lang-uk/court-decisions-uk (MIT)

Re-running is a no-op for files that already exist (use --force to rebuild).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silver.io import RESOURCES, TEST_POOL_PARQUET, VS_PARQUET  # noqa: E402
from silver.fill.static_lists import MILITARY_UNIT_NAMES, ORGS  # noqa: E402

RAW = "https://raw.githubusercontent.com/lang-uk/name_freq_dict_uk/master/generated/"
FILES = {
    "male_first": "male_fname_freq_dict.csv",
    "female_first": "female_fname_freq_dict.csv",
    "male_patr": "male_pname_freq_dict.csv",
    "female_patr": "female_pname_freq_dict.csv",
    "surname": "lname_freq_dict.csv",
}
BAD_CHARS = set("*_[]()/\\\"0123456789")

HF_DATASET = "https://huggingface.co/datasets/lang-uk/court-decisions-uk/resolve/main/"
UKRPOSHTA_URL = "https://www.ukrposhta.ua/files/shares/out/houses.zip"  # linked from index.ukrposhta.ua/find-post-index
SETTLEMENT_TYPES = {"м.", "с.", "с-ще"}
# street types as the directory writes them; the rest (хутір, урочище, присілок, …) are not streets
STREET_TYPES = {"вул.", "пров.", "просп.", "бульв.", "пл.", "проїзд", "узвіз", "майдан", "шосе", "тупик", "в’їзд", "м-р", "кв-л"}
HOUSES_KEPT = 8
_STREET_NAME = re.compile(r"[А-ЯІЇЄҐ0-9][А-ЯІЇЄҐа-яіїєґ0-9'’ʼ.\- ]{1,44}")
_HOUSE = re.compile(r"\d{1,3}(?:[-/]?[А-ЯІЇЄҐ]|/\d{1,3})?")
_TITLES = {"святого", "святої", "князя", "княгині", "гетьмана", "академіка", "генерала", "маршала", "героя", "отця",
           "митрополита", "професора", "полковника", "короля", "королеви", "патріарха", "єпископа", "лікаря", "імені"}


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


def _name_first(street: str, morph) -> str:
    """«Хмельницького Богдана» -> «Богдана Хмельницького»: the directory lists some streets surname first."""
    toks = street.split()
    if len(toks) != 2 or toks[0].lower() in _TITLES:
        return street
    surname, name = toks
    is_name = lambda w: any("Name" in p.tag for p in morph.parse(w))  # noqa: E731
    is_surname = any("Surn" in p.tag for p in morph.parse(surname)) or surname.endswith(("ського", "цького", "зького"))
    return f"{name} {surname}" if is_name(name) and not is_name(surname) and is_surname else street


def build_ukrposhta(force: bool) -> None:
    """houses.csv (cp1251, «;») -> ukrposhta.parquet: one row per settlement + street + index with a few real house numbers."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pymorphy3

    path, zip_path = RESOURCES / "ukrposhta.parquet", RESOURCES / "ukrposhta_houses.zip"
    if path.exists() and not force:
        print(f"skip {path.name}")
        return
    if not zip_path.exists() or force:
        r = requests.get(UKRPOSHTA_URL, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        zip_path.write_bytes(r.content)
    raw = zip_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("cp1251")
    morph = pymorphy3.MorphAnalyzer(lang="uk")
    fixed: dict[str, str] = {}
    cols = {k: [] for k in ("region", "district_old", "district_new", "np_type", "np", "zip", "st_type", "street", "houses")}
    total = 0
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=";")
    next(reader)
    for row in reader:
        if len(row) < 8:
            continue
        total += 1
        region, old, new, _otg, np_full, zip_code, street_full, houses = (c.strip() for c in row[:8])
        np_type, _, np_name = np_full.partition(" ")
        st_type, _, street = street_full.partition(" ")
        nums = [h for h in houses.split(",") if _HOUSE.fullmatch(h)]
        if np_type not in SETTLEMENT_TYPES or st_type not in STREET_TYPES or not nums or not re.fullmatch(r"\d{5}", zip_code) \
                or not _STREET_NAME.fullmatch(street) or not np_name:
            continue
        if street not in fixed:
            fixed[street] = _name_first(street, morph)
        step = max(1, len(nums) // HOUSES_KEPT)  # evenly spread over the street, so both small and large numbers stay
        for k, v in zip(cols, (region, old, new, np_type, np_name, zip_code, st_type, fixed[street], ",".join(nums[::step][:HOUSES_KEPT]))):
            cols[k].append(v)
    pq.write_table(pa.table(cols), path, compression="zstd")
    meta = {"url": UKRPOSHTA_URL, "sha256": hashlib.sha256(raw).hexdigest(), "downloaded": time.strftime("%Y-%m-%d"),
            "rows_source": total, "rows_kept": len(cols["np"]),
            "settlements": len(set(zip(cols["region"], cols["district_new"], cols["np_type"], cols["np"]))),
            "streets_reordered": sum(1 for k, v in fixed.items() if k != v)}
    (RESOURCES / "ukrposhta.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ukrposhta: {meta}")


def fetch_corpus(force: bool) -> None:
    for path in (VS_PARQUET, TEST_POOL_PARQUET):
        if path.exists() and not force:
            print(f"skip {path.name}")
            continue
        r = requests.get(HF_DATASET + path.name, timeout=600)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        print(f"{path.name}: {len(r.content)} bytes")


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
    a = ap.parse_args()
    RESOURCES.mkdir(parents=True, exist_ok=True)
    build_names(a.force)
    build_ukrposhta(a.force)
    build_orgs(a.force)
    fetch_corpus(a.force)


if __name__ == "__main__":
    main()
