#!/usr/bin/env python3
"""Build data/resources/org_names.parquet: real organisation names for ІНФОРМАЦІЯ placeholders.

    uv pip install openpyxl                               # the «orgs» extra of pyproject.toml
    uv run --no-sync scripts/fetch_orgs.py                # all three sources; downloads 6 MB + 330 MB on the first run
    uv run --no-sync scripts/fetch_orgs.py --skip edr     # without the register of legal entities
    uv run --no-sync scripts/fetch_orgs.py --force        # build again from the downloaded files
    uv run --no-sync scripts/fetch_orgs.py --force --download   # download again too

Sources (silver.orgnames):
- court    : the decisions the pipeline reads (data/train/*.parquet) - company names in quotes, state bodies;
- treasury : «Єдиний реєстр розпорядників бюджетних коштів та одержувачів бюджетних коштів» (Treasury, data.gov.ua) -
             social-protection bodies, prosecutor's offices, courts;
- edr      : «Єдиний державний реєстр юридичних осіб, фізичних осіб-підприємців та громадських формувань» (Ministry of
             Justice, data.gov.ua), UO.zip - a sample of registered companies.

Both registers are published under Creative Commons Attribution. The generator (silver.fill.orgs) uses the table when
it is there, so this script is optional. The links below point at dated files; `--treasury-url` / `--edr-url` take
newer ones from the dataset pages.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silver.exclude import ExcludedPool  # noqa: E402
from silver.io import RESOURCES, SOURCES, read_parquet_texts  # noqa: E402
from silver.normalize import normalize  # noqa: E402
from silver.orgnames import ORG_NAMES, harvest_bodies, harvest_companies, read_edr, read_treasury, write_table  # noqa: E402

CACHE = RESOURCES / "orgs"
DATASETS = "https://data.gov.ua/dataset/"
TREASURY_URL = DATASETS + "8df43bdf-4417-42b4-bc50-bd296af4058e/resource/48163e60-531d-44e0-9e97-2b80b89dda7a/download/rrk_20260901.xlsx"
EDR_URL = DATASETS + "03cc1239-3988-4451-aa0d-aadb77448714/resource/d40cc921-39bb-44fd-be06-dc02589f45c6/download/uo.zip"
MIN_COMPANY_MENTIONS = 2  # a name written once is often a typing slip
EDR_KEEP_ONE_IN = 25  # of about a million registered companies


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total, done, t0 = int(r.headers.get("Content-Length", 0)), 0, time.time()
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
                done += len(chunk)
                print(f"\r{path.name}: {done >> 20} / {total >> 20} MB, {time.time() - t0:.0f}s", end="", flush=True)
    print()
    part.rename(path)


def court_texts(stats: Counter):
    """Normalized texts of every source corpus, without the decisions the test set was made from."""
    pool = ExcludedPool()
    for corpus, path in SOURCES:
        for doc_id, raw in read_parquet_texts(path):
            text = normalize(raw)
            if pool.reason(doc_id, text):
                stats["court_excluded"] += 1
                continue
            stats[f"court_docs_{corpus}"] += 1
            yield text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", nargs="*", default=[], choices=["court", "treasury", "edr"])
    ap.add_argument("--treasury-url", default=TREASURY_URL)
    ap.add_argument("--edr-url", default=EDR_URL)
    ap.add_argument("--force", action="store_true", help="build the table again")
    ap.add_argument("--download", action="store_true", help="download the registers again")
    a = ap.parse_args()
    if ORG_NAMES.exists() and not a.force:
        print(f"skip {ORG_NAMES.name}")
        return
    rows: list[tuple[str, str, str]] = []
    stats: Counter = Counter()
    t0 = time.time()
    if "court" not in a.skip:
        texts = list(court_texts(stats))
        companies = harvest_companies(texts)
        rows += [("court", "company", n) for n, c in companies.items() if c >= MIN_COMPANY_MENTIONS]
        rows += [("court", "body", n) for n in harvest_bodies(texts)]
        print(f"court: {sum(stats[k] for k in stats if k.startswith('court_docs_'))} decisions, {time.time() - t0:.0f}s")
    if "treasury" not in a.skip:
        xlsx = CACHE / "treasury.xlsx"
        if a.download or not xlsx.exists():
            download(a.treasury_url, xlsx)
        rows += [("treasury", kind, name) for kind, name in read_treasury(xlsx)]
    if "edr" not in a.skip:
        archive = CACHE / "uo.zip"
        if a.download or not archive.exists():
            download(a.edr_url, archive)
        t1 = time.time()
        rows += [("edr", "company", name) for name in read_edr(archive, EDR_KEEP_ONE_IN)]
        print(f"edr: {time.time() - t1:.0f}s")
    write_table(rows)
    counts = Counter(f"{s}:{k}" for s, k, _ in set(rows))
    meta = {"built": time.strftime("%Y-%m-%d"), "treasury_url": a.treasury_url, "edr_url": a.edr_url,
            "licence": "court decisions: ЄДРСР open data; registers: Creative Commons Attribution (data.gov.ua)",
            "rows": dict(sorted(counts.items())), **dict(sorted(stats.items())), "seconds": round(time.time() - t0)}
    ORG_NAMES.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
