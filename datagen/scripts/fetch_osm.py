#!/usr/bin/env python3
"""Download the OpenStreetMap extract of Ukraine and build the address table data/resources/osm_addresses.parquet.

    uv pip install osmium shapely             # the «osm» extra of pyproject.toml
    uv run --no-sync scripts/fetch_osm.py     # ~880 MB download, then a few minutes of processing
    uv run --no-sync scripts/fetch_osm.py --pbf ~/Downloads/ukraine-latest.osm.pbf --force

The table has the layout of the Ukrposhta one (scripts/fetch_resources.py) plus the district of a city. The address
generator (silver.fill.addresses) uses every address table it finds in data/resources/, so this script is optional:
without its output the addresses come from the Ukrposhta directory alone.

Data © OpenStreetMap contributors, Open Database License 1.0 (https://www.openstreetmap.org/copyright).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silver.io import RESOURCES  # noqa: E402
from silver.osm import build  # noqa: E402

PBF_URL = "https://download.geofabrik.de/europe/ukraine-latest.osm.pbf"
PBF = RESOURCES / "osm" / "ukraine-latest.osm.pbf"
OUT = RESOURCES / "osm_addresses.parquet"


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pbf", default=None, help=f"an extract that is already on disk (default: download to {PBF})")
    ap.add_argument("--force", action="store_true", help="download and build again")
    a = ap.parse_args()
    if OUT.exists() and not a.force:
        print(f"skip {OUT.name}")
        return
    pbf = Path(a.pbf).expanduser() if a.pbf else PBF
    if not a.pbf and (a.force or not pbf.exists()):
        download(PBF_URL, pbf)
    meta = {"url": PBF_URL, "file": pbf.name, "file_date": time.strftime("%Y-%m-%d", time.localtime(pbf.stat().st_mtime)),
            "licence": "© OpenStreetMap contributors, ODbL 1.0", **build(pbf, OUT)}
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"osm: {meta}")


if __name__ == "__main__":
    main()
