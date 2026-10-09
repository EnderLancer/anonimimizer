#!/usr/bin/env python3
"""Fetch Supreme Court decisions of 2024 from the ЄДРСР open-data dump into data/train/2024-vs-edrsr.parquet.

    uv run --no-sync scripts/fetch_edrsr.py --dump /path/to/edrsr_data_2024 [--dry-run] [--workers 6] [--sample 2000]

The dump is «Єдиний державний реєстр судових рішень за 2024 рік» on data.gov.ua (edrsr_data_2024.zip, unzipped):
documents.csv lists every decision with its court, case category and a direct link to the RTF text on
od.reyestr.court.gov.ua. Selected here: the administrative chamber's cases of public / military service, pensions,
social protection and defence - where the clerks anonymise military bodies as ІНФОРМАЦІЯ - plus a random sample of
each other chamber. RTF files are cached in data/train/edrsr_rtf/. Decisions whose only placeholders are ОСОБА are
mostly dropped, decisions already in the Hugging Face 5K file are skipped, and near-copies (templated rulings) are
removed. The pipeline reads the parquet as the corpus «vs_edrsr» (silver.io.SOURCES).
"""
from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import os
import random
import re
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from striprtf.striprtf import rtf_to_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silver.exclude import shingles  # noqa: E402
from silver.io import DATA, VS_EDRSR_PARQUET, VS_PARQUET  # noqa: E402

RTF_DIR = DATA / "train" / "edrsr_rtf"
KAS = "9921"  # Касаційний адміністративний суд Верховного Суду
OTHER_CHAMBERS = {"9911": "КГС", "9931": "КЦС", "9941": "ККС", "9951": "ВП", "9901": "ВС"}
# case categories of the administrative chamber that are taken whole (matched against the category name)
KAS_CATEGORIES = re.compile(r"(публічн\w+ служб|проходження служби|пенсійн|соціального захисту|оборони|військов|мобілізац|"
                            r"грошового забезпечення)", re.I)
PLACEHOLDER = re.compile(r"(ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_\d+")
MIN_CHARS = 1200
P_PERSONS_ONLY = 0.25   # share of decisions kept when ОСОБА is their only placeholder type
NEAR_COPY = 0.75        # share of a document's sampled word 5-grams found in an earlier kept document
SHINGLE_SAMPLE = 8      # one sampled shingle in 8 is indexed for the near-copy check
ATTEMPTS = 3            # downloads of one decision before it is given up
RETRY_STATUS = (429, 503)  # the server asks to slow down: wait and try again
# request errors that can pass on a second try; any other RequestException (a missing or invalid URL) cannot
RETRY_ERRORS = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ContentDecodingError)
PAUSE = 0.05            # seconds a worker rests after a download, to stay polite to the server


def read_dictionary(dump: Path, name: str, key: str) -> dict[str, dict]:
    with open(dump / name, encoding="utf-8", newline="") as f:
        return {r[key]: r for r in csv.DictReader(f, delimiter="\t")}


def select(dump: Path, sample: int, seed: int, known: set[str]) -> list[dict]:
    """Rows of documents.csv to fetch: every КАС decision of the selected categories, a sample of the other chambers."""
    csv.field_size_limit(10**9)
    cats = read_dictionary(dump, "cause_categories.csv", "category_code")
    forms = read_dictionary(dump, "judgment_forms.csv", "judgment_code")
    rng = random.Random(seed)
    kas, others = [], {c: [] for c in OTHER_CHAMBERS}
    seen: set[str] = set()  # a decision listed twice is fetched once (and would only be dropped as a near-copy later)
    with open(dump / "documents.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["status"] != "1" or r["doc_id"] in known or r["doc_id"] in seen:
                continue
            row = {"id": r["doc_id"], "court_code": r["court_code"], "url": r["doc_url"],
                   "category": cats.get(r["category_code"], {}).get("name", ""),
                   "judgment": forms.get(r["judgment_code"], {}).get("name", "")}
            if r["court_code"] == KAS and KAS_CATEGORIES.search(row["category"]):
                kas.append(row)
            elif r["court_code"] in others:
                others[r["court_code"]].append(row)
            else:
                continue
            seen.add(row["id"])  # only the selected ones: the dump lists the decisions of every court of the country
    out = kas
    for code, rows in others.items():
        rng.shuffle(rows)
        out += rows[:sample]
    return out


def is_cached(path: Path) -> bool:
    """A downloaded decision is there. An empty file is not one: it is downloaded again."""
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def save(path: Path, content: bytes) -> None:
    """Write through a temporary name in the same folder, so that a write cut off half-way (full disk, killed
    process) never leaves a truncated file that the next run would take for a cached one. The name belongs to this
    writer alone (process and thread): two writers of one decision - a second run on the same folder, an id listed
    twice - must not truncate or rename each other's temporary file."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    try:
        tmp.write_bytes(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_error(e: OSError) -> str:
    """Status of a failed write. It names the cause (ENOSPC, EROFS, EIO, ...): the class alone is OSError for most."""
    return f"write error: {errno.errorcode.get(e.errno, type(e).__name__)}"


_reported: set[str] = set()
_reported_lock = threading.Lock()


def report_once(e: BaseException, doc_id: object) -> str:
    """Status of an error nobody expected (a bug, most likely). The counters keep only the class name, so the first
    error of each class is printed to stderr with its traceback; the later ones are only counted."""
    name = type(e).__name__
    with _reported_lock:
        first = name not in _reported
        _reported.add(name)
        if first:
            print(f"unexpected {name} at decision {doc_id} (the first one; later ones are only counted):", file=sys.stderr)
            traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
            sys.stderr.flush()
    return f"error: {name}"


def download(session: requests.Session, row: dict) -> str:
    path = RTF_DIR / f"{row['id']}.rtf"
    if is_cached(path):
        return "cached"
    err = "not tried"
    for attempt in range(ATTEMPTS):
        try:
            resp = session.get(row["url"], timeout=30)
        except RETRY_ERRORS as e:
            err, wait = type(e).__name__, 2
        except requests.RequestException as e:
            return type(e).__name__
        else:
            if resp.status_code == 200 and resp.content[:5] == b"{\\rtf":
                try:
                    save(path, resp.content)
                except OSError as e:
                    return write_error(e)
                time.sleep(PAUSE)
                return "ok"
            if resp.status_code not in RETRY_STATUS:
                return "not rtf" if resp.status_code == 200 else f"http {resp.status_code}"
            err, wait = f"http {resp.status_code}", 5
        if attempt + 1 < ATTEMPTS:
            time.sleep(wait * (attempt + 1))
    return f"{err}, gave up after {ATTEMPTS} attempts"


def fetch(session: requests.Session, row: dict) -> tuple[dict, str]:
    """Download the RTF (cached) and return (row, status); status is 'ok', 'cached' or the error. Never raises.
    A throttled request (429 / 503), a connection error, a timeout or a broken transfer is tried again; when the
    attempts run out the status names the last failure and says that it was retried. Everything else - another HTTP
    status, an answer that is not an RTF, an invalid URL, a failed write («write error: ENOSPC») - is returned at
    once. An error of any other kind is «error: <class>», with the first traceback of each class on stderr."""
    try:
        return row, download(session, row)
    except Exception as e:  # one decision must not stop the other downloads
        return row, report_once(e, row.get("id") if isinstance(row, dict) else None)


def rtf_text(path: Path) -> str:
    raw = path.read_bytes().decode("latin-1")
    text = rtf_to_text(raw, encoding="cp1251", errors="replace")
    return re.sub(r"[ \t ]+", " ", text).strip()


def keep_rule(doc_id: str, text: str) -> str | None:
    """None when the decision is kept, otherwise why it is dropped."""
    if len(text) < MIN_CHARS:
        return "short"
    types = {m.group(1) for m in PLACEHOLDER.finditer(text)}
    if not types:
        return "no_placeholders"
    if types == {"ОСОБА"} and int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % 10000 >= P_PERSONS_ONLY * 10000:
        return "persons_only"
    return None


class NearCopies:
    """Greedy near-copy removal: a decision whose sampled word 5-grams are mostly inside an earlier one is dropped."""

    def __init__(self):
        self.index: dict[int, list[int]] = {}
        self.sizes: list[int] = []

    def is_copy(self, text: str) -> bool:
        sh = [h for h in shingles(text) if h % SHINGLE_SAMPLE == 0]
        hits = Counter(n for h in sh for n in self.index.get(h, ()))
        if sh and any(c >= NEAR_COPY * min(len(sh), self.sizes[n]) for n, c in hits.items()):
            return True
        n = len(self.sizes)
        self.sizes.append(len(sh))
        for h in sh:
            self.index.setdefault(h, []).append(n)
        return False


def main() -> int:
    """The exit code: 1 when a worker failed with an unexpected error (the parquet is still built), otherwise 0."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="folder with the unzipped edrsr_data_2024.zip (documents.csv, courts.csv, ...)")
    ap.add_argument("--out", default=str(VS_EDRSR_PARQUET))
    ap.add_argument("--sample", type=int, default=2000, help="random decisions per other chamber (КГС, КЦС, ККС, ВП, ВС)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="fetch at most this many decisions (debug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="only count what would be fetched")
    a = ap.parse_args()
    dump = Path(a.dump)

    known = set(pq.read_table(VS_PARQUET, columns=["id"]).column("id").to_pylist()) if VS_PARQUET.exists() else set()
    rows = select(dump, a.sample, a.seed, known)
    by_court = Counter(r["court_code"] for r in rows)
    cached = sum(is_cached(RTF_DIR / f"{r['id']}.rtf") for r in rows)
    print(f"selected {len(rows)} decisions: КАС {by_court[KAS]}, " + ", ".join(f"{n} {by_court[c]}" for c, n in OTHER_CHAMBERS.items())
          + f"; {cached} already cached; {len(known)} ids of the 5K file skipped", flush=True)
    if a.dry_run:
        return 0
    if a.limit:
        rows = rows[: a.limit]
    RTF_DIR.mkdir(parents=True, exist_ok=True)
    status: Counter = Counter()
    t0 = time.time()
    with requests.Session() as session, ThreadPoolExecutor(a.workers) as pool:
        session.headers["User-Agent"] = "ai-camp-datagen/1.0 (research; court-decision pseudonymisation)"
        futures = {pool.submit(fetch, session, r): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                _, st = fut.result()
            except Exception as e:  # whatever a worker dies of, the rest is downloaded and the parquet is built
                st = report_once(e, futures[fut]["id"])
            status[st] += 1
            if i % 500 == 0:
                print(f"  {i}/{len(rows)} fetched ({(time.time() - t0) / 60:.1f} min): {dict(status)}", flush=True)
    print(f"download done in {(time.time() - t0) / 60:.1f} min: {dict(status)}", flush=True)

    kept, drops, copies = [], Counter(), NearCopies()
    for r in rows:
        path = RTF_DIR / f"{r['id']}.rtf"
        if not is_cached(path):
            drops["missing"] += 1
            continue
        try:
            text = rtf_text(path)
        except Exception as e:  # a broken file must not stop the build
            drops[f"rtf_error:{type(e).__name__}"] += 1
            continue
        why = keep_rule(r["id"], text)
        if why:
            drops[why] += 1
            continue
        if copies.is_copy(text):
            drops["near_copy"] += 1
            continue
        kept.append({**{k: r[k] for k in ("id", "court_code", "category", "judgment")}, "text": text})
    table = pa.table({k: [d[k] for d in kept] for k in ("id", "text", "court_code", "category", "judgment")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, a.out)
    info = Counter(bool(re.search(r"ІНФОРМАЦІЯ_\d+", d["text"])) for d in kept)
    print(f"kept {len(kept)} decisions -> {a.out} ({info[True]} with ІНФОРМАЦІЯ); dropped {dict(drops)}")
    unexpected = sum(n for st, n in status.items() if st.startswith("error: "))
    if unexpected:  # a bug must not pass for a finished download in an unattended run
        print(f"{unexpected} of {len(rows)} downloads failed with an unexpected error (traceback above)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
