"""Real organisation names for ІНФОРМАЦІЯ, collected by scripts/fetch_orgs.py into data/resources/org_names.parquet.

Three sources, each used where it fits (silver.fill.orgs):

- «court»    - company names in quotes and state bodies as the clerks spell them, harvested from the court decisions
               the pipeline already reads (the test pool is left out by silver.exclude);
- «treasury» - social-protection bodies, prosecutor's offices and courts from the Treasury's register of budget
               holders; only for servicemen's and social cases;
- «edr»      - company names from the Ministry of Justice register of legal entities; only for commercial cases.

The table is optional: without it the generator composes every name from its own word lists.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from silver.io import RESOURCES

ORG_NAMES = RESOURCES / "org_names.parquet"
COLUMNS = ("source", "kind", "name")
KINDS = ("company", "body", "social", "prosecutor", "court")

_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'", "‘": "'"})
_RUSSIAN = re.compile(r"[ыэъёЫЭЪЁ]")
_PLACEHOLDER = re.compile(r"(?:ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_\d+")
_COMPANY_NAME = re.compile(r"[A-Za-zА-ЯІЇЄҐа-яіїєґ0-9][A-Za-zА-ЯІЇЄҐа-яіїєґ0-9'.&+\- ]{1,44}")
_LETTERS = re.compile(r"[A-Za-zА-ЯІЇЄҐа-яіїєґ]")
_VOWELS = re.compile(r"[аеиіоуюяєїАЕИІОУЮЯЄЇaeiouyAEIOUY]")
# legal forms as they stand before the quotes; the name inside the quotes is what ЄДРСР anonymises
_FORMS = (r"(?:Товариств\w+ з обмеженою відповідальністю|Товариств\w+ з додатковою відповідальністю|Приватн\w+ підприємств\w+|"
          r"Приватн\w+ акціонерн\w+ товариств\w+|Публічн\w+ акціонерн\w+ товариств\w+|Акціонерн\w+ товариств\w+|"
          r"Державн\w+ підприємств\w+|Комунальн\w+ підприємств\w+|Фермерськ\w+ господарств\w+|Дочірн\w+ підприємств\w+|"
          r"ТОВ|ПП|ПрАТ|ПАТ|АТ|ДП|КП|ФГ|ТДВ)")
_QUOTED = re.compile(_FORMS + r"\s*(?:«\s*([^«»\"\n]{2,60}?)\s*»|\"\s*([^«»\"\n]{2,60}?)\s*\")")
# words of a legal form inside a name mean the match ran over two names
_FORM_WORDS = re.compile(r"\b(?:товариств|підприємств|відповідальністю|ТОВ|ПрАТ|ПАТ)\w*\b")

# state bodies in the nominative: the head words give the case away («Головне управління», not «Головного управління»)
_BODY_HEAD = (r"(?:Головне управління|Міжрегіональне управління|Регіональне управління|Територіальне управління|"
              r"Головне територіальне управління|Департамент|Міністерство|Державна \w+ служба|Державна служба|"
              r"Національне агентство|Національна комісія|Державне агентство|Державна інспекція|Адміністрація|"
              r"Регіональний сервісний центр|Державне бюро|Національне бюро|Фонд)")
_BODY_END = r"(?:України|області|міста [А-ЯІЇЄҐ][\w'-]+|місті [А-ЯІЇЄҐ][\w'-]+|м\. ?[А-ЯІЇЄҐ][\w'-]+|регіону)"
# «… України» may go on with the place: «в Одеській області», «у місті Києві»
_BODY_PLACE = r"(?: (?:в|у) (?:[А-ЯІЇЄҐ][\w'-]+ області|місті [А-ЯІЇЄҐ][\w'-]+|м\. ?[А-ЯІЇЄҐ][\w'-]+))?"
_BODY = re.compile(r"(?<![\w«\"])(" + _BODY_HEAD + r"(?: [\w'-]+){1,9}? " + _BODY_END + _BODY_PLACE + r")(?![\w'-])")
# what must not stand inside a body's name: a verb or a conjunction means the match ran into the sentence
_BODY_STOP = re.compile(r"\b(?:про|щодо|який|яка|яке|які|що|від|для|або|звернул\w+|подал\w+|прийнял\w+|винесл\w+|є|не|за|до|на|"
                        r"при|під|із|зі|був|була|було|може|має|надал\w+|відмовил\w+|розглянул\w+|повідомил\w+)\b", re.I)
MIN_BODY_DOCS = 2  # a body that only one decision names is as likely a slip of the pattern as a body
MAX_BODY_CHARS = 130


def _squash(s: str) -> str:
    return " ".join(s.translate(_APOSTROPHES).split())


def clean_company(name: str) -> str | None:
    """A bare company name as it stands inside quotes, or None for what is not one (a placeholder, a phrase with
    a second legal form, Russian spelling, digits only)."""
    name = _squash(name).strip(" .,-")
    if not _COMPANY_NAME.fullmatch(name) or _PLACEHOLDER.search(name) or _RUSSIAN.search(name) or _FORM_WORDS.search(name):
        return None
    if len(_LETTERS.findall(name)) < 2 or len(name.split()) > 5 or name.startswith(("Про ", "Щодо ")):  # the title of an act
        return None
    return name


def harvest_companies(texts: Iterable[str]) -> Counter:
    """Company names in quotes after a legal form: spelling -> number of mentions. The spelling the clerks use most
    is kept for a name («Комтех Логістік» over «КОМТЕХ ЛОГІСТІК» when both occur)."""
    spellings: dict[str, Counter] = {}
    for text in texts:
        for m in _QUOTED.finditer(text):
            name = clean_company(m.group(1) or m.group(2))
            if name and name[0].isupper():
                spellings.setdefault(name.lower(), Counter())[name] += 1
    return Counter({c.most_common(1)[0][0]: sum(c.values()) for c in spellings.values()})


def harvest_bodies(texts: Iterable[str]) -> Counter:
    """State bodies written in the nominative: name -> number of decisions that name it."""
    docs: Counter = Counter()
    for text in texts:
        found = set()
        for m in _BODY.finditer(text):
            name = _squash(m.group(1))
            if len(name) <= MAX_BODY_CHARS and not _BODY_STOP.search(name) and not _PLACEHOLDER.search(name) and not re.search(r"\d", name):
                found.add(name)
        docs.update(found)
    return Counter({n: c for n, c in docs.items() if c >= MIN_BODY_DOCS})


# ---- the Treasury register ---------------------------------------------------------------------------------------
_SOCIAL = re.compile(r"^(?:Управління|Департамент|Відділ)(?: праці та)? соціального захисту населення [А-ЯІЇЄҐ]")
_PROSECUTOR = re.compile(r"^[А-ЯІЇЄҐ][\w'-]+ (?:обласна|міська) прокуратура$|^Офіс Генерального прокурора$")
_COURT = re.compile(r"\bсуд(?: [А-ЯІЇЄҐ][\w'-]+ області| міста Києва)?$")
_OCCUPIED = re.compile(r"Автономн|Севастопол|військов\w+ адміністрац")


def treasury_kind(name: str) -> str | None:
    """Which of the narrow groups a budget holder belongs to: «social», «prosecutor», «court» or None (the register
    is mostly schools, councils and hospitals, which court decisions do not anonymise)."""
    if len(name) > MAX_BODY_CHARS or '"' in name or "«" in name or "(" in name or _OCCUPIED.search(name) or re.search(r"\d", name):
        return None
    if _SOCIAL.search(name):
        return "social"
    if _PROSECUTOR.search(name):
        return "prosecutor"
    if _COURT.search(name) and name[0].isupper():
        return "court"
    return None


def read_treasury(xlsx: Path) -> Iterator[tuple[str, str]]:
    """(kind, name) of the selected budget holders. The name column is found by its header «Найменування»."""
    import openpyxl

    rows = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0].iter_rows(values_only=True)
    col = None
    seen = set()
    for row in rows:
        if col is None:
            col = next((i for i, v in enumerate(row) if str(v or "").strip() == "Найменування"), None)
            continue
        name = _squash(str(row[col] or ""))
        kind = treasury_kind(name)
        if kind and name not in seen:
            seen.add(name)
            yield kind, name


# ---- the register of legal entities ------------------------------------------------------------------------------
EDR_FORMS = ("ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ", "ПРИВАТНЕ ПІДПРИЄМСТВО", "АКЦІОНЕРНЕ ТОВАРИСТВО", "ПРИВАТНЕ АКЦІОНЕРНЕ ТОВАРИСТВО",
             "ПУБЛІЧНЕ АКЦІОНЕРНЕ ТОВАРИСТВО", "ФЕРМЕРСЬКЕ ГОСПОДАРСТВО", "ТОВАРИСТВО З ДОДАТКОВОЮ ВІДПОВІДАЛЬНІСТЮ", "ДОЧІРНЄ ПІДПРИЄМСТВО")
_EDR_FIELD = {f: re.compile(rf"<{f}>(.*?)</{f}>") for f in ("NAME", "OPF", "STAN", "EDRPOU")}
_EDR_QUOTED = re.compile(r"[\"«„“]\s*(.+)\s*[\"»”“]")
_ENTITIES = {"&quot;": '"', "&apos;": "'", "&amp;": "&", "&lt;": "<", "&gt;": ">"}
# written in capitals in the register and by the clerks alike
_KEEP_UPPER = {"ЛТД", "ЛТД.", "ТД", "НВП", "НВФ", "ВКФ", "ПКФ", "МП", "СП", "КО", "АГ", "ІТ", "УА", "ЮА"}


def company_case(name: str) -> str:
    """The register writes names in capitals, court decisions mostly do not: «КОМТЕХ ЛОГІСТІК» -> «Комтех Логістік».
    A short word without a vowel is an abbreviation and stays as it is («БМ ТРЕЙД» -> «БМ Трейд»)."""
    def word(w: str) -> str:
        if w in _KEEP_UPPER or (len(w) <= 4 and not _VOWELS.search(w)) or not w.isupper():
            return w
        return w[0] + w[1:].lower()

    return " ".join("-".join(word(p) for p in w.split("-")) for w in name.split(" "))


def edr_company(line: str) -> tuple[str, str] | None:
    """(ЄДРПОУ, bare name) of a registered company from one <SUBJECT> line of UO.xml, or None."""
    f = {k: (m.group(1) if (m := rx.search(line)) else "") for k, rx in _EDR_FIELD.items()}
    if f["STAN"] != "зареєстровано" or f["OPF"] not in EDR_FORMS:
        return None
    raw = f["NAME"]
    for ent, ch in _ENTITIES.items():
        raw = raw.replace(ent, ch)
    m = _EDR_QUOTED.search(raw)
    if not m or not m.group(1).isupper():
        return None
    inner = m.group(1)
    if any(q in inner for q in '"«»„“”'):
        return None  # a name with a second pair of quotes: «ФІРМА "АКАДЕМПРОМБУД"»
    name = clean_company(company_case(inner))
    return (f["EDRPOU"], name) if name else None


def read_edr(zip_path: Path, keep_one_in: int) -> Iterator[str]:
    """Names of registered companies from UO.zip: one in `keep_one_in`, chosen by the company's code, so that the
    choice does not depend on the order of the file."""
    import hashlib
    import zipfile

    with zipfile.ZipFile(zip_path) as z, z.open(z.infolist()[0]) as f:
        for raw in f:
            line = raw.decode("cp1251", errors="replace")
            if "<SUBJECT>" not in line:
                continue
            found = edr_company(line)
            if found and int(hashlib.sha1(found[0].encode()).hexdigest(), 16) % keep_one_in == 0:
                yield found[1]


# ---- the table ---------------------------------------------------------------------------------------------------
def write_table(rows: list[tuple[str, str, str]], out: Path = ORG_NAMES) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = sorted(set(rows))
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({c: [r[i] for r in rows] for i, c in enumerate(COLUMNS)}), out, compression="zstd")


def load_names(path: Path = ORG_NAMES) -> dict[tuple[str, str], list[str]]:
    """(source, kind) -> names, sorted; empty when the table was not built."""
    if not path.exists():
        return {}
    import pyarrow.parquet as pq

    t = pq.read_table(path).to_pydict()
    out: dict[tuple[str, str], list[str]] = {}
    for source, kind, name in zip(t["source"], t["kind"], t["name"]):
        out.setdefault((source, kind), []).append(name)
    return {k: sorted(v) for k, v in out.items()}
