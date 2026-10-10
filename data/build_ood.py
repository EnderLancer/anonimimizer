#!/usr/bin/env python3
"""Будує OOD-набір (тексти поза судовим доменом) для оцінки NER-моделі: data/ood/ood_cases.jsonl.

Кроки:
    python build_ood.py sample --out ood/01_candidates.jsonl          # вибірка з Hugging Face (потрібен пакет datasets)
    python build_ood.py export ood/01_candidates.jsonl ood/tasks/     # INSTRUCTIONS.md + <id>.md на документ
    … агенти читають tasks/<id>.md і пишуть ood/answers/<id>.json (формат у GUIDELINES.md)
    python build_ood.py import ood/01_candidates.jsonl ood/answers/ --out ood/ood_cases.jsonl
    python build_ood.py check ood/ood_cases.jsonl --sample 20          # документи з позначками для ручної перевірки
    python build_ood.py diff ood/ood_cases.jsonl submission.jsonl ood/tasks_adj/   # розбіжності gold ↔ модель для арбітражу

Джерела: Goader/kobza (CC0; новини = source «ukrainian-news», веб = hplt-2.0 / fineweb-2 / cultura-x, художня
література = ubertext2.0), HuggingFaceFW/finewiki config «uk» (CC-BY-SA 4.0). Квоти добираються регулярними
виразами за ключовими словами, не моделлю: відбір моделлю завищив би recall.

Gold-файл у схемі тесту: {id, original, pseudonymized, annotations: [{start, end, entity, pseudonym, kind}]} плюс
поля source / subset / origin / url / title / license. data/evaluate.py читає його як є.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as ev  # noqa: E402

TYPES = ev.TYPES
KINDS = {
    "ОСОБА": ("callsign",),
    "АДРЕСА": ("residence", "property", "office", "event", "other"),
    "НОМЕР": ("phone", "iban", "account", "card", "rnokpp", "passport", "license", "plate", "vin", "unit_code", "case", "other"),
    "ІНФОРМАЦІЯ": ("dob", "death", "email", "unit_name", "mil_org", "other"),
}
MIN_CHARS, MAX_CHARS = 400, 3000
SEED = 20261010

# --- вибірка -----------------------------------------------------------------------------------------------------

WAR = re.compile(r"бригад|військов(?:а|ої|ій|у|ою) частин|\bТЦК\b|\b[АТ]\d{4}\b|батальйон|\bполк[уа]?\b|угрупованн|"
                 r"механізован|штурмов|десантн", re.I)
MONEY = re.compile(r"\bIBAN\b|\bUA\d{27}\b|\+38|\(0\d{2}\)|\b0\d{2}[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}\b|[\w.-]+@[\w-]+\.[a-z]{2,}|"
                   r"картк[аиу]\s+(?:моно|приват)|monobank|реквізити", re.I)
BIO = re.compile(r"\(\s*\d{1,2}\s+[а-яії]+\s+\d{4}|народ(?:ився|илася|жений|жена)|\bнар\.\s*\d|\d{4}\s*[—–-]\s*\d{4}\s*\)")
MILITARY_WIKI = re.compile(r"окрем(?:а|ий|е)\s+\S+\s+(?:бригад|полк|батальйон)|військова частина|\b[АТ]\d{4}\b|"
                           r"(?:бригад|полк|батальйон|дивізі)\S*\s+(?:ЗСУ|Збройних|Національної|НГУ)", re.I)
CONTACTS = re.compile(r"за адресою|\bвул\.|вулиц[яіею]|просп\.|\bтел\.|телефон|\+38|e-?mail|@|графік роботи|"
                      r"\bбуд\.|\bкв\.|\bофіс\b", re.I)
COURT = re.compile(r"ВСТАНОВИ[ВЛ]|Іменем України|ІМЕНЕМ УКРАЇНИ|ЄДРСР|головуюч\w+ судд|у справі №|справа №", re.I)
RUSSIAN = re.compile(r"[ыэъёЫЭЪЁ]")
LETTER = re.compile(r"[^\W\d_]")
CYRILLIC = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")


class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "blockquote", "section"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def strip_html(text: str) -> str:
    if "<" not in text:
        return html.unescape(text)
    p = _Text()
    p.feed(text)
    p.close()
    return "".join(p.parts)


def clean(text: str) -> str:
    text = strip_html(text).replace("\r", "").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, limit: int = MAX_CHARS) -> tuple[str, bool]:
    """Обрізає до limit символів на межі абзацу, інакше речення, інакше слова."""
    if len(text) <= limit:
        return text, False
    head = text[:limit]
    for pat in (r"\n", r"[.!?»]\s", r"\s"):
        cuts = [m.end() for m in re.finditer(pat, head)]
        cuts = [c for c in cuts if c >= MIN_CHARS]
        if cuts:
            return head[:cuts[-1]].rstrip(), True
    return head.rstrip(), True


def usable(text: str) -> str | None:
    """Чому текст не годиться, або None."""
    if len(text) < MIN_CHARS:
        return "short"
    letters = len(LETTER.findall(text))
    if not letters or len(CYRILLIC.findall(text)) / letters < 0.7:
        return "not_ukrainian"
    if RUSSIAN.search(text):
        return "russian"
    if COURT.search(text):
        return "court"
    return None


def doc_id(source: str, text: str) -> str:
    return f"{source}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def shingles(text: str, n: int = 5) -> set[int]:
    words = re.findall(r"[^\W\d_]+|\d+", text.lower())
    return {hash(" ".join(words[i: i + n])) for i in range(len(words) - n + 1)}


class Reservoir:
    """Вибірка k елементів з потоку з однаковою ймовірністю (детерміновано за seed)."""

    def __init__(self, k: int, rng: random.Random):
        self.k, self.rng, self.items, self.seen = k, rng, [], 0

    def offer(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.k:
                self.items[j] = item


# subset -> (джерело, квота, фільтр(row, text) -> bool); перший підхожий subset у порядку списку
def subsets(scale: float = 1.0) -> list[tuple[str, str, int, object]]:
    war_era = lambda r: str(r.get("timestamp") or "") >= "2022-02-24"  # noqa: E731
    return [
        ("news", "war", round(40 * scale), lambda r, t: war_era(r) and len(WAR.findall(t)) >= 2),
        ("news", "money", round(20 * scale), lambda r, t: bool(MONEY.search(t))),
        ("news", "random", round(40 * scale), lambda r, t: True),
        ("wiki", "military", round(20 * scale), lambda r, t: bool(MILITARY_WIKI.search(t[:600]))),
        ("wiki", "bio", round(40 * scale), lambda r, t: bool(BIO.search(t[:400]))),
        ("wiki", "random", round(40 * scale), lambda r, t: True),
        ("web", "contacts", round(40 * scale), lambda r, t: len(CONTACTS.findall(t)) >= 3),
        ("web", "ubertext", round(30 * scale), lambda r, t: r.get("source") == "ubertext2.0" and r.get("subsource") not in ("news", "wikipedia")),
        ("web", "random", round(30 * scale), lambda r, t: r.get("source") != "ubertext2.0" or r.get("subsource") not in ("news", "wikipedia")),
    ]


ORIGIN_SOURCES = {"kobza": ("news", "web"), "finewiki": ("wiki",)}


def source_of(row: dict, origin: str) -> str | None:
    if origin == "finewiki":
        return "wiki"
    if row.get("source") == "ukrainian-news":
        return "news"
    if row.get("source") == "ubertext2.0" and row.get("subsource") == "wikipedia":
        return None
    if "wikipedia.org" in (row.get("url") or ""):  # копії Вікіпедії у веб-краулах: Вікіпедія йде окремим джерелом
        return None
    return "web"


SHARDS = {"kobza": ("Goader/kobza", "000_ukr_corpus_00000.parquet"),
          "finewiki": ("HuggingFaceFW/finewiki", "data/ukwiki/000_00000.parquet")}
COLUMNS = ("text", "id", "url", "timestamp", "subsource", "source", "title", "wikiname")


def rows_of(origin: str, path: str | None):
    """Рядки одного parquet-шарда (streaming через datasets читає kobza по 12 рядків на секунду, шард цілком — швидше)."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    if not path:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        repo, filename = SHARDS[origin]
        path = hf_hub_download(repo, filename, repo_type="dataset")
    pf = pq.ParquetFile(path)
    cols = [c for c in pf.schema.names if c in COLUMNS]
    for batch in pf.iter_batches(batch_size=2000, columns=cols):
        yield from batch.to_pylist()


def sample(args) -> None:
    rng = random.Random(args.seed)
    plan = subsets(args.scale)
    pools = {(src, name): Reservoir(k, random.Random(f"{args.seed}:{src}:{name}")) for src, name, k, _ in plan}
    streams = [("kobza", args.kobza_file, args.max_rows), ("finewiki", args.wiki_file, args.max_rows // 2)]
    rejected = Counter()
    for origin, path, budget in streams:
        n = 0
        for row in rows_of(origin, path):
            n += 1
            if n > budget:
                break
            if n % 10000 == 0:
                print(f"{origin}: {n} рядків, у пулах {sum(len(p.items) for p in pools.values())}", file=sys.stderr)
            src = source_of(row, origin)
            if not src:
                continue
            text, cut = truncate(clean(row["text"]))
            why = usable(text)
            if why:
                rejected[(src, why)] += 1
                continue
            for s, subset, _, ok in plan:
                if s == src and ok(row, text):
                    pools[(s, subset)].offer({"id": doc_id(src, text), "original": text, "source": src, "subset": subset,
                                              "origin": origin, "subsource": row.get("source") or row.get("wikiname"),
                                              "url": row.get("url"), "title": row.get("title"), "truncated": cut,
                                              "license": "CC0-1.0 (Kobza)" if origin == "kobza" else "CC-BY-SA-4.0"})
                    break
            mine = [p for (s, _), p in pools.items() if s in ORIGIN_SOURCES[origin]]
            if n >= args.min_rows and all(p.seen >= 20 * p.k for p in mine):  # резервуари вже достатньо випадкові
                break
    rows = [r for p in pools.values() for r in p.items]
    rows = dedup(rows)
    rng.shuffle(rows)
    rows.sort(key=lambda r: (r["source"], r["subset"]))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} документів -> {args.out}", file=sys.stderr)
    for (src, subset), p in pools.items():
        got = sum(1 for r in rows if r["source"] == src and r["subset"] == subset)
        print(f"  {src}/{subset}: {got}/{p.k} (кандидатів {p.seen})", file=sys.stderr)
    for (src, why), c in sorted(rejected.items()):
        print(f"  відкинуто {src}/{why}: {c}", file=sys.stderr)


def dedup(rows: list[dict], near: float = 0.5) -> list[dict]:
    out, seen, shingle_sets = [], set(), []
    for r in rows:
        if r["id"] in seen:
            continue
        sh = shingles(r["original"])
        if any(len(sh & other) >= near * min(len(sh), len(other)) for other in shingle_sets if sh and other):
            continue
        seen.add(r["id"])
        shingle_sets.append(sh)
        out.append(r)
    return out


# --- export / import ----------------------------------------------------------------------------------------------

def fname(doc_id) -> str:
    """Назва файла для документа: «news:a1b2» -> «news_a1b2» (двокрапка недопустима в іменах файлів Windows)."""
    return str(doc_id).replace(":", "_")


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def guidelines() -> str:
    return open(os.path.join(HERE, "ood", "GUIDELINES.md"), encoding="utf-8").read()


def export(args) -> None:
    rows = load_jsonl(args.candidates)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "INSTRUCTIONS.md"), "w", encoding="utf-8") as f:
        f.write(guidelines())
        f.write("\n\n## Як працювати з задачами\n\nКожен файл `<id>.md` у цій теці — один документ: заголовок з метаданими, "
                "далі текст між рядками `=== ТЕКСТ ===` і `=== КІНЕЦЬ ===`. Прочитайте документ цілком, випишіть усі сутності за "
                "інструкцією і запишіть відповідь у `answers/<id>.json` (та сама назва, розширення .json). Рядок `text` копіюйте "
                "з тексту точно.\n")
    batches = defaultdict(list)
    for n, r in enumerate(rows):
        with open(os.path.join(args.out_dir, fname(r['id']) + ".md"), "w", encoding="utf-8") as f:
            f.write(f"# {r['id']}\n\nДжерело: {r['source']} / {r.get('subset')} ({r.get('origin')}); "
                    f"назва: {r.get('title') or '—'}\n\n=== ТЕКСТ ===\n{r['original']}\n=== КІНЕЦЬ ===\n")
        batches[n % args.batches].append(r["id"])
    with open(os.path.join(args.out_dir, "batches.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(batches.items())}, f, ensure_ascii=False, indent=1)
    print(f"{len(rows)} задач у {args.out_dir}, {len(batches)} партій", file=sys.stderr)


_FOLD = str.maketrans({"’": "'", "ʼ": "'", "`": "'", "‘": "'", "–": "-", "—": "-", "«": '"', "»": '"', "„": '"', "“": '"',
                       "”": '"', " ": " "})


def fold(text: str) -> str:
    """Та сама довжина, уніфіковані апострофи, лапки, тире і нерозривні пробіли: пошук рядка агента в тексті."""
    return text.translate(_FOLD)


def _word_char(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch == "_" or bool(unicodedata.combining(ch)))  # знак наголосу — частина слова


def _inside_word(text: str, s: int, e: int) -> bool:
    """Чи входження text[s:e] — частина більшого слова: поруч літера або апостроф між літерами («Мар'яна»).
    Дефіс слова не склеює («Жансема-художника», «Льовочкін-Бойко»: складене ім'я агент дає цілим рядком, і в місці
    збігу виграє довший), англійське «'s» теж («Vallejo's»)."""
    before, after = text[s - 1] if s > 0 else "", text[e] if e < len(text) else ""
    if _word_char(text[s]) and (_word_char(before) or (before == "'" and s > 1 and _word_char(text[s - 2]))):
        return True
    if _word_char(text[e - 1]):
        if _word_char(after):
            return True
        if after == "'" and e + 1 < len(text) and _word_char(text[e + 1]):
            possessive = text[e + 1] in "sS" and not _word_char(text[e + 2] if e + 2 < len(text) else "")
            return not possessive
    return False


def occurrences(text: str, needle: str, excluded: list[str] = (), whole: bool = True) -> list[tuple[int, int]]:
    """Усі входження needle цілими словами, крім тих, що лежать усередині входжень рядків excluded.
    whole=False — будь-яке входження підрядком (для текстів, де слова злиплися)."""
    t, n = fold(text), fold(needle).strip()
    if not n:
        return []
    banned = []
    for ex in excluded:
        x = fold(ex)
        pos = t.find(x)
        while pos >= 0:
            banned.append((pos, pos + len(x)))
            pos = t.find(x, pos + 1)
    out, pos = [], t.find(n)
    while pos >= 0:
        s, e = pos, pos + len(n)
        if not (whole and _inside_word(t, s, e)) and not any(bs <= s and e <= be for bs, be in banned):
            out.append((s, e))
        pos = t.find(n, pos + 1)
    return out


def _normalize_entity(item: dict, doc: str, warnings: list[str]) -> dict | None:
    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
        warnings.append(f"{doc}: пропущено запис без text: {item!r}")
        return None
    try:
        typ = ev.normalize_label(str(item.get("type", "")).replace("_", ""))[0]
    except ev.FormatError:
        warnings.append(f"{doc}: невідомий тип {item.get('type')!r} для {item['text']!r}, пропущено")
        return None
    kind = item.get("kind") or None
    if kind and kind not in KINDS[typ]:
        warnings.append(f"{doc}: невідомий kind {kind!r} для {typ} {item['text']!r}, замінено на other")
        kind = "other" if "other" in KINDS[typ] else None
    ent = item.get("id") or item.get("person")
    return {"text": item["text"], "type": typ, "kind": kind, "id": str(ent) if ent else None,
            "except": [x for x in (item.get("except") or []) if isinstance(x, str)]}


def spans_for(text: str, entities: list[dict], doc: str, warnings: list[str]) -> list[dict]:
    """Записи агента -> неперетинні фрагменти {start, end, type, kind, entity}; довший рядок виграє у місці збігу."""
    found = []
    for n, e in enumerate(entities):
        occ = occurrences(text, e["text"], e["except"])
        if not occ:  # джерело зі злиплими словами («доцентАЛЕКСАНДРОВ»): агент скопіював рядок точно, беремо підрядком
            occ = occurrences(text, e["text"], e["except"], whole=False)
        if not occ:
            warnings.append(f"{doc}: не знайдено в тексті {e['type']} {e['text']!r}")
        key = e["id"] or f"{e['type']}:{fold(e['text']).strip().lower()}"
        for s, t in occ:
            found.append({"start": s, "end": t, "type": e["type"], "kind": e["kind"], "entity": key, "order": n})
    found.sort(key=lambda x: (-(x["end"] - x["start"]), x["order"], x["start"]))
    taken: list[dict] = []
    for f in found:
        if not any(f["start"] < t["end"] and t["start"] < f["end"] for t in taken):
            taken.append(f)
    for f in taken:
        f.pop("order")
    return sorted(taken, key=lambda x: x["start"])


def pseudonymize(text: str, spans: list[dict]) -> tuple[list[dict], str]:
    """Псевдоніми за порядком першої появи сутності кожного типу; текст із замінами як у тесті."""
    numbers: dict[str, dict[str, int]] = defaultdict(dict)
    anns = []
    for s in spans:
        n = numbers[s["type"]].setdefault(s["entity"], len(numbers[s["type"]]) + 1)
        anns.append({"start": s["start"], "end": s["end"], "entity": text[s["start"]:s["end"]],
                     "pseudonym": f"{s['type']}_{n}", **({"kind": s["kind"]} if s["kind"] else {})})
    out, pos = [], 0
    for a in anns:
        out += [text[pos:a["start"]], a["pseudonym"]]
        pos = a["end"]
    out.append(text[pos:])
    return anns, "".join(out)


def read_answer(path: str, doc: str, warnings: list[str]) -> list[dict] | None:
    try:
        data = json.load(open(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        warnings.append(f"{doc}: відповідь не читається ({e})")
        return None
    items = data.get("entities") if isinstance(data, dict) else data
    if not isinstance(items, list):
        warnings.append(f"{doc}: у відповіді немає списку entities")
        return None
    return [e for e in (_normalize_entity(it, doc, warnings) for it in items) if e]


def import_answers(rows: list[dict], in_dir: str, warnings: list[str]) -> list[dict]:
    out = []
    for r in rows:
        path = os.path.join(in_dir, fname(r['id']) + ".json")
        if not os.path.exists(path):
            warnings.append(f"{r['id']}: відповіді немає")
            continue
        entities = read_answer(path, r["id"], warnings)
        if entities is None:
            continue
        spans = spans_for(r["original"], entities, r["id"], warnings)
        anns, pseudo = pseudonymize(r["original"], spans)
        out.append({**{k: v for k, v in r.items() if k != "truncated"}, "pseudonymized": pseudo, "annotations": anns})
    return out


def report(rows: list[dict]) -> str:
    by = defaultdict(Counter)
    kinds = defaultdict(Counter)
    docs = Counter()
    for r in rows:
        docs[r["source"]] += 1
        for a in r["annotations"]:
            t = a["pseudonym"].rsplit("_", 1)[0]
            by[r["source"]][t] += 1
            by["усього"][t] += 1
            kinds[t][a.get("kind") or "—"] += 1
    docs["усього"] = sum(docs.values())
    lines = ["# OOD-набір: склад", "", "| Джерело | Документів | " + " | ".join(TYPES) + " |", "|---|---|" + "---|" * len(TYPES)]
    for src in [s for s in ("news", "wiki", "web") if s in docs] + ["усього"]:
        lines.append(f"| {src} | {docs[src]} | " + " | ".join(str(by[src][t]) for t in TYPES) + " |")
    lines += ["", "Підвиди (kind):", ""]
    for t in TYPES:
        lines.append(f"- {t}: " + ", ".join(f"{k} {c}" for k, c in kinds[t].most_common()))
    empty = sum(1 for r in rows if not r["annotations"])
    lines += ["", f"Документів без сутностей: {empty}.", ""]
    return "\n".join(lines)


def do_import(args) -> None:
    rows = load_jsonl(args.candidates)
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    warnings: list[str] = []
    out = import_answers(rows, args.answers, warnings)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for w in warnings:
        print("  " + w, file=sys.stderr)
    text = report(out)
    if args.report:
        open(args.report, "w", encoding="utf-8").write(text)
    print(text, file=sys.stderr)
    print(f"{len(out)} з {len(rows)} документів -> {args.out}; попереджень {len(warnings)}", file=sys.stderr)


# --- check / diff -------------------------------------------------------------------------------------------------

def marked(text: str, anns: list[dict]) -> str:
    out, pos = [], 0
    for a in sorted(anns, key=lambda a: a["start"]):
        out += [text[pos:a["start"]], f"[{text[a['start']:a['end']]}|{a['pseudonym']}" + (f"/{a['kind']}" if a.get("kind") else "") + "]"]
        pos = a["end"]
    out.append(text[pos:])
    return "".join(out)


def check(args) -> None:
    rows = load_jsonl(args.gold)
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    elif args.sample:
        rows = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    for r in rows:
        print(f"===== {r['id']} ({r['source']}/{r.get('subset')}; {len(r['annotations'])} сутностей)\n{marked(r['original'], r['annotations'])}\n")


def diff(args) -> None:
    """Для кожного документа — список розбіжностей між gold і submission для арбітражу другим агентом."""
    gold = load_jsonl(args.gold)
    pred = {str(r["id"]): r for r in load_jsonl(args.submission)}
    os.makedirs(args.out_dir, exist_ok=True)
    n_docs = n_diff = 0
    for r in gold:
        g = ev.spans_from_annotations(r["annotations"], len(r["original"]), r["id"])
        p = ev.spans_from_annotations(pred.get(str(r["id"]), {}).get("annotations", []), len(r["original"]), r["id"])
        pairs = ev.match(g, p, strict=False)
        gi, pi = {a for a, _ in pairs}, {b for _, b in pairs}
        missed = [s for i, s in enumerate(g) if i not in gi]
        extra = [s for i, s in enumerate(p) if i not in pi]
        if not missed and not extra:
            continue
        n_docs += 1
        n_diff += len(missed) + len(extra)
        lines = [f"# {r['id']}: розбіжності gold ↔ модель", "",
                 "Модель не знайшла (є в gold; перевірте, чи сутність справді треба розмічати):", ""]
        lines += [f"- {s.type} [{r['original'][s.start:s.end]}] … {ev_context(r['original'], s)}" for s in missed] or ["- (немає)"]
        lines += ["", "Модель знайшла, а в gold немає (перевірте, чи це не пропуск розмітки):", ""]
        lines += [f"- {s.type} [{r['original'][s.start:s.end]}] … {ev_context(r['original'], s)}" for s in extra] or ["- (немає)"]
        lines += ["", "Рішення запишіть правкою файла answers/<id>.json: додайте пропущені сутності, приберіть зайві, лишіть решту.", ""]
        open(os.path.join(args.out_dir, fname(r['id']) + ".md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"{n_docs} документів з розбіжностями ({n_diff} фрагментів) -> {args.out_dir}", file=sys.stderr)


def ev_context(text: str, s, width: int = 60) -> str:
    a, b = max(0, s.start - width), min(len(text), s.end + width)
    return ("…" if a else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="вибірка кандидатів з Hugging Face")
    s.add_argument("--out", default=os.path.join(HERE, "ood", "01_candidates.jsonl"))
    s.add_argument("--scale", type=float, default=1.0, help="множник квот (1.0 = 300 документів)")
    s.add_argument("--max-rows", type=int, default=600_000, help="скільки рядків kobza переглянути (finewiki — половину)")
    s.add_argument("--min-rows", type=int, default=100_000, help="менше рядків не переглядати, навіть якщо квоти набрано")
    s.add_argument("--kobza-file", default=None, help="локальний parquet-шард kobza (за замовчуванням — з Hugging Face)")
    s.add_argument("--wiki-file", default=None, help="локальний parquet-шард finewiki/ukwiki")
    s.add_argument("--seed", type=int, default=SEED)
    s.set_defaults(func=sample)
    e = sub.add_parser("export", help="задачі для агентів")
    e.add_argument("candidates"); e.add_argument("out_dir")
    e.add_argument("--batches", type=int, default=20)
    e.set_defaults(func=export)
    i = sub.add_parser("import", help="відповіді агентів -> gold")
    i.add_argument("candidates"); i.add_argument("answers")
    i.add_argument("--out", default=os.path.join(HERE, "ood", "ood_cases.jsonl"))
    i.add_argument("--report", default=None, help="куди записати REPORT.md")
    i.add_argument("--ids", nargs="*", default=None)
    i.set_defaults(func=do_import)
    c = sub.add_parser("check", help="документи з позначками для ручної перевірки")
    c.add_argument("gold"); c.add_argument("--sample", type=int, default=0); c.add_argument("--ids", nargs="*", default=None)
    c.add_argument("--seed", type=int, default=SEED)
    c.set_defaults(func=check)
    d = sub.add_parser("diff", help="розбіжності gold ↔ submission для арбітражу")
    d.add_argument("gold"); d.add_argument("submission"); d.add_argument("out_dir")
    d.set_defaults(func=diff)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
