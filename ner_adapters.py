"""Адаптери NER-моделей: будь-яка модель (spaCy, Hugging Face token-classification, GLiNER) зводиться
до одного формату, з яким працює решта пайплайну, — списку EntitySpan (позиції символів у тексті + мітка).

Модель задається рядком "тип:назва[?опції]":
    spacy:models/uk_ner_web_trf_13class
    hf:dslim/bert-base-NER?aggregation=first&min_score=0.6&device=0
    gliner:urchade/gliner_multi-v2.1?labels=person,address&threshold=0.4

Адаптер повертає «сирі» мітки моделі (PER, PERS, person…; BIO-префікси прибрано). У типи челенджу
(ОСОБА, АДРЕСА, НОМЕР, ІНФОРМАЦІЯ) їх переводить map_labels() за label_map, тож змінити відповідність
можна без повторного інференсу: у кеші лежать сирі мітки.

Новий формат моделі = новий клас з методом predict(texts), зареєстрований через @register("тип").

Приватні моделі з Hugging Face: токен береться зі змінної середовища HF_TOKEN (у CI — секрет репозиторію).
У рядок моделі токен не пишемо: рядок потрапляє в логи й у назву файлу кешу.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from urllib.parse import parse_qsl

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
# сирі мітки (після upper() і без BIO-префікса) -> тип; мітки, яких тут немає, відкидаються.
# Окрім класичних NER-міток — імена з PII-моделей, які ділять ім'я на частини (напр. Wismut/nym-pii-multilingual:
# GIVEN_NAME/SURNAME; на тесті 98–99% таких фрагментів — gold ОСОБА). Решту PII-міток (DATE, CITY, COMPANY_NAME…)
# розмітка челенджу здебільшого не маскує, тож їх вмикають явно через --ner-label.
# Моделі, донавчені на розмітці челенджу (train/, напр. davidmelash/wechsel_base), віддають самі типи: вони йдуть як є.
DEFAULT_LABEL_MAP = {
    **{t: t for t in TYPES},
    **dict.fromkeys(("PERS", "PER", "PERSON"), "ОСОБА"),
    **dict.fromkeys(("GIVEN_NAME", "GIVENNAME", "FIRST_NAME", "FIRSTNAME", "MIDDLE_NAME", "MIDDLENAME", "SURNAME",
                     "LAST_NAME", "LASTNAME", "FAMILY_NAME", "NAME", "FULL_NAME", "FULLNAME", "NAME_STUDENT"), "ОСОБА"),
}
# Набори міток, які вмикаються одним --ner-label @назва. "pii": інші типи PII-моделей (Wismut/nym-pii-multilingual,
# piiranha тощо). Тип із * — слабка частина (див. ner_postprocess): місто чи область лише в складі адреси.
LABEL_PRESETS = {
    "pii": {
        **dict.fromkeys(("STREET_ADDRESS", "STREET_NAME", "STREET", "ADDRESS"), "АДРЕСА"),
        **dict.fromkeys(("BUILDING_NUMBER", "BUILDINGNUM", "SECONDARY_ADDRESS", "CITY", "STATE", "ZIP_CODE", "ZIPCODE",
                         "COUNTRY"), "АДРЕСА*"),
        **dict.fromkeys(("PASSPORT", "DRIVERS_LICENSE", "DRIVERLICENSENUM", "TAX_ID", "TAXNUM", "SSN", "SOCIALNUM",
                         "GOVERNMENT_ID", "IDCARDNUM", "IBAN", "CREDIT_DEBIT_CARD", "CREDITCARDNUMBER", "ACCOUNT_NUMBER",
                         "ACCOUNTNUM", "ROUTING_NUMBER", "SWIFT_BIC", "PHONE", "TELEPHONENUM", "FAX_NUMBER",
                         "LICENSE_PLATE", "CUSTOMER_ID", "EMPLOYEE_ID", "MEDICAL_RECORD_NUMBER"), "НОМЕР"),
        **dict.fromkeys(("DATE_OF_BIRTH", "DATEOFBIRTH", "EMAIL", "URL", "USERNAME"), "ІНФОРМАЦІЯ"),
    },
}
BIO = re.compile(r"^[BIESLU]-", re.I)


@dataclass(frozen=True)
class EntitySpan:
    start: int        # позиція першого символа в тексті документа
    end: int          # позиція після останнього символа (не включно)
    text: str         # text[start:end]
    label: str        # сира мітка моделі (після map_labels — тип челенджу)
    score: float = 1.0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, x) -> "EntitySpan":
        if isinstance(x, (list, tuple)):  # старий формат ner_cache.json: [start, end, text], лише PERS
            return cls(int(x[0]), int(x[1]), x[2], "PERS")
        return cls(int(x["start"]), int(x["end"]), x["text"], x["label"], float(x.get("score", 1.0)))


def _is_word(ch: str) -> bool:
    return ch.isalnum() or ch in "'’ʼ-"


def normalize(text: str, raw, expand_words: bool = True, min_score: float = 0.0, join_spaces: bool = False) -> list[EntitySpan]:
    """raw: (start, end, мітка, score). Прибирає BIO-префікси, пробіли по краях, за потреби розширює
    фрагмент до меж слів (субтокени «Павлов» + «ський») і склеює фрагменти з однаковою міткою, що
    перетинаються або стикуються. join_spaces: склеювати й ті, між якими лише пробіли в межах рядка
    (токени B-PER I-PER без агрегації або «Павловський» + «Олег Іванович» окремими групами)."""
    items = []
    for s, e, label, score in raw:
        if score < min_score:
            continue
        s, e = max(0, int(s)), min(len(text), int(e))
        if expand_words:
            while 0 < s < len(text) and _is_word(text[s - 1]) and _is_word(text[s]):
                s -= 1
            while 0 < e < len(text) and _is_word(text[e - 1]) and _is_word(text[e]):
                e += 1
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            items.append([s, e, BIO.sub("", str(label)).upper(), float(score)])
    items.sort()
    merged: list[list] = []
    for it in items:
        last = merged[-1] if merged else None
        if last and last[2] == it[2] and (it[0] <= last[1] or join_spaces and re.fullmatch(r"[ \t\xa0]+", text[last[1]:it[0]])):
            last[1], last[3] = max(last[1], it[1]), max(last[3], it[3])
        else:
            merged.append(it)
    return [EntitySpan(s, e, text[s:e], label, score) for s, e, label, score in merged]


def map_labels(spans: list[EntitySpan], label_map: dict[str, str | None] | None = None) -> list[EntitySpan]:
    """Сирі мітки -> типи челенджу (з * — «слабка» частина, див. ner_postprocess.assemble). Мітка без
    відповідності (або з порожнім значенням) відкидається. Лише заміна мітки: обробку даних робить ner_postprocess."""
    label_map = DEFAULT_LABEL_MAP if label_map is None else label_map
    return [replace(sp, label=t) for sp in spans if (t := label_map.get(sp.label.upper()))]


def windows(text: str, max_chars: int, overlap: int):
    """Вікна (a, b) не довші за max_chars, що перекриваються на ~overlap символів і ріжуться по пробілах.
    Потрібні моделям з обмеженою довжиною входу (512 токенів), а судові рішення значно довші."""
    n = len(text)
    if n <= max_chars:
        yield 0, n
        return
    a = 0
    while True:
        b = min(n, a + max_chars)
        if b < n:
            cut = b
            while cut > a + max_chars // 2 and not text[cut - 1].isspace():
                cut -= 1
            if cut > a + max_chars // 2:
                b = cut
        yield a, b
        if b >= n:
            return
        nxt = max(a + 1, b - overlap)
        while nxt < b and not text[nxt - 1].isspace():
            nxt += 1
        a = nxt


def hf_token() -> str | None:
    """Токен Hugging Face для приватних моделей (None — лише публічні)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


# ---- адаптери ----

ADAPTERS: dict[str, type] = {}


def register(kind: str):
    def deco(cls):
        ADAPTERS[kind] = cls
        return cls
    return deco


class NerAdapter:
    """Інтерфейс: тексти -> для кожного список EntitySpan з сирими мітками моделі."""

    def predict(self, texts: list[str]) -> list[list[EntitySpan]]:
        return [self.predict_one(t) for t in texts]

    def predict_one(self, text: str) -> list[EntitySpan]:
        raise NotImplementedError


@register("spacy")
class SpacyAdapter(NerAdapter):
    """spaCy-пайплайн (у т. ч. spaCy-моделі з Hugging Face, встановлені як пакет). Фрагменти вже
    вирівняні по токенах, тому межі не розширюємо."""

    def __init__(self, name: str, batch_size: int = 1):
        import spacy
        if not os.path.exists(name) and re.fullmatch(r"[\w.-]+/[\w.-]+", name):  # org/repo на HF: теку без .whl
            from huggingface_hub import snapshot_download
            name = snapshot_download(name, ignore_patterns=["*.whl"], token=hf_token())
        self.nlp = spacy.load(name)
        self.batch_size = batch_size

    def predict(self, texts):
        return [normalize(t, [(e.start_char, e.end_char, e.label_, 1.0) for e in doc.ents], expand_words=False)
                for t, doc in zip(texts, self.nlp.pipe(texts, batch_size=self.batch_size))]


@register("hf")
class HFTokenClassificationAdapter(NerAdapter):
    """Будь-яка token-classification модель з Hugging Face через transformers.pipeline.

    Розуміє обидва формати виходу пайплайна: з агрегацією ({entity_group, start, end, score, word})
    і без неї ({entity: "B-PER", ...} по субтокенах). Якщо токенізатор не дає позицій (start/end = None),
    фрагмент шукається в тексті за word. Довгі тексти ріжуться на вікна, що вміщаються в модель: з fast-токенізатором
    по токенах (stride — перекриття в токенах), інакше по символах (max_chars, overlap).
    quantize=1 — динамічна int8-квантизація лінійних шарів: на CPU швидше в ~1,5–2 рази, точність може трохи змінитися.
    trust_remote_code=1 — виконати код з репозиторію моделі (токенізатор LiBERTa у modern_liberta_*); лише для репозиторіїв, яким довіряєте.
    """

    def __init__(self, name: str, aggregation: str = "simple", min_score: float = 0.0, device: int | str | None = None,
                 max_chars: int = 1500, overlap: int = 200, batch_size: int = 8, join_spaces: int = 1,
                 revision: str | None = None, stride: int = 64, quantize: int = 0, trust_remote_code: int = 0, pipe=None):
        if pipe is None:
            from transformers import pipeline
            # device=None у transformers означає cuda:0, якщо CUDA «доступна»; на ZeroGPU це падає поза @spaces.GPU.
            # За замовчуванням — CPU; GPU вмикається явно: &device=0.
            pipe = pipeline("token-classification", model=name, aggregation_strategy=aggregation,
                            device=-1 if device is None else device,
                            revision=revision and str(revision), token=hf_token(),
                            trust_remote_code=bool(trust_remote_code))
            if quantize:
                import torch
                pipe.model = torch.ao.quantization.quantize_dynamic(pipe.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.pipe, self.stride = pipe, stride
        self.min_score, self.max_chars, self.overlap, self.batch_size = min_score, max_chars, overlap, batch_size
        self.join_spaces = bool(join_spaces)
        tok = getattr(pipe, "tokenizer", None)
        limit = getattr(tok, "model_max_length", None) or 512
        if limit > 100_000:  # «необмежено» в токенізаторі: беремо з конфігу моделі
            limit = getattr(getattr(getattr(pipe, "model", None), "config", None), "max_position_embeddings", 512)
        self.tok, self.limit = tok, limit
        # Окрема копія fast-токенізатора для нарізання на вікна. Пайплайн під час першого виклику вмикає у своєму
        # токенізаторі обрізання до ліміту моделі, і воно лишається ввімкненим: далі encode() всього документа віддавав би
        # лише перші ~512 токенів, і з другого документа решта тексту не потрапляла б у жодне вікно.
        self.enc = None
        backend = getattr(tok, "backend_tokenizer", None)
        if backend is not None:
            self.enc = copy.deepcopy(backend)
            if hasattr(self.enc, "no_truncation"):
                self.enc.no_truncation()
                self.enc.no_padding()

    def _fast(self) -> bool:
        return self.enc is not None

    def _n_tokens(self, s: str) -> int:
        if self._fast():  # без попередження «sequence length is longer than … 512» на довгих текстах
            return len(self.enc.encode(s, add_special_tokens=True).ids)
        return len(self.tok(s, add_special_tokens=True)["input_ids"])

    def _fits(self, s: str) -> bool:
        return self.tok is None or self._n_tokens(s) <= self.limit

    def _token_windows(self, text: str):
        """Документ токенізується один раз; вікно — до limit токенів, перекриття stride токенів, межі — на початку слова.
        Для моделей з англомовним словником кирилиця йде майже по токену на символ, тож вікна по символах
        виходять або задовгими, або напівпорожніми."""
        offs = self.enc.encode(text, add_special_tokens=False).offsets
        n = len(offs)
        size = self.limit - self.tok.num_special_tokens_to_add() - 8  # запас: на межі вікна токенізація може трохи змінитися
        stride = min(self.stride, size // 4)
        word_start = lambda k: k == 0 or offs[k][0] > offs[k - 1][1]  # перед токеном пробіл
        i = 0
        while i < n:
            j = min(n, i + size)
            if j < n:
                k = j
                while k > i + size // 2 and not word_start(k):
                    k -= 1
                if k > i + size // 2:
                    j = k
            yield offs[i][0], offs[j - 1][1]
            if j >= n:
                return
            k = max(i + 1, j - stride)
            while k < j and not word_start(k):
                k += 1
            i = k

    def _windows(self, text: str, a: int = 0, b: int | None = None, max_chars: int | None = None):
        if a == 0 and b is None and self._fast():
            for x, y in self._token_windows(text):
                if self._fits(text[x:y]):
                    yield x, y
                else:
                    yield from self._windows(text, x, y, self.max_chars)
            return
        b = len(text) if b is None else b
        max_chars = max_chars or self.max_chars
        for x, y in windows(text[a:b], max_chars, min(self.overlap, max_chars // 4)):
            if self._fits(text[a + x:a + y]) or max_chars < 50:
                yield a + x, a + y
            else:
                yield from self._windows(text, a + x, a + y, max_chars // 2)

    @staticmethod
    def _locate(chunk: str, word: str, cursor: int) -> tuple[int, int] | None:
        w = re.sub(r"^(##|▁|Ġ)", "", word or "").strip()
        if not w:
            return None
        i = chunk.lower().find(w.lower(), cursor)
        return (i, i + len(w)) if i >= 0 else None

    def predict_one(self, text):
        wins = list(self._windows(text))
        outs = self.pipe([text[a:b] for a, b in wins], batch_size=self.batch_size)
        if wins and outs and isinstance(outs[0], dict):  # пайплайн на одному тексті повертає плаский список
            outs = [outs]
        raw = []
        for (a, b), ents in zip(wins, outs):
            chunk, cursor = text[a:b], 0
            for x in ents:
                s, e = x.get("start"), x.get("end")
                if s is None or e is None:
                    loc = self._locate(chunk, x.get("word", ""), cursor)
                    if loc is None:
                        continue
                    s, e = loc
                cursor = e
                raw.append((a + s, a + e, x.get("entity_group") or x.get("entity") or "", float(x.get("score", 1.0))))
        return normalize(text, raw, expand_words=True, min_score=self.min_score, join_spaces=self.join_spaces)


@register("gliner")
class GlinerAdapter(NerAdapter):
    """GLiNER (zero-shot NER): мітки задаються текстом, вихід {start, end, text, label, score}."""

    def __init__(self, name: str, labels: list[str] | str = ("person",), threshold: float = 0.5,
                 max_chars: int = 1000, overlap: int = 150, device: str | None = None, join_spaces: int = 1,
                 revision: str | None = None, model=None):
        if model is None:
            from gliner import GLiNER
            model = GLiNER.from_pretrained(name, revision=revision and str(revision), token=hf_token())
            if device:
                model = model.to(device)
        self.model = model
        self.labels = [labels] if isinstance(labels, str) else list(labels)
        self.threshold, self.max_chars, self.overlap, self.join_spaces = threshold, max_chars, overlap, bool(join_spaces)

    def predict_one(self, text):
        raw = []
        for a, b in windows(text, self.max_chars, self.overlap):
            for x in self.model.predict_entities(text[a:b], self.labels, threshold=self.threshold):
                raw.append((a + x["start"], a + x["end"], x["label"], float(x.get("score", 1.0))))
        return normalize(text, raw, expand_words=True, join_spaces=self.join_spaces)


# ---- фабрика і кеш ----

def _value(v: str):
    if "," in v:
        return [x for x in v.split(",") if x]
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    return v


def parse_spec(spec: str) -> tuple[str, str, dict]:
    """"hf:org/model?min_score=0.5" -> ("hf", "org/model", {"min_score": 0.5}). Рядок без відомого
    префікса (наприклад, шлях C:\\models\\x) вважається spaCy-моделлю."""
    kind, sep, rest = spec.partition(":")
    if not sep or kind not in ADAPTERS:
        kind, rest = "spacy", spec
    name, _, query = rest.partition("?")
    return kind, name, {k: _value(v) for k, v in parse_qsl(query)}


def adapter_options(kind: str) -> list[str]:
    """Опції, які приймає адаптер (параметри конструктора, крім назви і підставних об'єктів для тестів)."""
    import inspect
    params = inspect.signature(ADAPTERS[kind].__init__).parameters
    return [p for p in params if p not in ("self", "name", "pipe", "model")]


def make_spec(kind: str, name: str, **opts) -> str:
    """("hf", "org/model", min_score=0.6, overlap="") -> "hf:org/model?min_score=0.6". Порожні значення
    пропускаються (діє значення адаптера за замовчуванням), невідомі для цього типу опції — помилка."""
    from urllib.parse import urlencode
    if kind not in ADAPTERS:
        raise ValueError(f"невідомий тип моделі {kind!r}; доступні: {', '.join(ADAPTERS)}")
    bad = [k for k in opts if k not in adapter_options(kind)]
    if bad:
        raise ValueError(f"тип {kind} не має опцій {bad}; доступні: {', '.join(adapter_options(kind))}")
    q = {k: ",".join(v) if isinstance(v, (list, tuple)) else v for k, v in opts.items() if v not in (None, "")}
    return f"{kind}:{name}" + (f"?{urlencode(q, safe=',/')}" if q else "")


def load_adapter(spec: str) -> NerAdapter:
    kind, name, opts = parse_spec(spec)
    return ADAPTERS[kind](name, **opts)


def cache_path(spec: str, legacy_spec: str | None = None, legacy_path: str | None = None) -> str:
    """Окремий кеш на кожну модель; для моделі за замовчуванням — історичний ner_cache.json."""
    if legacy_spec and legacy_path and parse_spec(spec) == parse_spec(legacy_spec):
        return legacy_path
    return os.path.join(HERE, "ner_cache", re.sub(r"[^\w.-]+", "_", spec).strip("_") + ".json")


def run_ner(rows: list[dict], spec: str, cache_file: str, save_every: int = 10) -> dict[str, list[EntitySpan]]:
    """Сирі фрагменти моделі для кожного документа {id, original}; інференс лише для документів, яких немає в кеші."""
    cache = json.load(open(cache_file, encoding="utf8")) if os.path.exists(cache_file) else {}
    todo = [r for r in rows if str(r["id"]) not in cache]
    if todo:
        adapter = load_adapter(spec)
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        t0 = time.time()
        for i, r in enumerate(todo, 1):
            cache[str(r["id"])] = [sp.to_json() for sp in adapter.predict([r["original"]])[0]]
            if i % save_every == 0 or i == len(todo):
                json.dump(cache, open(cache_file, "w", encoding="utf8"), ensure_ascii=False)
                print(f"NER {spec}: {i}/{len(todo)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return {str(r["id"]): [EntitySpan.from_json(x) for x in cache[str(r["id"])]] for r in rows}


def shard(rows: list[dict], i: int, n: int) -> list[dict]:
    """i-та з n частин документів, збалансованих за довжиною тексту (найдовші — у найменш завантажену частину)."""
    load, part = [0] * n, {}
    for k in sorted(range(len(rows)), key=lambda k: (-len(rows[k]["original"]), k)):
        j = min(range(n), key=lambda j: (load[j], j))
        load[j] += len(rows[k]["original"])
        part[k] = j
    return [r for k, r in enumerate(rows) if part[k] == i]


def main(argv=None):
    """CLI для CI: інференс частини документів і злиття частин у кеш, який читає run.py.

    python ner_adapters.py infer data.jsonl --ner SPEC --shard 2/8 --out part2.json
    python ner_adapters.py merge data.jsonl --ner SPEC part*.json      # -> ner_cache/<модель>.json
    """
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["infer", "merge"])
    ap.add_argument("data")
    ap.add_argument("parts", nargs="*", help="merge: файли частин")
    ap.add_argument("--ner", required=True)
    ap.add_argument("--shard", default="0/1", help="infer: i/n, i від 0")
    ap.add_argument("--out", help="infer: файл результату (за замовчуванням — кеш моделі)")
    a = ap.parse_intermixed_args(argv)
    rows = [json.loads(l) for l in open(a.data, encoding="utf8") if l.strip()]
    target = cache_path(a.ner)
    if a.cmd == "infer":
        i, n = map(int, a.shard.split("/"))
        part = shard(rows, i, n)
        print(f"частина {i}/{n}: {len(part)} документів, {sum(len(r['original']) for r in part)} символів", file=sys.stderr)
        out = a.out or target
        run_ner(part, a.ner, out)
        if not os.path.exists(out):  # порожня частина: файл однаково потрібен для злиття
            json.dump({}, open(out, "w", encoding="utf8"))
        return
    cache = json.load(open(target, encoding="utf8")) if os.path.exists(target) else {}
    for p in a.parts:
        cache.update(json.load(open(p, encoding="utf8")))
    missing = [str(r["id"]) for r in rows if str(r["id"]) not in cache]
    if missing:
        sys.exit(f"у кеші немає {len(missing)} документів, напр. {missing[:5]}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    json.dump(cache, open(target, "w", encoding="utf8"), ensure_ascii=False)
    print(target)


if __name__ == "__main__":
    main()
