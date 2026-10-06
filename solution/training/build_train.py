"""Збирає тренувальні дані для CRF.

Джерело: 2024-court-decisions.parquet (ЄДРСР, плейсхолдери ОСОБА_n/АДРЕСА_n/НОМЕР_n/ІНФОРМАЦІЯ_n).
Плейсхолдери замінюються на синтетичні значення (імена з NER-UK/Ukr-Synth, згенеровані адреси,
номери, дати), щоб модель бачила "справжні" форми, а не рядок "ОСОБА_1".
Тестова вибірка тут НЕ використовується.
"""
import gzip
import pickle
import random
import re
import sys

import pandas as pd

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pseudonymizer.text import tokenize  # noqa: E402

D = os.environ.get("ANON_TRAIN_DATA", "data/train").rstrip("/") + "/"


def parquet_path():
    for p in (D + "2024-court-decisions.parquet", D + "court-decisions-uk/2024-court-decisions.parquet"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("2024-court-decisions.parquet не знайдено у " + D)

random.seed(7)
PH = re.compile(r"(ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_(\d+)")
UP = "АБВГДЕЄЖЗИІКЛМНОПРСТУФХЦЧШЩЮЯ"


def load_names():
    names = set()
    cur = []
    for line in open(D + "ner-uk/v2.0/iob/train.iob", encoding="utf8"):
        p = line.split()
        if len(p) == 2 and p[1] in ("B-PERS", "I-PERS"):
            if p[1] == "B-PERS" and cur:
                names.add(" ".join(cur)); cur = []
            cur.append(p[0])
        else:
            if cur:
                names.add(" ".join(cur)); cur = []
    synth = set()
    cur = []
    with gzip.open(D + "Ukr-Synth/data/train.conllu.gz", "rt", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i > 3_000_000:
                break
            p = line.rstrip("\n").split("\t")
            if len(p) == 10 and p[9] in ("B-PER", "I-PER"):
                if p[9] == "B-PER" and cur:
                    synth.add(" ".join(cur)); cur = []
                cur.append(p[1])
            else:
                if cur:
                    synth.add(" ".join(cur)); cur = []
    ok = lambda n: all(w[0].isupper() and w.isalpha() or "'" in w or "-" in w for w in n.split()) and len(n.split()) <= 3
    pool = [n for n in (names | synth) if ok(n) and 1 <= len(n.split())]
    print("name pool", len(pool), file=sys.stderr)
    return pool


NAMES = load_names()
CITIES = ["Київ", "Харків", "Одеса", "Дніпро", "Львів", "Запоріжжя", "Вінниця", "Полтава", "Черкаси", "Суми",
          "Миколаїв", "Житомир", "Чернігів", "Рівне", "Тернопіль", "Луцьк", "Ужгород", "Кропивницький", "Біла Церква", "Бровари"]
STREETS = ["Шевченка", "Лесі Українки", "Франка", "Грушевського", "Незалежності", "Соборна", "Центральна", "Миру",
           "Героїв Майдану", "Садова", "Польова", "Миколаївська", "Київська", "Перемоги", "Гагаріна", "Молодіжна"]
MONTHS = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня", "листопада", "грудня"]


def rd(n):
    return "".join(random.choice("0123456789") for _ in range(n))


def gen_person():
    n = random.choice(NAMES)
    w = n.split()
    r = random.random()
    if len(w) >= 2 and r < 0.12:
        return f"{w[0]} {random.choice(UP)}.{random.choice(UP)}."
    if len(w) >= 2 and r < 0.2:
        return f"{w[0]} {random.choice(UP)}. {random.choice(UP)}."
    if len(w) >= 2 and r < 0.26:
        return f"{random.choice(UP)}. {random.choice(UP)}. {w[0]}"
    return n


def gen_address():
    c = random.choice(CITIES)
    s = random.choice(STREETS)
    r = random.random()
    if r < 0.3:
        return f"{c}, вул. {s}, буд. {random.randint(1, 150)}, кв. {random.randint(1, 200)}"
    if r < 0.5:
        return f"м. {c}, вулиця {s}, {random.randint(1, 99)}"
    if r < 0.65:
        return f"{rd(5)}, {random.choice(['Київська', 'Львівська', 'Одеська', 'Харківська'])} область, м. {c}, вул. {s}, {random.randint(1,120)}"
    if r < 0.8:
        return f"с. {random.choice(['Нове', 'Липове', 'Зелене', 'Дубове'])}, {random.choice(['Черкаський', 'Бердичівський', 'Обухівський'])} район, вул. {s}, {random.randint(1,80)}"
    return f"м. {c}, просп. {s}, буд. {random.randint(1,200)}"


def gen_number(ctx):
    c = ctx.lower()
    if re.search(r"(тел|моб|номер\s*телефон|абонент)", c):
        return random.choice([f"+380{rd(9)}", f"0{rd(2)} {rd(3)} {rd(2)} {rd(2)}", f"+38 (0{rd(2)}) {rd(3)}-{rd(2)}-{rd(2)}", f"0{rd(9)}", f"+38{rd(10)}"])
    if re.search(r"(паспорт|id|ід|картк)", c) and "банків" not in c:
        return random.choice([f"{random.choice(UP)}{random.choice(UP)} {rd(6)}", f"{random.choice(UP)}{random.choice(UP)}{rd(6)}", rd(9)])
    if re.search(r"(рнокпп|іпн|ідентифікац|податков)", c):
        return rd(10)
    if re.search(r"(iban|рахунок|р/р)", c):
        return f"UA{rd(2)} {rd(6)} {rd(19)}"
    if re.search(r"(номерн|д\.н\.з|реєстраційн|державн)", c):
        L = random.choice(UP)
        return f"{random.choice('АВЕІКМНОРСТХ')}{random.choice('АВЕІКМНОРСТХ')}{rd(4)}{random.choice('АВЕІКМНОРСТХ')}{random.choice('АВЕІКМНОРСТХ')}"
    if re.search(r"(карт)", c):
        return f"{rd(4)} {rd(4)} {rd(4)} {rd(4)}"
    return random.choice([rd(10), rd(9), f"{random.choice(UP)}{random.choice(UP)} {rd(6)}", f"+380{rd(9)}", f"{random.choice(UP)}{random.choice(UP)}{rd(4)}{random.choice(UP)}{random.choice(UP)}"])


def gen_info(ctx):
    c = ctx.lower()
    if re.search(r"(e-?mail|ел\.? ?пошт|пошт)", c):
        return random.choice([f"user{rd(3)}@gmail.com", f"{random.choice(['ivan','olena','petro'])}{rd(2)}@ukr.net"])
    if re.search(r"(http|сайт|сторінк|facebook|telegram|instagram|url)", c):
        return random.choice([f"https://www.facebook.com/profile.php?id={rd(10)}", f"https://t.me/user{rd(4)}", f"www.site{rd(2)}.com.ua"])
    r = random.random()
    d, m, y = random.randint(1, 28), random.randint(1, 12), random.randint(1940, 2006)
    if r < 0.5:
        return f"{d:02d}.{m:02d}.{y}"
    if r < 0.75:
        return f"{d} {MONTHS[m-1]} {y} року"
    if r < 0.9:
        return f"{d:02d}.{m:02d}.{y} року народження"
    return f"{y} року народження"


# Незамасковані ПІБ суддів/адвокатів тощо: слабка розмітка регулярками (прізвище + ініціали)
INIT = r"[А-ЯІЇЄҐ]\.\s?[А-ЯІЇЄҐ]\."
SURN = r"[А-ЯІЇЄҐ][а-яіїєґ'’ʼ]{2,}(?:-[А-ЯІЇЄҐ][а-яіїєґ'’ʼ]+)?"
PERS_RULE = re.compile(rf"(?:{SURN}\s+{INIT})|(?:{INIT}\s*{SURN})")


def render(text):
    """text із плейсхолдерами -> (новий текст, список (start,end,type))."""
    vals, out, spans, pos = {}, [], [], 0
    cur = 0
    for m in PH.finditer(text):
        out.append(text[pos:m.start()])
        cur += m.start() - pos
        typ, k = m.group(1), m.group(2)
        key = (typ, k)
        if key not in vals:
            ctx = text[max(0, m.start() - 40):m.start()]
            vals[key] = {"ОСОБА": lambda: gen_person(), "АДРЕСА": gen_address,
                         "НОМЕР": lambda: gen_number(ctx), "ІНФОРМАЦІЯ": lambda: gen_info(ctx)}[typ]()
        v = vals[key]
        out.append(v)
        spans.append((cur, cur + len(v), typ))
        cur += len(v)
        pos = m.end()
    out.append(text[pos:])
    new = "".join(out)
    # слабкі ОСОБА за правилом ініціалів (лише там, де не перетинається з уже відомими)
    for m in PERS_RULE.finditer(new):
        s, e = m.start(), m.end()
        if not any(s < b and a < e for a, b, _ in spans):
            spans.append((s, e, "ОСОБА"))
    return new, sorted(spans)


def bio(text, spans):
    toks = tokenize(text)
    labels = []
    si = 0
    for (a, b) in toks:
        while si < len(spans) and spans[si][1] <= a:
            si += 1
        if si < len(spans) and spans[si][0] <= a < spans[si][1]:
            t = spans[si][2]
            labels.append(("B-" if a == spans[si][0] or (labels and labels[-1] == "O") else "I-") + t
                          if (not labels or labels[-1][2:] != t or a == spans[si][0]) else "I-" + t)
        else:
            labels.append("O")
    return toks, labels


def main(n_docs=2500, max_sents=70000):
    df = pd.read_parquet(parquet_path())
    df = df[df.text.str.len() < 60000].sample(n_docs, random_state=3)
    data = []
    for text in df.text:
        # ділимо на «рядки» ~ речення, щоб зберегти розумну довжину послідовностей
        chunks = re.split(r"(?<=[.;:!?])\s+|\n+", text)
        buf = []
        for ch in chunks:
            buf.append(ch)
            if sum(len(x) for x in buf) > 200:
                data.append(" ".join(buf)); buf = []
        if buf:
            data.append(" ".join(buf))
    ent = [d for d in data if PH.search(d)]
    non = [d for d in data if not PH.search(d)]
    random.shuffle(non)
    sel = ent + non[: int(len(ent) * 0.5)]
    random.shuffle(sel)
    sel = sel[:max_sents]
    out = []
    for d in sel:
        new, spans = render(d)
        toks, labels = [], []
        tk = tokenize(new)
        si = 0
        lab = []
        prev = None
        for (a, b) in tk:
            while si < len(spans) and spans[si][1] <= a:
                si += 1
            if si < len(spans) and spans[si][0] <= a < spans[si][1]:
                t = spans[si][2]
                lab.append(("I-" if prev == si else "B-") + t)
                prev = si
            else:
                lab.append("O"); prev = None
        out.append(([new[a:b] for a, b in tk], lab))
    # додаємо PERS з NER-UK як додаткову позитивну розмітку (інший домен, менша вага — лише 6000 речень)
    sents, cur = [], []
    for line in open(D + "ner-uk/v2.0/iob/train.iob", encoding="utf8"):
        p = line.split()
        if len(p) == 2:
            t = p[1].replace("PERS", "ОСОБА")
            cur.append((p[0], t if t.endswith("ОСОБА") else "O"))
        elif cur:
            sents.append(cur); cur = []
    random.shuffle(sents)
    for s in sents[:6000]:
        out.append(([w for w, _ in s], [t for _, t in s]))
    random.shuffle(out)
    print("sentences", len(out), "tokens", sum(len(x[0]) for x in out), file=sys.stderr)
    pickle.dump(out, open("train.pkl", "wb"))  # застарілий варіант v1; для CRF v2 див. build_train2.py


if __name__ == "__main__":
    main()
