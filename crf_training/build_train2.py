"""v2 тренувальних даних для окремої CRF-моделі під ІНФОРМАЦІЯ / АДРЕСА / НОМЕР.

Відмінності від build_train.py:
- багатші генератори (формати дат, адреси в стилі Укрпошти, нікнейми, соцмережі);
- ІНФОРМАЦІЯ у контексті «ТОВ/ПрАТ/АТ «...»» замінюється вигаданою назвою компанії з міткою O
  (у ЄДРСР такі назви здебільшого НЕ маскують, тож модель не має їх ловити);
- речення з ІНФОРМАЦІЯ невідомого типу відкидаємо замість вгадування мітки.
Тестова вибірка не використовується.
"""
import pickle
import random
import re
import sys

import pandas as pd

import build_train as B
from common import tokenize

random.seed(11)
rd, UP = B.rd, B.UP
PH = B.PH
MONTHS = B.MONTHS

# Назви місць з NER-UK (LOC) як джерело міст/сіл/вулиць різноманітнішої форми
def load_locs():
    locs, cur = set(), []
    for line in open(B.D + "ner-uk/v2.0/iob/train.iob", encoding="utf8"):
        p = line.split()
        if len(p) == 2 and p[1] in ("B-LOC", "I-LOC"):
            if p[1] == "B-LOC" and cur:
                locs.add(" ".join(cur)); cur = []
            cur.append(p[0])
        elif cur:
            locs.add(" ".join(cur)); cur = []
    return [l for l in locs if l[0].isupper() and len(l.split()) <= 2 and l.replace(" ", "").replace("-", "").isalpha() and 3 <= len(l) <= 20]

LOCS = load_locs()
print("locs", len(LOCS), file=sys.stderr)
REGIONS = ["Київська", "Львівська", "Одеська", "Харківська", "Дніпропетровська", "Полтавська", "Вінницька", "Житомирська", "Сумська",
           "Черкаська", "Чернігівська", "Рівненська", "Волинська", "Запорізька", "Херсонська", "Закарпатська", "Тернопільська"]
STREET_KIND = ["вул.", "вулиця", "просп.", "пров.", "бульвар", "пл.", "провулок", "проспект"]
COMP_NAMES = ["Шевченка", "Лесі Українки", "Франка", "Грушевського", "Незалежності", "Соборна", "Центральна", "Миру", "Садова",
              "Польова", "Перемоги", "Гагаріна", "Молодіжна", "Київська", "Миколаївська", "Українська", "Січових Стрільців"]
CORP = ["А", "Б", "В", "а", "б"]


def gen_street():
    nm = random.choice(COMP_NAMES) if random.random() < 0.6 else random.choice(LOCS).split()[0]
    k = random.choice(STREET_KIND)
    return f"{k} {nm}" if random.random() < 0.9 else f"{nm} {k}"


def gen_address():
    n = random.randint(1, 180)
    house = random.choice([f"{n}", f"{n}", f"{n}", f"{n}/{random.randint(1,9)}", f"{n}{random.choice(CORP)}", f"буд. {n}", f"будинок {n}"])
    apt = random.choice(["", "", f", кв. {random.randint(1,300)}", f", квартира {random.randint(1,300)}", f", кв.{random.randint(1,300)}"])
    city = random.choice(LOCS)
    kind = random.choice(["м.", "м.", "с.", "смт", "м.", "місто", "село"])
    cityfull = f"{kind} {city}" if kind not in ("м.",) else f"м. {city}"
    reg = f"{random.choice(REGIONS)} {random.choice(['область', 'обл.'])}"
    dist = f"{random.choice(LOCS).split()[0]}ський {random.choice(['район', 'р-н'])}"
    parts = []
    r = random.random()
    if r < 0.45:
        parts.append(rd(5))
    if r < 0.55 or random.random() < 0.15:
        parts.append(reg)
    if random.random() < 0.2:
        parts.append(dist)
    if random.random() < 0.85:
        parts.append(cityfull)
    parts.append(gen_street() + (", " if random.random() < 0.9 else " ") + house + apt)
    s = ", ".join(parts)
    return s.replace(",  ", ", ")


def gen_date(kind=None):
    d, m, y = random.randint(1, 28), random.randint(1, 12), random.randint(1938, 2008)
    r = random.random()
    forms = [
        f"{d:02d}.{m:02d}.{y}", f"{d}.{m}.{y}", f"{d:02d}.{m:02d}.{y} р.н.", f"{d:02d}.{m:02d}.{y} року народження",
        f"{d} {MONTHS[m-1]} {y} року", f"{d:02d} {MONTHS[m-1]} {y} р.", f"{d:02d}.{m:02d}.{y} р.", f"{y}-{m:02d}-{d:02d}",
        f"{d:02d}/{m:02d}/{y}", f"{y} року народження", f"{y} р.н.", f"{d:02d}.{m:02d}.{str(y)[2:]}",
    ]
    return random.choice(forms)


def gen_nick():
    return random.choice([f"@user{rd(3)}", f"@{random.choice(['ivan','olena','petro','kate'])}_{rd(2)}", f"nick{rd(4)}",
                          f"https://www.instagram.com/{random.choice(['ivan','olena','kate'])}{rd(2)}", f"https://t.me/user{rd(5)}",
                          f"facebook.com/profile.php?id={rd(12)}"])


def gen_org():
    return random.choice(["Альфа", "Прогрес", "Будінвест", "Агросвіт", "Укртранс", "Сервіс-Плюс", "Нова Лінія", "Мегаполіс", "Еталон"])


def gen_info(ctx_b, ctx_a):
    """повертає (значення, мітка) або None, якщо тип невідомий (речення відкидаємо)"""
    b = ctx_b.lower(); a = ctx_a.lower()
    if re.search(r"(тов|прат|пат|ат|кб|фоп|підприємств\w*|компані\w*|товариств\w*)\s*[«\"“]?\s*$", b):
        return gen_org(), "O"
    if re.search(r"(e-?mail|ел\.? ?пошт|електронн\w+ пошт)", b[-40:]):
        return random.choice([f"user{rd(3)}@gmail.com", f"{random.choice(['ivan','olena','petro'])}.{random.choice(['k','p','s'])}{rd(2)}@ukr.net", f"mail{rd(4)}@meta.ua"]), "ІНФОРМАЦІЯ"
    if re.search(r"(посиланн\w*|сайт\w*|сторінк\w*|facebook|instagram|telegram|акаунт\w*|аккаунт\w*|профіл\w*|нік\w*|назвою)", b[-45:]):
        return gen_nick(), "ІНФОРМАЦІЯ"
    if re.search(r"(помер\w*|смерт\w*|померл\w+|загинув\w*)\s*$", b[-25:]) or re.search(r"(народж|нар\.|уродж|р\.н)", b[-35:] + a[:30]):
        return gen_date(), "ІНФОРМАЦІЯ"
    if re.search(r"ОСОБА_\d+\s*[,(]\s*$", ctx_b) or re.search(r"особа_\d+\s*[,(]\s*$", b):
        return gen_date(), "ІНФОРМАЦІЯ"
    return None


def render(text):
    vals, out, spans, pos, cur = {}, [], [], 0, 0
    for m in PH.finditer(text):
        out.append(text[pos:m.start()])
        cur += m.start() - pos
        typ, k = m.group(1), m.group(2)
        key = (typ, k)
        if key not in vals:
            cb, ca = text[max(0, m.start() - 60):m.start()], text[m.end():m.end() + 40]
            if typ == "ОСОБА":
                vals[key] = (B.gen_person(), "ОСОБА")
            elif typ == "АДРЕСА":
                vals[key] = (gen_address(), "АДРЕСА")
            elif typ == "НОМЕР":
                vals[key] = (B.gen_number(cb), "НОМЕР")
            else:
                r = gen_info(cb, ca)
                if r is None:
                    return None
                vals[key] = r
        v, lab = vals[key]
        out.append(v)
        if lab != "O":
            spans.append((cur, cur + len(v), lab))
        cur += len(v)
        pos = m.end()
    out.append(text[pos:])
    new = "".join(out)
    for m in B.PERS_RULE.finditer(new):
        s, e = m.start(), m.end()
        if not any(s < b and a < e for a, b, _ in spans):
            spans.append((s, e, "ОСОБА"))
    return new, sorted(spans)


def label(new, spans):
    tk = tokenize(new)
    si, prev, lab = 0, None, []
    for (a, b) in tk:
        while si < len(spans) and spans[si][1] <= a:
            si += 1
        if si < len(spans) and spans[si][0] <= a < spans[si][1]:
            lab.append(("I-" if prev == si else "B-") + spans[si][2]); prev = si
        else:
            lab.append("O"); prev = None
    return [new[a:b] for a, b in tk], lab


def main(n_docs=3500, max_sents=90000):
    df = pd.read_parquet(B.D + "court-decisions-uk/2024-court-decisions.parquet")
    df = df[df.text.str.len() < 60000].sample(n_docs, random_state=21)
    data = []
    for text in df.text:
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
    out, dropped = [], 0
    for d in sel[:max_sents * 2]:
        r = render(d)
        if r is None:
            dropped += 1; continue
        out.append(label(*r))
        if len(out) >= max_sents:
            break
    sents, cur = [], []
    for line in open(B.D + "ner-uk/v2.0/iob/train.iob", encoding="utf8"):
        p = line.split()
        if len(p) == 2:
            t = p[1].replace("PERS", "ОСОБА")
            cur.append((p[0], t if t.endswith("ОСОБА") else "O"))
        elif cur:
            sents.append(cur); cur = []
    random.shuffle(sents)
    for s in sents[:4000]:
        out.append(([w for w, _ in s], [t for _, t in s]))
    random.shuffle(out)
    print("sentences", len(out), "tokens", sum(len(x[0]) for x in out), "dropped(unknown info)", dropped, file=sys.stderr)
    pickle.dump(out, open("train2.pkl", "wb"))


if __name__ == "__main__":
    main()
