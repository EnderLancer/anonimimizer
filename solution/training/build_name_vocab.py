#!/usr/bin/env python3
"""Будує словники для regex-режиму з відкритих тренувальних корпусів (NER-UK 2.0, Ukr-Synth).

    python solution/training/build_name_vocab.py --data-dir data/train

Результат (solution/pseudonymizer/resources/):
  names_vocab.txt  — слова з великої літери, що >=85% входжень є частиною імені людини (>=3 входжень)
  common_caps.txt  — часті слова з великої літери, які майже ніколи не є іменами (>=20 входжень, <10% людей),
                     наприклад «Суд», «Позивач», «Україна»: вони не мають ставати частиною імені.
Тестова вибірка не використовується.
"""
import argparse
import collections
import gzip
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "pseudonymizer" / "resources"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--max-synth-lines", type=int, default=6_000_000)
    a = ap.parse_args()
    d = Path(a.data_dir)
    per, allcap = collections.Counter(), collections.Counter()

    def feed(sent):
        for w, t in sent:
            if w[:1].isupper() and w.isalpha():
                lw = w.lower()
                allcap[lw] += 1
                if t.endswith("PER") or t.endswith("PERS"):
                    per[lw] += 1

    sent = []
    with gzip.open(d / "Ukr-Synth" / "data" / "train.conllu.gz", "rt", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i > a.max_synth_lines:
                break
            line = line.rstrip("\n")
            if not line:
                if sent:
                    feed(sent); sent = []
                continue
            if line.startswith("#"):
                continue
            p = line.split("\t")
            if len(p) == 10:
                sent.append((p[1], p[9]))
    sent = []
    for line in open(d / "ner-uk" / "v2.0" / "iob" / "train.iob", encoding="utf8"):
        p = line.split()
        if len(p) == 2:
            sent.append((p[0], p[1]))
        elif sent:
            feed(sent); sent = []

    # частота великих слів у судових рішеннях ЄДРСР (імена там замасковані, тож часте слово — не ім'я)
    court = collections.Counter()
    pq = next((x for x in (d / "2024-court-decisions.parquet", d / "court-decisions-uk" / "2024-court-decisions.parquet") if x.exists()), None)
    if pq:
        import re
        import pandas as pd
        tok = re.compile(r"[А-ЯІЇЄҐ][а-яіїєґ'’ʼ-]+")
        for text in pd.read_parquet(pq).text.sample(4000, random_state=1):
            for w in tok.findall(text[:30000]):
                court[w.lower()] += 1
    court_common = {w for w, c in court.items() if c >= 150}
    names = sorted(w for w, c in per.items() if c >= 3 and per[w] / allcap[w] >= 0.85 and w not in court_common)
    common = sorted({w for w, c in allcap.items() if c >= 20 and per[w] / c < 0.10} | court_common)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "names_vocab.txt").write_text("\n".join(names), encoding="utf8")
    (OUT / "common_caps.txt").write_text("\n".join(common), encoding="utf8")
    print(f"names_vocab: {len(names)}, common_caps: {len(common)} -> {OUT}")


if __name__ == "__main__":
    main()
