#!/usr/bin/env python3
"""Кеш виходу NER-моделі за замовчуванням для OOD-набору лежить окремо від ner_cache.json (той покриває лише тест).

    python data/ood/merge_ner_cache.py            # дописати data/ood/spacy_ner.json у ner_cache.json (перед run.py)
    python data/ood/merge_ner_cache.py --extract  # навпаки: винести записи OOD-документів з ner_cache.json у data/ood/spacy_ner.json

run.py читає кеш моделі за замовчуванням лише з ner_cache.json, тож перед запуском на OOD-наборі записи
дописуються туди (у CI — у робочій копії, файл у репозиторії не змінюється).
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAIN = os.path.join(ROOT, "ner_cache.json")
OOD = os.path.join(HERE, "spacy_ner.json")
GOLD = os.path.join(HERE, "ood_cases.jsonl")


def ood_ids() -> list[str]:
    return [str(json.loads(l)["id"]) for l in open(GOLD, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()
    main_cache = json.load(open(MAIN, encoding="utf-8"))
    ids = ood_ids()
    if a.extract:
        missing = [i for i in ids if i not in main_cache]
        if missing:
            raise SystemExit(f"{len(missing)} OOD-документів немає в ner_cache.json, напр. {missing[:3]}")
        with open(OOD, "w", encoding="utf-8") as f:
            json.dump({i: main_cache[i] for i in ids}, f, ensure_ascii=False)
        print(f"{len(ids)} записів -> {OOD}")
        return
    ood_cache = json.load(open(OOD, encoding="utf-8"))
    missing = [i for i in ids if i not in ood_cache]
    if missing:
        raise SystemExit(f"{len(missing)} OOD-документів немає в {OOD}, напр. {missing[:3]}")
    main_cache.update(ood_cache)
    with open(MAIN, "w", encoding="utf-8") as f:
        json.dump(main_cache, f, ensure_ascii=False)
    print(f"{len(ood_cache)} записів дописано в {MAIN}")


if __name__ == "__main__":
    main()
