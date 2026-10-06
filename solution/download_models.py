#!/usr/bin/env python3
"""Завантажує нейромережу uk_ner_web_trf_13class (~1,3 ГБ) у solution/models/spacy/, якщо її ще немає.

    python solution/download_models.py

CRF-модель (solution/models/crf2.model, ~7 МБ) лежить у репозиторії, завантажувати її не треба.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pseudonymizer import ner  # noqa: E402

if __name__ == "__main__":
    found = ner.find_model_dir()
    if found:
        print(f"Модель уже є: {found}")
    else:
        print(f"Модель розпакована в: {ner.download_model()}")
