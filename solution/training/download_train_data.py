#!/usr/bin/env python3
"""Завантажує відкриті тренувальні дані, потрібні для перенавчання (не входять у репозиторій):

    python solution/training/download_train_data.py [--data-dir data/train]

  - NER-UK 2.0 (git clone lang-uk/ner-uk)         -> <data-dir>/ner-uk/
  - Ukr-Synth, train.conllu.gz (~190 МБ)          -> <data-dir>/Ukr-Synth/data/
Корпус рішень 2024-court-decisions.parquet уже лежить у data/train/ (git LFS).
"""
import argparse
import subprocess
import urllib.request
from pathlib import Path

SYNTH = "https://huggingface.co/datasets/ukr-models/Ukr-Synth/resolve/main/data/train.conllu.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train")
    d = Path(ap.parse_args().data_dir)
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "ner-uk").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/lang-uk/ner-uk.git", str(d / "ner-uk")])
    target = d / "Ukr-Synth" / "data" / "train.conllu.gz"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print("Завантаження Ukr-Synth…")
        urllib.request.urlretrieve(SYNTH, target)
    print("Готово:", d)


if __name__ == "__main__":
    main()
