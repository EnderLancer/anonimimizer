#!/usr/bin/env python3
"""Етап Model: документи -> анотації NER-моделі.

Запуск:
    python solution/model_annotate.py inputs.jsonl model.jsonl
    python solution/model_annotate.py inputs.jsonl model.jsonl --model ukr-models/uk-ner --min-score 0.8

Вхід — JSON Lines з полями id та original. Вихід — анотації у форматі anonymizer/annotations.py,
їх можна одразу оцінити: python data/evaluate.py data/test/court_cases.jsonl model.jsonl
Потрібні пакети з solution/requirements.txt.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from anonymizer import mapping, read_jsonl, write_jsonl  # noqa: E402
from anonymizer.annotations import document_row  # noqa: E402
from anonymizer.model_stage import NerModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("output")
    ap.add_argument("--model", default=os.environ.get("NER_MODEL", "ukr-models/uk-ner"), help="NER-модель з Hugging Face")
    ap.add_argument("--min-score", type=float, default=0.8, help="мінімальна впевненість моделі")
    args = ap.parse_args()

    docs = read_jsonl(args.inputs)
    model = NerModel(args.model, min_score=args.min_score)
    t = time.time()
    found = model.find_batch([d["original"] for d in docs])
    rows = [document_row(d["id"], d["original"], hidden, mapping.assign(hidden), model.metadata)
            for d, hidden in zip(docs, found)]
    write_jsonl(args.output, rows)
    print(f"Model: {len(rows)} документів, {sum(len(r['annotations']) for r in rows)} анотацій за {time.time() - t:.0f} с",
          file=sys.stderr)


if __name__ == "__main__":
    main()
