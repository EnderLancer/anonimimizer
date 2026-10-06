#!/usr/bin/env python3
"""Запускає Regex+model flow над тестовими даними.

Запуск:
    python solution/predict.py inputs.jsonl submission.jsonl
    python solution/predict.py inputs.jsonl submission.jsonl --no-model         # лише RegEx
    python solution/predict.py inputs.jsonl submission.jsonl --model ukr-models/uk-ner

Вхід — JSON Lines з полями id та original. Вихід — JSON Lines з annotations (див. data/evaluate.py)
і pseudonymized (очищений документ).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from anonymizer import Pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("output")
    ap.add_argument("--no-model", action="store_true", help="пропустити етап Model")
    ap.add_argument("--model", default=os.environ.get("NER_MODEL", "ukr-models/uk-ner"), help="NER-модель з Hugging Face")
    args = ap.parse_args()

    with open(args.inputs, encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    model = None
    if not args.no_model:
        from anonymizer.model_stage import NerModel
        model = NerModel(args.model)

    t = time.time()
    results = Pipeline(model).run_batch([r["original"] for r in rows])
    with open(args.output, "w", encoding="utf-8") as f:
        for r, res in zip(rows, results):
            f.write(json.dumps({"id": r["id"], "annotations": res.annotations, "pseudonymized": res.cleaned},
                               ensure_ascii=False) + "\n")
    print(f"{len(rows)} документів опрацьовано за {time.time() - t:.0f} с", file=sys.stderr)


if __name__ == "__main__":
    main()
