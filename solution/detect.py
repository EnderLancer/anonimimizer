#!/usr/bin/env python3
"""Етап пошуку: документи -> кандидати з джерелом, ще без псевдонімів.

Запуск:
    python solution/detect.py inputs.jsonl candidates.jsonl [--mode full|regex] [--model-dir DIR] [--chunk-chars N]

Вхід — JSON Lines з полями id та original. Вихід — один рядок на документ:
    {"id", "original", "metadata": {"stage": "detect", "mode"},
     "annotations": [{"start", "end", "type", "text", "metadata": {"source": "ner" | "regex_persons" | "rules" | "crf"}}]}
Далі: solution/combine.py (злиття й псевдоніми) -> scripts/to_submission.py -> data/evaluate.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("output")
    ap.add_argument("--mode", choices=["full", "regex"], default="full")
    ap.add_argument("--model-dir", help="тека моделі spaCy (інакше пошук/завантаження автоматично)")
    ap.add_argument("--chunk-chars", type=int, default=0, help="ділити довгі тексти на шматки для нейромережі")
    args = ap.parse_args()

    if args.mode == "regex":
        from pseudonymizer.regex_only import candidates
    else:
        from pseudonymizer import Pseudonymizer
        candidates = Pseudonymizer(model_dir=args.model_dir, chunk_chars=args.chunk_chars).candidates

    t0, n = time.time(), 0
    with open(args.inputs, encoding="utf-8-sig") as fin, open(args.output, "w", encoding="utf-8") as out:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            out.write(json.dumps({"id": row["id"], "original": row["original"],
                                  "metadata": {"stage": "detect", "mode": args.mode},
                                  "annotations": candidates(row["original"])}, ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"{n} рішень, {time.time() - t0:.0f} с", file=sys.stderr, flush=True)
    print(f"detect ({args.mode}): {n} рішень за {time.time() - t0:.0f} с", file=sys.stderr)


if __name__ == "__main__":
    main()
