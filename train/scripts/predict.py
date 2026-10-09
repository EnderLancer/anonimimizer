#!/usr/bin/env python3
"""Pseudonymize a JSONL of documents with a trained model and write a submission.

    uv run scripts/predict.py --model runs/wechsel_2k/model --input ../data/silver/dev.jsonl --output runs/wechsel_2k/dev_sub.jsonl
    uv run python ../data/evaluate.py ../data/silver/dev.jsonl runs/wechsel_2k/dev_sub.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="JSONL with id + original (test schema) or id + text")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: best available)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification

    from pseudo.predict import best_device, predict_doc, spans_to_submission, write_submission
    from pseudo.tokenization import load_tokenizer

    if a.threads:
        torch.set_num_threads(a.threads)
    device = a.device or best_device()
    tok = load_tokenizer(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(device).eval()
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    t0 = time.time()
    out = []
    for i, r in enumerate(rows, 1):
        text = r.get("original", r.get("text"))
        out.append(spans_to_submission(r["id"], text, predict_doc(model, tok, text, device, a.batch)))
        if i % 50 == 0:
            print(f"{i}/{len(rows)} docs, {time.time() - t0:.0f}s", file=sys.stderr)
    write_submission(a.output, out)
    print(f"{len(rows)} docs in {time.time() - t0:.0f}s on {device} -> {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
