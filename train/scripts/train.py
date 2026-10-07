#!/usr/bin/env python3
"""Fine-tune a model on the silver data and report dev F1 after every epoch.

    uv run scripts/train.py --model benjamin/roberta-base-wechsel-ukrainian --n-docs 2000 --epochs 1 --out runs/wechsel_2k
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pseudo.train import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="run directory (model/, dev submissions, config.json)")
    ap.add_argument("--n-docs", type=int, default=None, help="train documents (default: all)")
    ap.add_argument("--dev-docs", type=int, default=None, help="dev documents for the per-epoch score (default: all 500)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oversample-number", type=float, default=1.0, help="add this share of НОМЕР docs again (1.0 = x2)")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--push-to", default=None, help="HF Hub repo id to push the model to")
    ap.add_argument("--max-windows", type=int, default=None, help="debug: cap training windows")
    a = ap.parse_args()
    run(a.model, Path(a.out), a.n_docs, a.dev_docs, a.epochs, a.lr, a.batch, a.grad_accum, a.seed, a.oversample_number,
        not a.no_fp16, a.push_to, a.max_windows)


if __name__ == "__main__":
    main()
