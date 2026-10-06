#!/usr/bin/env python3
"""Псевдонімізатор: python solution/predict.py inputs.jsonl submission.jsonl

Вхід — JSON Lines з полями id та original. Вихід — JSON Lines {"id", "annotations": [{start, end, pseudonym}]}
(позиції символів у original, end не включно), формат data/evaluate.py.

Режими (--mode):
  full  (типово) — нейромережа spaCy для імен + regex-правила + CRF для НОМЕР/ІНФОРМАЦІЯ/АДРЕСА.
                   Модель шукається локально, за потреби завантажується з Hugging Face (~1,3 ГБ, один раз).
  regex          — лише правила та словник імен: без нейромережі, без CRF, швидко, без завантажень.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("submission")
    ap.add_argument("--mode", choices=["full", "regex"], default="full")
    ap.add_argument("--model-dir", help="тека моделі spaCy (інакше пошук/завантаження автоматично)")
    ap.add_argument("--chunk-chars", type=int, default=0,
                    help="ділити довгі тексти на шматки такого розміру для нейромережі (економія пам'яті; 0 = вимкнено)")
    ap.add_argument("--stats", action="store_true", help="надрукувати повний час і пік пам'яті (потрібен psutil)")
    args = ap.parse_args()

    peak = [0]
    stop = [False]
    if args.stats:
        try:
            import psutil
            proc = psutil.Process()

            def watch():
                while not stop[0]:
                    peak[0] = max(peak[0], proc.memory_info().rss)
                    time.sleep(0.5)
            threading.Thread(target=watch, daemon=True).start()
        except ImportError:
            print("psutil не встановлено: пам'ять не вимірюється", file=sys.stderr)

    t0 = time.time()
    if args.mode == "regex":
        from pseudonymizer.regex_only import annotate
        annotate_fn = annotate
    else:
        from pseudonymizer import Pseudonymizer
        annotate_fn = Pseudonymizer(model_dir=args.model_dir, chunk_chars=args.chunk_chars)

    n = 0
    with open(args.submission, "w", encoding="utf-8") as out:
        for row in read_rows(args.inputs):
            out.write(json.dumps({"id": row["id"], "annotations": annotate_fn(row["original"])}, ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"{n} рішень, {time.time() - t0:.0f} с", file=sys.stderr, flush=True)
    stop[0] = True
    el = time.time() - t0
    msg = {"docs": n, "total_seconds": round(el, 1), "seconds_per_doc": round(el / max(1, n), 3)}
    if peak[0]:
        msg["peak_rss_gb"] = round(peak[0] / 1e9, 2)
    print(json.dumps(msg), file=sys.stderr)


if __name__ == "__main__":
    main()
