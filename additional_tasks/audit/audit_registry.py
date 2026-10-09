"""Local candidate discovery and manual review of published court decisions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import predict_trf as P
from regex_persons import persons
from additional_tasks.audit.service_filter import mark_service_candidates

PLACEHOLDER = re.compile(r"(?:ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_\d+")


def select_rows(path, limit, seed, text_column=None, id_column="id"):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    text_column = text_column or next((n for n in ("text", "pseudonymized") if n in names), None)
    if text_column not in names or id_column not in names:
        raise ValueError(f"Specify --text-column / --id-column; columns: {names}")
    rng = random.Random(seed)
    rows, seen = [], 0
    for batch in parquet.iter_batches(batch_size=256, columns=[id_column, text_column]):
        for item in batch.to_pylist():
            text = item[text_column]
            if not isinstance(text, str) or not text.strip():
                continue
            row = {"id": str(item[id_column]), "text": text, "source_row": seen}
            seen += 1
            if len(rows) < limit:
                rows.append(row)
            else:
                index = rng.randrange(seen)
                if index < limit:
                    rows[index] = row
    return sorted(rows, key=lambda r: r["source_row"]), seen, text_column


def candidates(text, tagger=None, ner=()):
    found = {}
    placeholders = [(m.start(), m.end()) for m in PLACEHOLDER.finditer(text)]

    def add(start, end, kind, source):
        if not (0 <= start < end <= len(text)):
            return
        if any(start < b and a < end for a, b in placeholders):
            return
        found.setdefault((start, end, kind), set()).add(source)

    people = persons(text)
    for s, e, _ in people:
        add(s, e, "ОСОБА", "regex_person")
    person_spans = [(s, e) for s, e, _ in people] + [(x.start, x.end) for x in ner if x.label == "ОСОБА"]
    for s, e, kind in P.rule_spans(text, person_spans):
        add(s, e, kind, "rules")
    # Recover institutional addresses/URLs suppressed by production rules.
    # Preserve all raw detections, including those later marked as service data.
    for name, kind in (("ADDR_STREET", "АДРЕСА"), ("ADDR_VILLAGE", "АДРЕСА"), ("URL", "ІНФОРМАЦІЯ")):
        for match in getattr(P, name).finditer(text):
            add(match.start(), match.end(), kind, "raw_" + name.lower())
    if tagger:
        from predict import predict_doc
        for s, e, kind in predict_doc(tagger, text):
            if kind in ("ОСОБА", "НОМЕР", "ІНФОРМАЦІЯ", "АДРЕСА"):
                add(s, e, kind, "crf")
    for span in ner:
        add(span.start, span.end, span.label, "ner")
    result = [{"start": s, "end": e, "type": kind, "value": text[s:e],
             "before": text[max(0, s - 180):s], "after": text[e:e + 180],
             "sources": sorted(sources)}
            for (s, e, kind), sources in sorted(found.items())]
    return mark_service_candidates(text, result)


def write_report(path, payload):
    template = (HERE / "audit_review.html").read_text(encoding="utf8")
    # Keep arbitrary decision text inert inside the embedded JSON script.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    path.write_text(template.replace("__AUDIT_DATA__", data), encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", nargs="?", type=Path, default=ROOT / "data/train/2024-court-decisions.parquet")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--mode", choices=("regex", "regex_crf", "full"), default="full")
    parser.add_argument("--ner", help="NER specification for full mode (as in run.py)")
    parser.add_argument("--text-column")
    parser.add_argument("--id-column", default="id")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.ner and args.mode != "full":
        parser.error("--ner requires --mode full")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, total, column = select_rows(args.src, args.limit, args.seed, args.text_column, args.id_column)
    if not rows:
        parser.error("No nonempty texts in source")
    print(f"Selected {len(rows)} / {total} texts; column={column}; mode={args.mode}", flush=True)
    tagger = None
    if args.mode != "regex":
        from run import load_tagger
        tagger = load_tagger()
    ner = {}
    if args.mode == "full":
        import ner_adapters as N
        import ner_postprocess as NP
        from run import DEFAULT_NER
        spec = args.ner or DEFAULT_NER
        # Separate cache, keyed by model spec, document ID and exact published text.
        keyed = [{"id": hashlib.sha256((spec + "\0" + r["id"] + "\0" + r["text"]).encode()).hexdigest(),
                  "original": r["text"]} for r in rows]
        raw = N.run_ner(keyed, spec, str(args.out / "ner_cache.json"))
        ner = {r["source_row"]: NP.process(N.map_labels(raw[k["id"]]), r["text"]) for r, k in zip(rows, keyed)}
    counts, excluded_counts, exclusion_reasons = Counter(), Counter(), Counter()
    with (args.out / "candidates.jsonl").open("w", encoding="utf8") as output:
        for i, row in enumerate(rows, 1):
            row["candidates"] = candidates(row["text"], tagger, ner.get(row["source_row"], []))
            for candidate in row["candidates"]:
                candidate["key"] = hashlib.sha256((row["id"] + "\0" + row["text"] + "\0" + str(candidate["start"]) + ":" + str(candidate["end"]) + ":" + candidate["type"]).encode()).hexdigest()
                if candidate["excluded"]:
                    excluded_counts[candidate["type"]] += 1
                    exclusion_reasons[candidate["exclusion_reason"]] += 1
                else:
                    counts[candidate["type"]] += 1
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(rows):
                print(f"Processed {i}/{len(rows)}; candidates={sum(counts.values())}; excluded={sum(excluded_counts.values())}", flush=True)
    metadata = {"source": str(args.src.resolve()), "mode": args.mode, "ner": args.ner,
                "seed": args.seed, "selected": len(rows), "total": total, "text_column": column,
                "counts": dict(counts), "excluded_counts": dict(excluded_counts),
                "exclusion_reasons": dict(exclusion_reasons), "service_filter_version": 2,
                "documents_with_candidates": sum(any(not c["excluded"] for c in r["candidates"]) for r in rows),
                "documents_with_raw_candidates": sum(bool(r["candidates"]) for r in rows)}
    (args.out / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf8")
    write_report(args.out / "review.html", {"metadata": metadata, "documents": rows})
    print(f"Review: {(args.out / 'review.html').resolve()}")


if __name__ == "__main__":
    main()
