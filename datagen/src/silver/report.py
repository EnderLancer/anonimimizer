"""Write data/silver/REPORT.md from the pipeline outputs."""
from __future__ import annotations

import json
from collections import Counter

import pyarrow.parquet as pq

from silver.io import SILVER, read_jsonl, step_path


def run_report() -> None:
    lines = ["# Silver dataset report", ""]
    n_src = sum(1 for _ in read_jsonl(step_path("00_filtered")))
    lines += [f"- documents after filter: **{n_src}**"]
    for name in ("train", "dev", "dev_judges"):
        p = SILVER / f"{name}.parquet"
        if not p.exists():
            continue
        t = pq.read_table(p)
        ann = [a for d in t.column("annotations").to_pylist() for a in d]
        msk = [m for d in t.column("masks").to_pylist() for m in d]
        by_type = Counter(a["type"] for a in ann)
        by_src = Counter(a["source"] for a in ann)
        by_reason = Counter(m["reason"] for m in msk)
        tot = sum(by_type.values()) or 1
        lines += ["", f"## {name}.parquet", "", f"- documents: {t.num_rows}", f"- annotations: {len(ann)}; masks: {len(msk)}",
                  "- by type: " + ", ".join(f"{k} {v} ({v / tot:.1%})" for k, v in by_type.most_common()),
                  "- by source: " + ", ".join(f"{k} {v}" for k, v in by_src.most_common()),
                  "- masks by reason: " + ", ".join(f"{k} {v}" for k, v in by_reason.most_common())]
    m = SILVER / "audit" / "metrics.json"
    if m.exists():
        a = json.loads(m.read_text(encoding="utf-8"))
        lines += ["", "## NER audit", "", f"- docs: {a['docs']}, PERS total: {a['pers_total']}",
                  f"- holes_rate: {a['holes_rate']:.4f}, masked_rate: {a['masked_rate']:.4f}, labeled_rate: {a['labeled_rate']:.4f}",
                  "- unconfirmed by source: " + ", ".join(f"{k} {v:.3f}" for k, v in a["unconfirmed_rate_by_source"].items())]
    lines += ["", "## Known limitations", "",
              "- grammatical case of synthetic names is guessed from the left context (heuristic); mismatches remain in a minority of mentions",
              "- addresses are synthetic (embedded street/city lists), not from a real address registry",
              "- ІНФОРМАЦІЯ placeholders with unrecognised context are filled with an institution / unit name (60 %) or a date (40 %) and labelled",
              "- street and settlement names are mined from ЄДРСР court addresses; address formats are 14 templates",
              "- open IBAN/phone/plate values without a clear personal context are masked, not labelled",
              "- test set (data/test) was never used to build or tune this data"]
    (SILVER / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
