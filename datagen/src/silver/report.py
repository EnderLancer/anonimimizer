"""Write data/silver/REPORT.md from the pipeline outputs."""
from __future__ import annotations

import json
from collections import Counter

import pyarrow.parquet as pq

from silver.io import SILVER, VARIANT, read_jsonl, step_path


def run_report() -> None:
    lines = ["# Silver dataset report", ""]
    n_src = sum(1 for _ in read_jsonl(step_path("00_filtered")))
    lines += [f"- variant: **{VARIANT}**", f"- documents after filter: **{n_src}**"]
    for name in ("train", "dev", "dev_judges"):
        p = SILVER / f"{name}.parquet"
        if not p.exists():
            continue
        t = pq.read_table(p)
        ann = [a for d in t.column("annotations").to_pylist() for a in d]
        msk = [m for d in t.column("masks").to_pylist() for m in d]
        by_type = Counter(a["type"] for a in ann)
        by_src = Counter(a["source"] for a in ann)
        by_kind = Counter(f"{c}: {k}" for c, k in zip(t.column("corpus").to_pylist(), t.column("kind").to_pylist()))
        by_reason = Counter(m["reason"] for m in msk)
        tot = sum(by_type.values()) or 1
        lines += ["", f"## {name}.parquet", "", f"- documents: {t.num_rows}", f"- annotations: {len(ann)}; masks: {len(msk)}",
                  "- by corpus and kind of decision: " + ", ".join(f"{k} {v}" for k, v in by_kind.most_common()),
                  "- by type: " + ", ".join(f"{k} {v} ({v / tot:.1%})" for k, v in by_type.most_common()),
                  "- by source: " + ", ".join(f"{k} {v}" for k, v in by_src.most_common()),
                  "- masks by reason: " + ", ".join(f"{k} {v}" for k, v in by_reason.most_common())]
    fs = SILVER / "fill_stats.json"
    if fs.exists():
        s = json.loads(fs.read_text(encoding="utf-8"))

        def shares(prefix: str) -> str:
            sel = {k[len(prefix):]: v for k, v in s.items() if k.startswith(prefix)}
            total = sum(sel.values()) or 1
            return ", ".join(f"{k} {v / total:.1%}" for k, v in sorted(sel.items(), key=lambda x: -x[1]))

        n_addr = sum(v for k, v in s.items() if k.startswith("addr_kind_")) or 1
        lines += ["", "## Synthetic values (fill step, before excerpts and the split)", "",
                  f"- addresses: {n_addr}; settlement: {shares('addr_kind_')}",
                  f"- address begins with: {shares('addr_order_')}",
                  "- address parts: " + ", ".join(f"{name} {s.get(key, 0) / n_addr:.1%}" for name, key in (
                      ("settlement named in the decision", "addr_from_doc"), ("index", "addr_with_zip"),
                      ("apartment", "addr_apt"), ("no street", "addr_no_street"))),
                  f"- НОМЕР placeholders: {shares('num_')}",
                  f"- ІНФОРМАЦІЯ placeholders: {shares('info_')}",
                  f"- date format of a document: {shares('date_doc_style_')}"]
    m = SILVER / "audit" / "metrics.json"
    if m.exists():
        a = json.loads(m.read_text(encoding="utf-8"))
        lines += ["", "## NER audit", "", f"- docs: {a['docs']}, PERS total: {a['pers_total']}",
                  f"- holes_rate: {a['holes_rate']:.4f}, masked_rate: {a['masked_rate']:.4f}, labeled_rate: {a['labeled_rate']:.4f}",
                  "- unconfirmed by source: " + ", ".join(f"{k} {v:.3f}" for k, v in a["unconfirmed_rate_by_source"].items())]
    lines += ["", "## Known limitations", "",
              "- grammatical case of synthetic names is guessed from the left context (heuristic); mismatches remain in a minority of mentions",
              "- address values come from the Ukrposhta directory (settlement, district, street, index and house number of one row); "
              "apartments and the order / wording of the parts are generated, city districts are not",
              "- ІНФОРМАЦІЯ placeholders with unrecognised context are filled with an institution / unit name (60 %) or a date (40 %) and labelled",
              "- documents are excerpts: the heading is kept in about 15 % and the signature block in about 13 % of them",
              "- open IBAN/phone/plate values without a clear personal context are masked, not labelled",
              "- this module does not read data/test; the 250 decisions the test set was cut from are used only as an exclusion list "
              "(their ids, their cases and near-copies are dropped)",
              "- three choices rest on knowledge about the test set: the shares of date formats and the cutting of excerpts follow "
              "an outside report that measured the test set, and labelling of open enforcement-proceeding numbers («ВП №») was "
              "checked against the test labels (17 of 17 such numbers are labelled there)"]
    (SILVER / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
