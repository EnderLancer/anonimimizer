"""Step 7: export fine-tune parquet files and dev JSONL in the test-set schema."""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from silver.io import SILVER, read_jsonl, step_path, write_jsonl
from silver.split import OPEN_NAME_SOURCES

log = logging.getLogger("silver")
_INIT = re.compile(r"^[А-ЯІЇЄҐ]\.?$")


def _person_key(mention: str) -> tuple[str, str]:
    """(surname stem, initials) for grouping open-name mentions of the same person inside a document."""
    toks = [t for t in re.split(r"[\s.]+", mention.replace("’", "'")) if t]
    words = [t for t in toks if not _INIT.match(t)]
    inits = "".join(t[0] for t in toks if t and t != words[0]) if words else ""
    surname = words[0] if words else mention
    stem = surname.lower()
    stem = re.sub(r"(ського|ському|ським|ської|ській|ською|ська|ський|ова|ову|овим|овій|ової|ою|ові|ею|єю|ем|ом|ам|ях|ях|а|у|і|ю|ї|е|я|й|о|и|ь)$", "", stem)
    return stem[:6], inits[:2]


def assign_pseudonyms(doc: dict) -> list[dict]:
    """Placeholders keep их idx; open-name mentions are grouped by surname stem (+ compatible initials)."""
    counters = defaultdict(int)
    taken = defaultdict(set)
    out = []
    for l in doc["labels"]:
        if l["source"] == "placeholder" and l.get("pseudonym"):
            taken[l["type"]].add(int(l["pseudonym"].rsplit("_", 1)[1]))
    groups: dict[tuple, str] = {}
    stem_only: dict[str, str] = {}

    def fresh(t: str) -> str:
        counters[t] += 1
        while counters[t] in taken[t]:
            counters[t] += 1
        taken[t].add(counters[t])
        return f"{t}_{counters[t]}"

    for l in doc["labels"]:
        t = l["type"]
        if l["source"] == "placeholder" and l.get("pseudonym"):
            out.append(l)
            continue
        if t == "ОСОБА" and l["source"] in OPEN_NAME_SOURCES:
            stem, inits = _person_key(doc["text"][l["start"]: l["end"]])
            key = (stem, inits)
            if key in groups:
                ps = groups[key]
            elif inits and stem in stem_only and (stem, "") in groups:
                ps = groups[(stem, "")]
                groups[key] = ps
            else:
                ps = fresh(t)
                groups[key] = ps
                stem_only.setdefault(stem, ps)
            out.append({**l, "pseudonym": ps})
        else:
            out.append({**l, "pseudonym": fresh(t)})
    return out


SCHEMA = pa.schema([
    ("id", pa.string()),
    ("src_id", pa.string()),
    ("text", pa.string()),
    ("text_pseudonymized", pa.string()),
    ("text_source", pa.string()),
    ("annotations", pa.list_(pa.struct([("start", pa.int32()), ("end", pa.int32()), ("type", pa.string()),
                                         ("pseudonym", pa.string()), ("source", pa.string())]))),
    ("masks", pa.list_(pa.struct([("start", pa.int32()), ("end", pa.int32()), ("reason", pa.string())]))),
    ("n_persons", pa.int32()),
])


def render_pseudonymized(text: str, annotations: list[dict]) -> str:
    """`text` with every annotated span replaced by its pseudonym (the same view ЄДРСР publishes, but complete)."""
    out, pos = [], 0
    for a in sorted(annotations, key=lambda x: x["start"]):
        if a["start"] < pos:
            continue
        out += [text[pos: a["start"]], a["pseudonym"]]
        pos = a["end"]
    out.append(text[pos:])
    return "".join(out)


def _rows(docs):
    for d in docs:
        labels = assign_pseudonyms(d)
        annotations = [{k: l[k] for k in ("start", "end", "type", "pseudonym", "source")} for l in labels]
        yield {
            "id": f"{d['id']}-v1",
            "src_id": d["id"],
            "text": d["text"],
            # fully pseudonymized: placeholders AND the names/numbers ЄДРСР left open
            "text_pseudonymized": render_pseudonymized(d["text"], annotations),
            # the ЄДРСР text as published (normalized); judges/lawyers are still open here
            "text_source": d["text_pseudonymized"],
            "annotations": annotations,
            "masks": [{k: m[k] for k in ("start", "end", "reason")} for m in d["masks"]],
            "n_persons": len({l["pseudonym"] for l in labels if l["type"] == "ОСОБА"}),
        }


def _test_schema(rows):
    for r in rows:
        yield {
            "id": r["src_id"],
            "pseudonymized": r["text_pseudonymized"],
            "original": r["text"],
            "annotations": [{"start": a["start"], "end": a["end"], "entity": r["text"][a["start"]: a["end"]],
                             "pseudonym": a["pseudonym"]} for a in r["annotations"]],
        }


def run_export() -> None:
    for name in ("train", "dev", "dev_judges"):
        rows = list(_rows(read_jsonl(step_path(f"split_{name}"))))
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        pq.write_table(table, SILVER / f"{name}.parquet", compression="zstd")
        if name != "train":
            write_jsonl(SILVER / f"{name}.jsonl", _test_schema(rows))
        n_ann = sum(len(r["annotations"]) for r in rows)
        log.info("export %s: %d docs, %d annotations -> %s", name, len(rows), n_ann, SILVER / f"{name}.parquet")
