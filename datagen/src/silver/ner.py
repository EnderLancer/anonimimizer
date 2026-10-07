"""Step 4: run uk_ner_web_trf_13class (PERS only) over the documents; cache per document id; add
PERS spans in a name shape as ОСОБА(source=ner), other PERS as masks."""
from __future__ import annotations

import glob
import json
import logging
import time
from pathlib import Path

from silver import patterns as P
from silver.io import SILVER, RESOURCES, read_jsonl, step_path, write_jsonl
from silver.merge import any_overlap, merge
from silver.persons import _ok

log = logging.getLogger("silver")
CACHE = SILVER / "ner_cache"
MODEL_GLOB = str(RESOURCES / "models" / "uk_ner_web_trf_13class" / "uk_ner_web_trf_13class-*")


def load_model():
    import spacy
    import torch

    torch.set_num_threads(8)
    path = glob.glob(MODEL_GLOB)[0]
    nlp = spacy.load(path, exclude=["parser", "lemmatizer", "tagger", "attribute_ruler"])
    return nlp


def _cache_path(doc_id: str, text: str | None = None) -> Path:
    """Cache key includes a hash of the text: a regenerated document (new synthetic fills, new offsets) is recomputed."""
    import hashlib

    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10] if text is not None else "legacy"
    return CACHE / f"{doc_id}-{h}.json"


def run_ner_cache(src: str = "02_labeled", batch: int = 8, limit: int | None = None) -> None:
    """Fill the PERS cache for every document of `src` that is not cached yet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    docs = list(read_jsonl(step_path(src)))
    if limit:
        docs = docs[:limit]
    todo = [d for d in docs if not _cache_path(d["id"], d["text"]).exists()]
    log.info("NER: %d documents to process", len(todo))
    if not todo:
        return
    nlp = load_model()
    t0 = time.time()
    done = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i: i + batch]
        for d, doc in zip(chunk, nlp.pipe([d["text"] for d in chunk], batch_size=batch)):
            ents = [{"start": e.start_char, "end": e.end_char, "label": e.label_, "text": e.text}
                    for e in doc.ents if e.label_ == "PERS"]
            _cache_path(d["id"], d["text"]).write_text(json.dumps(ents, ensure_ascii=False), encoding="utf-8")
        done += len(chunk)
        if done % 200 < batch:
            rate = done / (time.time() - t0)
            log.info("NER %d/%d  %.1f docs/s  ETA %.0f min", done, len(todo), rate, (len(todo) - done) / rate / 60)


def apply_ner(doc: dict) -> dict:
    cp = _cache_path(doc["id"], doc["text"])
    if not cp.exists():
        return doc
    text = doc["text"]
    labels, masks = list(doc["labels"]), list(doc["masks"])
    for e in json.loads(cp.read_text(encoding="utf-8")):
        span = {"start": e["start"], "end": e["end"], "type": "ОСОБА", "pseudonym": None, "source": "ner"}
        if any_overlap(span, labels):
            continue
        shaped = P.NAME_ANY.fullmatch(e["text"].strip()) is not None
        if shaped and _ok(text, e["start"], e["end"]):
            labels.append(span)
        else:
            masks.append({"start": e["start"], "end": e["end"], "reason": "ner_other"})
    labels, masks = merge(text, labels, masks)
    return {**doc, "labels": labels, "masks": masks}


def run_ner(batch: int = 8, src: str = "02_labeled", dst: str = "03_ner", limit: int | None = None) -> None:
    run_ner_cache(src, batch, limit)
    n = write_jsonl(step_path(dst), (apply_ner(d) for d in read_jsonl(step_path(src))))
    log.info("NER layer applied to %d docs", n)
