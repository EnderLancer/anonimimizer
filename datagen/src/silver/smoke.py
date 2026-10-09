"""Smoke test of the fine-tune path: parquet -> BIO tags -> xlm-roberta-base (300 docs, 1 epoch) -> submission -> evaluate.py.

Not a real training run; it proves the data format works end to end.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from silver.io import DATA, SILVER

log = logging.getLogger("silver")
TYPES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
L2I = {l: i for i, l in enumerate(LABELS)}
MODEL = "FacebookAI/xlm-roberta-base"
WINDOW, STRIDE = 512, 128


def encode(tokenizer, text: str, annotations: list[dict], masks: list[dict]):
    """Char spans -> token labels for every 512-token window of the document (offset mapping based)."""
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=WINDOW, stride=STRIDE,
                    return_overflowing_tokens=True, padding="max_length")
    char_tag = [0] * (len(text) + 1)
    for a in annotations:
        for i in range(a["start"], a["end"]):
            char_tag[i] = L2I[("B-" if i == a["start"] else "I-") + a["type"]]
    char_mask = [False] * (len(text) + 1)
    for m in masks:
        for i in range(m["start"], m["end"]):
            char_mask[i] = True
    out = []
    for ids, offs, att in zip(enc["input_ids"], enc["offset_mapping"], enc["attention_mask"]):
        labels = []
        prev_end = -1
        for (s, e), a in zip(offs, att):
            if a == 0 or e == 0 and s == 0:
                labels.append(-100)
                continue
            if any(char_mask[s:e]):
                labels.append(-100)
            else:
                tag = char_tag[s]
                # a token that continues an entity started in a previous token gets I- even if the char is B-
                if tag and LABELS[tag].startswith("B-") and s > 0 and char_tag[s - 1] and LABELS[char_tag[s - 1]][2:] == LABELS[tag][2:] and prev_end == s:
                    tag = L2I["I-" + LABELS[tag][2:]]
                labels.append(tag)
            prev_end = e
        out.append({"input_ids": ids, "attention_mask": att, "labels": labels, "offsets": offs})
    return out


def decode(text: str, windows: list[dict], preds: list[list[int]]) -> list[dict]:
    """Token predictions -> char spans (first prediction wins for overlapping windows)."""
    char_pred = [None] * len(text)
    for w, p in zip(windows, preds):
        for (s, e), lab in zip(w["offsets"], p):
            if e == 0 and s == 0:
                continue
            for i in range(s, e):
                if char_pred[i] is None:
                    char_pred[i] = LABELS[lab]
    # chars -> spans: a token-level B- or I- continues the current span of the same type when the gap is
    # whitespace only (lenient merge: IoU-based scoring prefers one long span over two fragments)
    spans, cur = [], None
    for i, lab in enumerate(char_pred):
        if lab and lab != "O":
            t = lab[2:]
            if cur and cur["type"] == t and text[cur["end"]:i].strip() == "" and i - cur["end"] <= 1:
                cur["end"] = i + 1
            else:
                if cur:
                    spans.append(cur)
                cur = {"start": i, "end": i + 1, "type": t}
        elif cur and text[i].isspace() and i - cur["end"] <= 0:
            continue  # whitespace right after the span: decide on the next char
        else:
            if cur:
                spans.append(cur)
                cur = None
    if cur:
        spans.append(cur)
    out = []
    for s in spans:
        a, b = s["start"], s["end"]
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            out.append({"start": a, "end": b, "type": s["type"]})
    return out


def run_smoke(n_train: int = 300, n_dev: int = 50) -> None:
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments
    from datasets import Dataset

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    train = pq.read_table(SILVER / "train.parquet").slice(0, n_train).to_pylist()
    dev = pq.read_table(SILVER / "dev.parquet").slice(0, n_dev).to_pylist()
    feats = [w for d in train for w in encode(tok, d["text"], d["annotations"], d["masks"])]
    for f in feats:
        f.pop("offsets")
    ds = Dataset.from_list(feats)
    log.info("smoke: %d windows from %d docs", len(ds), n_train)
    model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)
    out_dir = SILVER / "smoke_model"
    args = TrainingArguments(output_dir=str(out_dir), per_device_train_batch_size=8, num_train_epochs=1, learning_rate=3e-5,
                             logging_steps=10, save_strategy="no", report_to=[], use_mps_device=torch.backends.mps.is_available(),
                             dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForTokenClassification(tok))
    res = trainer.train()
    log.info("smoke: train loss %.4f", res.training_loss)
    model.eval()
    device = next(model.parameters()).device
    sub = []
    for d in dev:
        windows = encode(tok, d["text"], [], [])
        preds = []
        for w in windows:
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([w["input_ids"]], device=device), attention_mask=torch.tensor([w["attention_mask"]], device=device)).logits
            preds.append(logits.argmax(-1)[0].tolist())
        spans = decode(d["text"], windows, preds)
        counters = {t: 0 for t in TYPES}
        anns = []
        for s in spans:
            counters[s["type"]] += 1
            anns.append({"start": s["start"], "end": s["end"], "pseudonym": f"{s['type']}_{counters[s['type']]}"})
        sub.append({"id": d["src_id"], "annotations": anns})
    sub_path = SILVER / "smoke_submission.jsonl"
    with open(sub_path, "w", encoding="utf-8") as f:
        for r in sub:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gold_path = SILVER / "smoke_dev_gold.jsonl"
    ids = {d["src_id"] for d in dev}
    with open(gold_path, "w", encoding="utf-8") as f, open(SILVER / "dev.jsonl", encoding="utf-8") as g:
        for line in g:
            if json.loads(line)["id"] in ids:
                f.write(line)
    print(subprocess.run([sys.executable, str(DATA / "evaluate.py"), str(gold_path), str(sub_path)], capture_output=True, text=True).stdout)
