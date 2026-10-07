"""Run a fine-tuned model over documents and write a submission (test schema)."""
from __future__ import annotations

import json
from collections import Counter

import torch

from pseudo import TYPES
from pseudo.data import decode, encode


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def predict_doc(model, tokenizer, text: str, device: str, batch: int = 16) -> list[dict]:
    window = getattr(model.config, "pseudo_window", 512)
    stride = getattr(model.config, "pseudo_stride", 128)
    windows = encode(tokenizer, text, [], [], window=window, stride=stride, pad=True)
    preds = []
    for i in range(0, len(windows), batch):
        chunk = windows[i: i + batch]
        ids = torch.tensor([w["input_ids"] for w in chunk], device=device)
        att = torch.tensor([w["attention_mask"] for w in chunk], device=device)
        logits = model(input_ids=ids, attention_mask=att).logits
        preds += logits.argmax(-1).tolist()
    return decode(text, windows, preds)


def spans_to_submission(doc_id, spans: list[dict]) -> dict:
    """Pseudonyms numbered per type in order of appearance (linking across mentions is a later step)."""
    counters = Counter()
    anns = []
    for s in sorted(spans, key=lambda x: x["start"]):
        counters[s["type"]] += 1
        anns.append({"start": s["start"], "end": s["end"], "pseudonym": f"{s['type']}_{counters[s['type']]}"})
    return {"id": doc_id, "annotations": anns}


def write_submission(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
