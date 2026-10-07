"""Fine-tune a token-classification model on data/silver/train.parquet and score it on dev with evaluate.py."""
from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer,
                          TrainerCallback, TrainingArguments)

from pseudo import DATA, L2I, LABELS, SILVER, TYPES
from pseudo.data import encode
from pseudo.predict import best_device, predict_doc, spans_to_submission, write_submission

log = logging.getLogger("pseudo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def load_docs(path: Path, n: int | None, seed: int, oversample_number: float = 0.0) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    rng = random.Random(seed)
    rng.shuffle(rows)
    if n:
        rows = rows[:n]
    if oversample_number > 0:
        extra = [r for r in rows if any(a["type"] == "НОМЕР" for a in r["annotations"])]
        rng.shuffle(extra)
        rows = rows + extra[: int(len(extra) * oversample_number)]
        rng.shuffle(rows)
    return rows


def evaluate_on_dev(model, tokenizer, dev_rows: list[dict], device: str, out_dir: Path, tag: str) -> dict:
    model.eval()
    sub = [spans_to_submission(d["src_id"], predict_doc(model, tokenizer, d["text"], device)) for d in dev_rows]
    sub_path = out_dir / f"dev_submission_{tag}.jsonl"
    write_submission(sub_path, sub)
    ids = {d["src_id"] for d in dev_rows}
    gold_path = out_dir / "dev_gold.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f, open(SILVER / "dev.jsonl", encoding="utf-8") as g:
        for line in g:
            if json.loads(line)["id"] in ids:
                f.write(line)
    res = subprocess.run([sys.executable, str(DATA / "evaluate.py"), str(gold_path), str(sub_path), "--json"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        log.error("evaluate.py failed: %s", res.stderr[-500:])
        return {}
    metrics = json.loads(res.stdout.strip().splitlines()[-1])
    per = metrics["ner_relaxed"]["per_type"]
    log.info("dev[%s] macro F1 %.4f | %s | strict %.4f | risk %.3f", tag, metrics["final_score"],
             " ".join(f"{t[:3]} {per[t]['f1']:.3f}" for t in TYPES), metrics["ner_strict"]["macro"]["f1"],
             metrics["deanonymization_risk_by_name"])
    model.train()
    return metrics


class DevEval(TrainerCallback):
    def __init__(self, tokenizer, dev_rows, device, out_dir):
        self.tokenizer, self.dev_rows, self.device, self.out_dir = tokenizer, dev_rows, device, out_dir
        self.history = []

    def on_epoch_end(self, args, state, control, model=None, **kw):
        m = evaluate_on_dev(model, self.tokenizer, self.dev_rows, self.device, self.out_dir, f"epoch{int(state.epoch)}")
        if m:
            self.history.append({"epoch": state.epoch, "final_score": m["final_score"], "metrics": m})
            (self.out_dir / "dev_history.json").write_text(json.dumps(self.history, ensure_ascii=False, indent=1), encoding="utf-8")


def run(model_name: str, out_dir: Path, n_docs: int | None, dev_docs: int | None, epochs: float, lr: float, batch: int,
        grad_accum: int, seed: int, oversample_number: float, fp16: bool, push_to: str | None, max_windows: int | None) -> None:
    torch.manual_seed(seed)
    device = best_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    train_rows = load_docs(SILVER / "train.parquet", n_docs, seed, oversample_number)
    dev_rows = load_docs(SILVER / "dev.parquet", dev_docs, seed)
    t0 = time.time()
    feats = []
    for d in train_rows:
        for w in encode(tok, d["text"], d["annotations"], d["masks"]):
            w.pop("offsets")
            feats.append(w)
    if max_windows:
        feats = feats[:max_windows]
    log.info("%s: %d train docs -> %d windows (%.0fs), %d dev docs, device %s", model_name, len(train_rows), len(feats),
             time.time() - t0, len(dev_rows), device)
    ds = Dataset.from_list(feats)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
                                                            label2id=L2I)
    steps_per_epoch = max(1, len(ds) // (batch * grad_accum))
    warmup = int(0.06 * steps_per_epoch * epochs)
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), per_device_train_batch_size=batch, gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs, learning_rate=lr, warmup_steps=warmup, weight_decay=0.01, lr_scheduler_type="linear",
        logging_steps=50, save_strategy="no", report_to=[], seed=seed, fp16=fp16 and device == "cuda",
        dataloader_num_workers=0,
    )
    cb = DevEval(tok, dev_rows, device, out_dir)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForTokenClassification(tok), callbacks=[cb])
    res = trainer.train()
    log.info("train loss %.4f, %.1f min", res.training_loss, res.metrics["train_runtime"] / 60)
    model.save_pretrained(out_dir / "model")
    tok.save_pretrained(out_dir / "model")
    (out_dir / "config.json").write_text(json.dumps({
        "model": model_name, "n_docs": len(train_rows), "windows": len(feats), "epochs": epochs, "lr": lr, "batch": batch,
        "grad_accum": grad_accum, "seed": seed, "oversample_number": oversample_number, "device": device,
        "train_loss": res.training_loss, "train_minutes": res.metrics["train_runtime"] / 60,
        "dev_history": cb.history and [{"epoch": h["epoch"], "final_score": h["final_score"]} for h in cb.history],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    if push_to:
        model.push_to_hub(push_to, commit_message=f"{model_name} docs={len(train_rows)} ep={epochs} lr={lr}")
        tok.push_to_hub(push_to)
        log.info("pushed to %s", push_to)
