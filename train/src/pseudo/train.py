"""Fine-tune a token-classification model on data/silver/train.parquet and score it on dev with evaluate.py."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
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

for _noisy in ("httpx", "httpcore", "huggingface_hub", "huggingface_hub.utils._http", "urllib3", "filelock", "fsspec",
               "transformers.modeling_utils", "datasets"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
    logging.getLogger(_noisy).propagate = False


class _DropHttpNoise(logging.Filter):
    """Belt and braces: whatever logger emits them, HTTP request lines never reach the console / train.log."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not (msg.startswith("HTTP Request:") or "unauthenticated requests to the HF Hub" in msg)


for _h in logging.getLogger().handlers:
    _h.addFilter(_DropHttpNoise())


def file_sha1(path: Path, n: int = 12) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def data_version() -> dict:
    """Identifies the exact silver data a run was trained/evaluated on."""
    out = {}
    for name in ("train.parquet", "dev.parquet"):
        p = SILVER / name
        if p.exists():
            out[name] = file_sha1(p)
    return out


def fingerprint(params: dict) -> str:
    """Same model + hyper-parameters + data => same fingerprint => the run can be skipped."""
    return hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def attach_file_log(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    fh.addFilter(_DropHttpNoise())
    logging.getLogger().addHandler(fh)
    logging.getLogger("transformers").addHandler(fh)


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
        grad_accum: int, seed: int, oversample_number: float, fp16: bool, push_to: str | None, max_windows: int | None,
        window: int = 512, stride: int = 128, use_masks: bool = True, grad_checkpointing: bool = False,
        skip_if_done: Path | None = None) -> None:
    torch.manual_seed(seed)
    device = best_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    attach_file_log(out_dir)
    for name in ("train.parquet", "dev.parquet", "dev.jsonl"):
        if not (SILVER / name).exists():
            raise SystemExit(f"data file missing: {SILVER / name} - copy it from Drive (cell 2) before training")
    dv = data_version()
    params = {"model": model_name, "n_docs": n_docs, "epochs": epochs, "lr": lr, "batch": batch, "grad_accum": grad_accum,
              "seed": seed, "oversample_number": oversample_number, "window": window, "stride": stride, "use_masks": use_masks,
              "data_version": dv}
    fp = fingerprint(params)
    if skip_if_done and (skip_if_done / "config.json").exists():
        prev = json.loads((skip_if_done / "config.json").read_text(encoding="utf-8"))
        if prev.get("fingerprint") == fp and prev.get("status") == "done" and (skip_if_done / "model").exists():
            log.info("SKIP %s: identical model + hyper-parameters + data already trained (fingerprint %s) at %s",
                     out_dir.name, fp, skip_if_done)
            if skip_if_done.resolve() != out_dir.resolve():
                shutil.copytree(skip_if_done, out_dir, dirs_exist_ok=True)
            return
    (out_dir / "config.json").write_text(json.dumps({**params, "fingerprint": fp, "status": "running",
                                                     "started": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    log.info("run %s fingerprint %s data %s", out_dir.name, fp, dv)
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
    # weights always in fp32; mixed precision is the Trainer's job (mDeBERTa ships fp16 weights -> "Attempting to unscale FP16 gradients")
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
                                                            label2id=L2I, dtype=torch.float32,
                                                            ignore_mismatched_sizes=True)  # reuse encoders that already carry another NER/PII head
    # predict.py reads the window settings back from the saved config
    model.config.pseudo_window, model.config.pseudo_stride = window, stride
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
    steps_per_epoch = max(1, len(ds) // (batch * grad_accum))
    warmup = int(0.06 * steps_per_epoch * epochs)
    use_bf16 = fp16 and device == "cuda" and torch.cuda.is_bf16_supported()
    log.info("mixed precision: %s", "bf16" if use_bf16 else "fp16" if fp16 and device == "cuda" else "none")
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), per_device_train_batch_size=batch, gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs, learning_rate=lr, warmup_steps=warmup, weight_decay=0.01, lr_scheduler_type="linear",
        logging_steps=200, save_strategy="no", report_to=[], seed=seed,
        # A100 / L4 support bf16: no loss scaling, stable for mDeBERTa; T4 falls back to fp16
        bf16=use_bf16, fp16=fp16 and device == "cuda" and not use_bf16,
        dataloader_num_workers=0, disable_tqdm=False,
    )
    cb = DevEval(tok, dev_rows, device, out_dir)
    log.info("window %d stride %d, masks %s, grad checkpointing %s", window, stride, use_masks, grad_checkpointing)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForTokenClassification(tok), callbacks=[cb])
    res = trainer.train()
    log.info("train loss %.4f, %.1f min", res.training_loss, res.metrics["train_runtime"] / 60)
    model.save_pretrained(out_dir / "model")
    tok.save_pretrained(out_dir / "model")
    (out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=1), encoding="utf-8")
    best = max(cb.history, key=lambda h: h["final_score"]) if cb.history else None
    (out_dir / "config.json").write_text(json.dumps({
        **params, "fingerprint": fp, "status": "done", "n_docs_used": len(train_rows), "windows": len(feats), "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else device,
        "train_loss": res.training_loss, "train_minutes": res.metrics["train_runtime"] / 60,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dev_history": [{"epoch": h["epoch"], "final_score": h["final_score"]} for h in cb.history],
        "best_epoch": best and best["epoch"], "best_dev_f1": best and best["final_score"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    if push_to:
        model.push_to_hub(push_to, commit_message=f"{model_name} docs={len(train_rows)} ep={epochs} lr={lr}")
        tok.push_to_hub(push_to)
        log.info("pushed to %s", push_to)
