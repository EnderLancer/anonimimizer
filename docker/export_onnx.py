"""Експорт token-classification моделі з Hugging Face у ONNX для адаптера onnx: (ner_adapters.py).

    python docker/export_onnx.py <репозиторій або тека моделі> <тека виходу> [--quantize] [--opset 17]

У теку виходу лягають model.onnx (logits за input_ids і attention_mask, динамічні batch і seq), config.json,
tokenizer.json і tokenizer_config.json. --quantize — динамічна int8-квантизація onnxruntime (менша модель,
швидше на CPU, точність може трохи змінитися). Після експорту logits ONNX звіряються з torch.
Потрібні torch, transformers, onnx, onnxruntime; під час інференсу з onnx: — лише onnxruntime і tokenizers.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

SAMPLES = ["Позивач Іваненко Петро Васильович, РНОКПП 1234567890, проживає за адресою м. Київ, вул. Хрещатик, 1.",
           "Суд"]


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(src: str, dst: str, quantize: bool, opset: int, token: str | None) -> None:
    os.makedirs(dst, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(src, token=token)
    model = AutoModelForTokenClassification.from_pretrained(src, token=token, attn_implementation="eager").eval()
    tok.save_pretrained(dst)
    model.config.save_pretrained(dst)
    for f in os.listdir(dst):  # лишаємо лише те, що читає адаптер
        if f not in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            os.remove(os.path.join(dst, f))
    assert os.path.exists(os.path.join(dst, "tokenizer.json")), "потрібен fast-токенізатор (tokenizer.json)"
    enc = tok(SAMPLES, return_tensors="pt", padding=True)
    args = (enc["input_ids"], enc["attention_mask"])
    onnx_file = os.path.join(dst, "model.onnx")
    wrapped = Logits(model).eval()
    dyn = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}}
    with torch.no_grad():
        ref = wrapped(*args).numpy()  # до експорту: legacy-експортер після себе повертає модуль у режим train (dropout)
        try:
            torch.onnx.export(wrapped, args, onnx_file, input_names=["input_ids", "attention_mask"], output_names=["logits"],
                              dynamic_axes=dyn, opset_version=opset, dynamo=False)
        except Exception as e:  # новий експортер (dynamo)
            print(f"legacy export failed ({type(e).__name__}: {e}); trying dynamo", file=sys.stderr)
            from torch.export import Dim
            b, s = Dim("batch"), Dim("seq", max=4096)
            prog = torch.onnx.export(wrapped, args, input_names=["input_ids", "attention_mask"], output_names=["logits"],
                                     dynamic_shapes={"input_ids": {0: b, 1: s}, "attention_mask": {0: b, 1: s}},
                                     opset_version=opset, dynamo=True)
            prog.save(onnx_file)
    for f in list(os.listdir(dst)):  # зовнішні ваги великих моделей складаємо в один файл
        if f.endswith(".onnx_data") or f.endswith(".data"):
            import onnx
            m = onnx.load(onnx_file)
            onnx.save_model(m, onnx_file, save_as_external_data=False)
            os.remove(os.path.join(dst, f))
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        tmp = onnx_file + ".fp32"
        os.replace(onnx_file, tmp)
        quantize_dynamic(tmp, onnx_file, weight_type=QuantType.QInt8)
        os.remove(tmp)
    check(onnx_file, args, ref, quantize)
    print(f"{onnx_file}: {os.path.getsize(onnx_file) / 1e6:.0f} MB, quantize={int(quantize)}")


def check(onnx_file: str, args, ref: np.ndarray, quantize: bool) -> None:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_ids": args[0].numpy(), "attention_mask": args[1].numpy()})[0]
    valid = args[1].numpy().astype(bool)  # на позиціях padding logits не визначені і можуть відрізнятися
    diff = float(np.abs(out - ref)[valid].max())
    same = float((out.argmax(-1) == ref.argmax(-1))[valid].mean())
    print(f"max |onnx - torch| = {diff:.4f}, однакові мітки: {same:.3f}")
    if not quantize and diff > 1e-2:
        sys.exit("ONNX-вихід не збігається з torch")
    if same < 0.95:
        sys.exit("ONNX-мітки не збігаються з torch")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    export(a.src, a.dst, a.quantize, a.opset, os.environ.get("HF_TOKEN") or None)


if __name__ == "__main__":
    main()
