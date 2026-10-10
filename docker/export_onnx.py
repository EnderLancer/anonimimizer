"""Експорт token-classification моделі з Hugging Face у ONNX для адаптера onnx: (ner_adapters.py).

    python docker/export_onnx.py <тека моделі> <тека виходу> [--quantize] [--opset 17]

У теку виходу лягають model.onnx (logits за input_ids і attention_mask, динамічні batch і seq), config.json,
токенізатор (tokenizer.json або spm.model) і tokenizer_config.json. --quantize — динамічна int8-квантизація
onnxruntime (менша модель, швидше на CPU, точність може трохи змінитися). Після експорту logits ONNX звіряються
з torch на текстах іншої довжини, ніж приклад для експорту: так видно, якщо довжину «запекло» в граф.
Код токенізатора з репозиторію моделі не виконується: приклади кодуються бібліотекою tokenizers або sentencepiece.
Потрібні torch, transformers, onnx, onnxruntime; під час інференсу з onnx: — лише onnxruntime і токенізатор.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from transformers import AutoModelForTokenClassification

SENTENCE = ("Позивач Іваненко Петро Васильович, 12 березня 1980 року народження, РНОКПП 1234567890, проживає за адресою: "
            "м. Київ, вул. Хрещатик, буд. 1, кв. 15, тел. +380501234567, звернувся до суду з позовом до військової частини А1234. ")
EXPORT_SAMPLES = [SENTENCE * 6, "Суд"]                           # приклад для експорту: довгий і короткий (padding)
CHECK_SAMPLES = [[SENTENCE, "Суд ухвалив."], [SENTENCE * 14]]    # перевірка на інших довжинах


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def encoder(src: str, pad_id: int):
    """Функція тексти -> (input_ids, attention_mask) зі спеціальними токенами і доповненням, без transformers."""
    cfg_file = os.path.join(src, "tokenizer_config.json")
    cfg = json.load(open(cfg_file, encoding="utf8")) if os.path.exists(cfg_file) else {}
    if os.path.exists(os.path.join(src, "tokenizer.json")):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(src, "tokenizer.json"))
        tok.no_truncation()
        one = lambda t: tok.encode(t, add_special_tokens=True).ids
    elif os.path.exists(os.path.join(src, "spm.model")):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(os.path.join(src, "spm.model"))
        name = lambda k, d: (cfg.get(k).get("content") if isinstance(cfg.get(k), dict) else cfg.get(k)) or d
        cls, sep, pad_id = (sp.PieceToId(name(k, d)) for k, d in (("cls_token", "<cls>"), ("sep_token", "<sep>"), ("pad_token", "<pad>")))
        one = lambda t: [cls] + sp.encode(t) + [sep]
    else:
        sys.exit("потрібен tokenizer.json (fast-токенізатор) або spm.model (SentencePiece)")

    def encode(texts):
        rows = [one(t) for t in texts]
        width = max(map(len, rows))
        ids = torch.tensor([r + [pad_id] * (width - len(r)) for r in rows], dtype=torch.long)
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], dtype=torch.long)
        return ids, mask
    return encode


def export(src: str, dst: str, quantize: bool, opset: int) -> None:
    os.makedirs(dst, exist_ok=True)
    model = AutoModelForTokenClassification.from_pretrained(src, attn_implementation="eager").eval()
    limit = getattr(model.config, "max_position_embeddings", 512)
    for f in ("config.json", "tokenizer.json", "spm.model", "tokenizer_config.json"):  # лише те, що читає адаптер
        if os.path.exists(os.path.join(src, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    encode = encoder(src, model.config.pad_token_id or 0)
    clip = lambda pair: tuple(x[:, :limit - 2] for x in pair)  # моделі з вікном 512: приклад не довший за ліміт
    args = clip(encode(EXPORT_SAMPLES))
    checks = [args] + [clip(encode(c)) for c in CHECK_SAMPLES]
    onnx_file = os.path.join(dst, "model.onnx")
    wrapped = Logits(model).eval()
    dyn = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}}
    with torch.no_grad():
        refs = [wrapped(*c).numpy() for c in checks]  # до експорту: legacy-експортер повертає модуль у режим train (dropout)
        try:
            torch.onnx.export(wrapped, args, onnx_file, input_names=["input_ids", "attention_mask"], output_names=["logits"],
                              dynamic_axes=dyn, opset_version=opset, dynamo=False)
        except Exception as e:  # новий експортер (dynamo)
            print(f"legacy export failed ({type(e).__name__}: {e}); trying dynamo", file=sys.stderr)
            from torch.export import Dim
            b, s = Dim("batch"), Dim("seq", max=limit)
            prog = torch.onnx.export(wrapped, args, input_names=["input_ids", "attention_mask"], output_names=["logits"],
                                     dynamic_shapes={"input_ids": {0: b, 1: s}, "attention_mask": {0: b, 1: s}},
                                     opset_version=opset, dynamo=True)
            prog.save(onnx_file)
    del model, wrapped
    extra = [f for f in os.listdir(dst) if f not in ("model.onnx", "config.json", "tokenizer.json", "spm.model", "tokenizer_config.json")]
    if extra:  # зовнішні ваги складаємо в один файл
        import onnx
        onnx.save_model(onnx.load(onnx_file), onnx_file, save_as_external_data=False)
        for f in extra:
            os.remove(os.path.join(dst, f))
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        tmp = onnx_file + ".fp32"
        os.replace(onnx_file, tmp)
        quantize_dynamic(tmp, onnx_file, weight_type=QuantType.QInt8)
        os.remove(tmp)
    check(onnx_file, checks, refs, quantize)
    print(f"{onnx_file}: {os.path.getsize(onnx_file) / 1e6:.0f} MB, quantize={int(quantize)}")


def check(onnx_file: str, checks, refs, quantize: bool) -> None:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    for (ids, mask), ref in zip(checks, refs):
        out = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
        valid = mask.numpy().astype(bool)  # на позиціях padding logits не визначені і можуть відрізнятися
        diff = float(np.abs(out - ref)[valid].max())
        same = float((out.argmax(-1) == ref.argmax(-1))[valid].mean())
        print(f"{tuple(ids.shape)}: max |onnx - torch| = {diff:.4f}, однакові мітки: {same:.3f}")
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
    export(a.src, a.dst, a.quantize, a.opset)


if __name__ == "__main__":
    main()
