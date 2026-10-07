"""Character spans -> token labels for sliding windows, and back."""
from __future__ import annotations

from pseudo import L2I, LABELS

WINDOW, STRIDE = 512, 128


def _windows(tokenizer, text: str, window: int, stride: int) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """Tokenize the whole text once and cut it into overlapping windows ourselves (transformers 5 no
    longer returns sliding windows from `return_overflowing_tokens`)."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids, offs = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    body = window - 2  # room for <s> … </s>
    step = body - stride
    out = []
    start = 0
    while True:
        chunk_ids = [tokenizer.cls_token_id] + ids[start: start + body] + [tokenizer.sep_token_id]
        chunk_offs = [(0, 0)] + offs[start: start + body] + [(0, 0)]
        out.append((chunk_ids, chunk_offs))
        if start + body >= len(ids):
            break
        start += step
    return out


def encode(tokenizer, text: str, annotations: list[dict], masks: list[dict], window: int = WINDOW, stride: int = STRIDE,
           pad: bool = True) -> list[dict]:
    """Tokenize `text` into overlapping windows; label every token from the char spans.

    A token gets B- when it contains the first char of a span, I- when it is inside, -100 when it
    overlaps a mask or is special/padding.
    """
    n = len(text)
    char_tag = [0] * (n + 1)
    for a in annotations:
        tag_b, tag_i = L2I["B-" + a["type"]], L2I["I-" + a["type"]]
        char_tag[a["start"]] = tag_b
        for i in range(a["start"] + 1, a["end"]):
            char_tag[i] = tag_i
    char_mask = bytearray(n + 1)
    for m in masks:
        for i in range(m["start"], m["end"]):
            char_mask[i] = 1
    out = []
    pad_id = tokenizer.pad_token_id
    for ids, offs in _windows(tokenizer, text, window, stride):
        labels = []
        for (s, e) in offs:
            if s == 0 and e == 0:
                labels.append(-100)
                continue
            if any(char_mask[s:e]):
                labels.append(-100)
                continue
            tags = [t for t in char_tag[s:e] if t]
            if not tags:
                labels.append(0)
            else:
                first = tags[0]
                has_b = any(LABELS[t].startswith("B-") for t in tags)
                labels.append(L2I["B-" + LABELS[first][2:]] if has_b else L2I["I-" + LABELS[first][2:]])
        att = [1] * len(ids)
        if pad and len(ids) < window:
            k = window - len(ids)
            ids = ids + [pad_id] * k
            offs = offs + [(0, 0)] * k
            labels = labels + [-100] * k
            att = att + [0] * k
        out.append({"input_ids": ids, "attention_mask": att, "labels": labels, "offsets": offs})
    return out


def decode(text: str, windows: list[dict], preds: list[list[int]]) -> list[dict]:
    """Per-window token predictions -> char spans. Overlapping windows: the token farther from the
    window edge wins. Same-type B-/I- tokens separated only by whitespace are merged."""
    n = len(text)
    char_lab = [None] * n
    char_conf = [-1] * n
    for w, p in zip(windows, preds):
        L = len(w["offsets"])
        for k, ((s, e), lab) in enumerate(zip(w["offsets"], p)):
            if s == 0 and e == 0:
                continue
            conf = min(k, L - 1 - k)  # distance from the window edge
            for i in range(s, min(e, n)):
                if conf > char_conf[i]:
                    char_lab[i], char_conf[i] = LABELS[lab], conf
    spans, cur = [], None
    for i, lab in enumerate(char_lab):
        if lab and lab != "O":
            t = lab[2:]
            if cur and cur["type"] == t and text[cur["end"]:i].strip() == "" and i - cur["end"] <= 1 and not (lab.startswith("B-") and i > cur["end"] and False):
                cur["end"] = i + 1
            else:
                if cur:
                    spans.append(cur)
                cur = {"start": i, "end": i + 1, "type": t}
        elif cur and text[i].isspace() and i - cur["end"] <= 0:
            continue
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
