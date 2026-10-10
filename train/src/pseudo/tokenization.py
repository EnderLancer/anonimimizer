"""Tokenizers: loading, and token ids with character offsets for fast and SentencePiece-only tokenizers.

Most models ship a fast tokenizer, which returns the character span of every token. The LiBERTa family
(`Goader/liberta-large-v2`, `Goader/modern-liberta-large`) ships a Python tokenizer over a SentencePiece model with
its code in the model repo: it needs `trust_remote_code` and returns no offsets, so they are read from SentencePiece.
"""
from __future__ import annotations


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def piece_offsets(text: str, pieces: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    """Character spans of SentencePiece pieces given as (begin, end, surface).

    SentencePiece reports byte offsets in its newer releases and character offsets in older ones; the surface tells
    which. The whitespace a piece starts with («▁слово») is left out of its span, as fast tokenizers do; a piece that
    is whitespace only keeps it. A piece without surface (a byte of a character outside the vocabulary) takes the span
    of the next piece that has one."""
    if any(text[b:e] != s for b, e, s in pieces):  # byte offsets
        char_at, n = {}, 0
        for i, ch in enumerate(text):
            char_at[n] = i
            n += len(ch.encode("utf-8"))
        char_at[n] = len(text)
        pieces = [(char_at[b], char_at[e], s) for b, e, s in pieces]
    out: list[tuple[int, int]] = []
    following = None
    for b, e, s in reversed(pieces):
        if b == e:
            out.append(following or (b, e))
            continue
        lead = len(s) - len(s.lstrip())
        span = (b + lead, e) if b + lead < e else (b, e)
        out.append(span)
        following = span
    return out[::-1]


def ids_and_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids of the whole text (no special tokens, no truncation) and the character span of each."""
    if tokenizer.is_fast:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    sp = getattr(tokenizer, "sp_model", None)
    if sp is None:
        raise ValueError(f"{type(tokenizer).__name__} is neither a fast tokenizer nor a SentencePiece one: no character offsets")
    try:
        proto = sp.encode(text, out_type="proto")
    except (ValueError, TypeError):  # sentencepiece < 0.2.1
        proto = sp.encode(text, out_type="immutable_proto")
    pieces = list(proto.pieces)
    return [p.id for p in pieces], piece_offsets(text, [(p.begin, p.end, p.surface) for p in pieces])
