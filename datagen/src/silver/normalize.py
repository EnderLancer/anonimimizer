"""Text normalization applied before any annotation so that offsets are computed once.

ЄДРСР leaves a space before punctuation after placeholders («ОСОБА_1 ,»); the test set has none.
"""
from __future__ import annotations

import re

PLACEHOLDER = re.compile(r"(ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_(\d+)")
ODD_PLACEHOLDER = re.compile(r"\b(?:ІМЕІ\w*|ОСОБИ|ОСОБ)_\d+")

_SPACE_BEFORE_PUNCT = re.compile(r"[ \t ]+([,.;:)»])")
_SPACE_AFTER_OPEN = re.compile(r"([(«])[ \t ]+")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text


def find_placeholders(text: str) -> list[dict]:
    return [
        {"start": m.start(), "end": m.end(), "type": m.group(1), "idx": int(m.group(2))}
        for m in PLACEHOLDER.finditer(text)
    ]
