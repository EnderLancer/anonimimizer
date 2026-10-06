"""Друга модель: CRF (python-crfsuite) для НОМЕР / ІНФОРМАЦІЯ / АДРЕСА."""
import re

from .text import sent2feats, tokenize

CRF_TYPES = ("НОМЕР", "ІНФОРМАЦІЯ", "АДРЕСА")


def load_tagger(path):
    import pycrfsuite
    tagger = pycrfsuite.Tagger()
    tagger.open(str(path))
    return tagger


def predict_doc(tagger, text):
    """Теґує документ шматками до ~80 токенів і повертає спани (start, end, type) усіх типів моделі."""
    toks = tokenize(text)
    words = [text[a:b] for a, b in toks]
    labels = []
    i = 0
    while i < len(words):
        j = min(len(words), i + 80)
        if j < len(words):
            k = j
            while k > i + 30 and words[k - 1] not in (".", ";", "\n"):
                k -= 1
            j = k if k > i + 30 else j
        labels += tagger.tag(sent2feats(words[i:j]))
        i = j
    spans, cur = [], None
    for idx, l in enumerate(labels + ["O"]):
        if cur and (l == "O" or l.startswith("B-") or l[2:] != cur[0]):
            spans.append((cur[1], toks[cur[2]][1], cur[0])); cur = None
        if l.startswith("B-") or (l.startswith("I-") and cur is None):
            cur = [l[2:], toks[idx][0], idx]
        elif cur and l.startswith("I-"):
            cur[2] = idx
    out = []
    for s, e, t in spans:
        frag = text[s:e]
        while frag and frag[-1] in ",;:)(«»\"":
            frag = frag[:-1]
        if t == "ОСОБА":
            frag = frag.rstrip()
        e = s + len(frag)
        if e > s and re.search(r"\w", frag):
            out.append((s, e, t))
    return out


def crf_spans(tagger, text):
    """Лише типи, за які відповідає CRF; ОСОБА від CRF ігноруємо (її знаходить нейромережа)."""
    return [x for x in predict_doc(tagger, text) if x[2] in CRF_TYPES]
