"""Повний конвеєр: нейромережа (ОСОБА) + regex-правила + CRF (НОМЕР/ІНФОРМАЦІЯ/АДРЕСА) -> псевдоніми."""
import re
from collections import defaultdict
from pathlib import Path

from . import crf, ner
from .persons import cluster_persons
from .rules import clean_person, rule_spans, spans_overlap

DEFAULT_CRF = Path(__file__).resolve().parent.parent / "models" / "crf2.model"


def _ann(text, s, e, t, source):
    return {"start": s, "end": e, "type": t, "text": text[s:e], "metadata": {"source": source}}


def candidates(text, pers, extra=(), person_source="ner", extra_source="crf"):
    """Усі кандидати до злиття, без псевдонімів: ОСОБА (pers), правила, CRF (extra).

    Кожен кандидат: {"start", "end", "type", "text", "metadata": {"source"}}. Порядок важливий для combine:
    спершу ОСОБА, потім правила, потім extra.
    """
    out = []
    for s, e in pers:
        s, e = clean_person(text, s, e)
        if e > s and re.search(r"[^\W\d_]", text[s:e]):
            out.append(_ann(text, s, e, "ОСОБА", person_source))
    pers_spans = [(a["start"], a["end"]) for a in out]
    out += [_ann(text, s, e, t, "rules") for s, e, t in rule_spans(text, pers_spans)]
    out += [_ann(text, x[0], x[1], x[2], extra_source) for x in extra]
    return out


def combine(text, cands):
    """Зливає кандидатів і призначає псевдоніми. Повертає кандидатів-переможців з полем pseudonym.

    Політика злиття: ОСОБА фіксується першою; решта кандидатів (правила + CRF)
    додаються від довшого до коротшого, якщо не перетинаються з уже прийнятими.
    """
    spans = [a for a in cands if a["type"] == "ОСОБА"]
    rest = [a for a in cands if a["type"] != "ОСОБА"]
    for a in sorted(rest, key=lambda a: -(a["end"] - a["start"])):
        if not any(spans_overlap((a["start"], a["end"]), (o["start"], o["end"])) for o in spans):
            spans.append(a)
    spans.sort(key=lambda a: (a["start"], a["end"], a["type"]))

    labels, counters, seen = {}, defaultdict(int), {}
    pi = [k for k, a in enumerate(spans) if a["type"] == "ОСОБА"]
    gids = cluster_persons([text[spans[k]["start"]:spans[k]["end"]] for k in pi]) if pi else []
    pmap = {}
    for k, g in zip(pi, gids):
        if g not in pmap:
            counters["ОСОБА"] += 1
            pmap[g] = counters["ОСОБА"]
        labels[k] = f"ОСОБА_{pmap[g]}"
    for k, a in enumerate(spans):
        t = a["type"]
        if t == "ОСОБА":
            continue
        key = (t, re.sub(r"\W", "", text[a["start"]:a["end"]].lower()))   # однакове значення -> однаковий псевдонім
        if key not in seen:
            counters[t] += 1
            seen[key] = counters[t]
        labels[k] = f"{t}_{seen[key]}"
    return [{**a, "pseudonym": labels[k]} for k, a in enumerate(spans)]


def strip(annotations):
    """Лише поля, які читає data/evaluate.py."""
    return [{"start": a["start"], "end": a["end"], "pseudonym": a["pseudonym"]} for a in annotations]


def build_annotations(text, pers, extra=()):
    """candidates -> combine -> strip: анотації у форматі data/evaluate.py."""
    return strip(combine(text, candidates(text, pers, extra)))


class Pseudonymizer:
    def __init__(self, model_dir=None, crf_path=None, chunk_chars=0, allow_download=True):
        self.chunk_chars = chunk_chars
        self.nlp = ner.load_nlp(ner.ensure_model_dir(allow_download) if model_dir is None else model_dir)
        self.tagger = crf.load_tagger(crf_path or DEFAULT_CRF)

    def candidates(self, text):
        pers = ner.pers_spans_chunked(self.nlp, text, self.chunk_chars)
        return candidates(text, pers, extra=crf.crf_spans(self.tagger, text))

    def __call__(self, text):
        return strip(combine(text, self.candidates(text)))
