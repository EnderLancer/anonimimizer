"""Повний конвеєр: нейромережа (ОСОБА) + regex-правила + CRF (НОМЕР/ІНФОРМАЦІЯ/АДРЕСА) -> псевдоніми."""
import re
from collections import defaultdict
from pathlib import Path

from . import crf, ner
from .persons import cluster_persons
from .rules import clean_person, rule_spans, spans_overlap

DEFAULT_CRF = Path(__file__).resolve().parent.parent / "models" / "crf2.model"


def build_annotations(text, pers, extra=()):
    """Зливає кандидатів і призначає псевдоніми.

    Політика злиття: ОСОБА від нейромережі фіксується першою; решта кандидатів (правила + CRF)
    додаються від довшого до коротшого, якщо не перетинаються з уже прийнятими.
    """
    spans = []
    for s, e in pers:
        s, e = clean_person(text, s, e)
        if e > s and re.search(r"[^\W\d_]", text[s:e]):
            spans.append((s, e, "ОСОБА"))
    pers_spans = [(a, b) for a, b, t in spans]
    cand = list(rule_spans(text, pers_spans)) + [tuple(x) for x in extra]
    for sp in sorted(cand, key=lambda x: -(x[1] - x[0])):
        if not any(spans_overlap(sp, o) for o in spans):
            spans.append(sp)
    spans.sort()

    labels, counters, seen = {}, defaultdict(int), {}
    pi = [k for k, sp in enumerate(spans) if sp[2] == "ОСОБА"]
    gids = cluster_persons([text[spans[k][0]:spans[k][1]] for k in pi]) if pi else []
    pmap = {}
    for k, g in zip(pi, gids):
        if g not in pmap:
            counters["ОСОБА"] += 1
            pmap[g] = counters["ОСОБА"]
        labels[k] = f"ОСОБА_{pmap[g]}"
    for k, (s, e, t) in enumerate(spans):
        if t == "ОСОБА":
            continue
        key = (t, re.sub(r"\W", "", text[s:e].lower()))   # однакове значення -> однаковий псевдонім
        if key not in seen:
            counters[t] += 1
            seen[key] = counters[t]
        labels[k] = f"{t}_{seen[key]}"
    return [{"start": s, "end": e, "pseudonym": labels[k]} for k, (s, e, t) in enumerate(spans)]


class Pseudonymizer:
    def __init__(self, model_dir=None, crf_path=None, chunk_chars=0, allow_download=True):
        self.chunk_chars = chunk_chars
        self.nlp = ner.load_nlp(ner.ensure_model_dir(allow_download) if model_dir is None else model_dir)
        self.tagger = crf.load_tagger(crf_path or DEFAULT_CRF)

    def __call__(self, text):
        pers = ner.pers_spans_chunked(self.nlp, text, self.chunk_chars)
        return build_annotations(text, pers, extra=crf.crf_spans(self.tagger, text))
