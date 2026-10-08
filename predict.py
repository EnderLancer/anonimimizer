"""CRF-псевдонімізатор: python predict.py cases.jsonl out.jsonl"""
import json
import re
import sys
from collections import defaultdict

import pycrfsuite

from common import sent2feats, tokenize

ENDINGS = sorted(["ами", "ями", "ові", "еві", "ого", "ому", "ими", "ої", "ою", "ею", "ом", "ем", "ам", "ям", "ах", "ях",
                  "ів", "їв", "ий", "ій", "ая", "яя", "а", "я", "у", "ю", "е", "є", "и", "і", "ї", "о", "ь", "й"],
                 key=len, reverse=True)


def stem(w):
    w = w.lower().replace("’", "'").replace("ʼ", "'")
    for e in ENDINGS:
        if w.endswith(e) and len(w) - len(e) >= 3:
            return w[: -len(e)]
    return w


def person_words(s):
    ws = re.findall(r"[^\W\d_][\w'’ʼ-]*", s)
    names = [stem(w) for w in ws if len(w) > 1]
    return names


def cluster_persons(mentions):
    """mentions: list of str -> list of group ids (одна особа = одна група)."""
    order = sorted(range(len(mentions)), key=lambda i: -len(person_words(mentions[i])))
    groups = []  # list of set of stems
    gid = [None] * len(mentions)
    sizes = []
    for i in order:
        st = set(person_words(mentions[i]))
        best, bs = None, (0, 0)
        for g, gs in enumerate(groups):
            ov = 0
            for a in st:
                if any(a == b or (len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a))) for b in gs):
                    ov += 1
            need = 2 if len(st) >= 2 and len([1 for x in gs]) >= 2 else 1
            if ov >= need and (ov, sizes[g]) > bs:
                best, bs = g, (ov, sizes[g])
        if best is None:
            groups.append(set(st)); sizes.append(1); gid[i] = len(groups) - 1
        else:
            groups[best] |= st; sizes[best] += 1; gid[i] = best
    return gid


def predict_doc(tagger, text):
    toks = tokenize(text)
    words = [text[a:b] for a, b in toks]
    # довгі документи тегуємо шматками по ~60 токенів (за межами крапки)
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
    # прибрати хвостові розділові знаки, крім крапки після ініціалу
    out = []
    for s, e, t in spans:
        frag = text[s:e]
        while frag and frag[-1] in ",;:)(«»\"" :
            frag = frag[:-1]
        if t == "ОСОБА":
            frag = frag.rstrip()
        e = s + len(frag)
        if e > s and re.search(r"\w", frag):
            out.append((s, e, t))
    return out


def pseudonymize(tagger, text):
    spans = predict_doc(tagger, text)
    labels = {}
    counters = defaultdict(int)
    # ОСОБА: кластеризація
    pi = [k for k, (s, e, t) in enumerate(spans) if t == "ОСОБА"]
    gids = cluster_persons([text[spans[k][0]:spans[k][1]] for k in pi]) if pi else []
    pmap = {}
    for k, g in zip(pi, gids):
        if g not in pmap:
            counters["ОСОБА"] += 1; pmap[g] = counters["ОСОБА"]
        labels[k] = f"ОСОБА_{pmap[g]}"
    seen = {}
    for k, (s, e, t) in enumerate(spans):
        if t == "ОСОБА":
            continue
        key = (t, re.sub(r"\W", "", text[s:e].lower()))
        if key not in seen:
            counters[t] += 1; seen[key] = counters[t]
        labels[k] = f"{t}_{seen[key]}"
    return [{"start": s, "end": e, "pseudonym": labels[k]} for k, (s, e, t) in enumerate(spans)]


def main(src, dst):
    tagger = pycrfsuite.Tagger(); tagger.open("crf.model")
    with open(src, encoding="utf8") as f, open(dst, "w", encoding="utf8") as g:
        for line in f:
            r = json.loads(line)
            g.write(json.dumps({"id": r["id"], "annotations": pseudonymize(tagger, r["original"])}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
