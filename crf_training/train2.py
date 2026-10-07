import pickle, pycrfsuite
from common import sent2feats
data = pickle.load(open("train2.pkl", "rb"))
n_val = 2000
val, tr = data[:n_val], data[n_val:]
t = pycrfsuite.Trainer(verbose=False)
for toks, lab in tr:
    t.append(sent2feats(toks), lab)
t.set_params({"c1": 0.05, "c2": 0.05, "max_iterations": 100, "feature.possible_transitions": True})
t.train("crf2.model")
tg = pycrfsuite.Tagger(); tg.open("crf2.model")
from collections import Counter
tp, fp, fn = Counter(), Counter(), Counter()
def spans(l):
    s, cur = set(), None
    for i, x in enumerate(l + ["O"]):
        if cur and not x.startswith("I-"):
            s.add((cur[0], cur[1], i)); cur = None
        if x.startswith("B-") or (x.startswith("I-") and cur is None): cur = (x[2:], i)
    return s
for toks, lab in val:
    p = spans(tg.tag(sent2feats(toks))); g = spans(lab)
    for e in p & g: tp[e[0]] += 1
    for e in p - g: fp[e[0]] += 1
    for e in g - p: fn[e[0]] += 1
for k in tp.keys() | fn.keys():
    pr = tp[k] / max(1, tp[k] + fp[k]); rc = tp[k] / max(1, tp[k] + fn[k]); print(k, round(pr, 3), round(rc, 3), round(2*pr*rc/max(1e-9, pr+rc), 3))
