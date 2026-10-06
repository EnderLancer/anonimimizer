"""Токенізація та ознаки слів для CRF-моделі."""
import re

TOKEN = re.compile(r"\w+(?:['’ʼ`-]\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text):
    """Повертає список (start, end) для кожного токена."""
    return [(m.start(), m.end()) for m in TOKEN.finditer(text)]


def shape(w):
    s = re.sub(r"[A-ZА-ЯІЇЄҐ]", "A", w)
    s = re.sub(r"[a-zа-яіїєґ]", "a", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)


def word_feats(toks, i):
    w = toks[i]
    lw = w.lower()
    return {
        "b": 1.0, "w": lw, "sh": shape(w),
        "p1": lw[:1], "p2": lw[:2], "p3": lw[:3],
        "s1": lw[-1:], "s2": lw[-2:], "s3": lw[-3:], "s4": lw[-4:],
        "up": w.isupper(), "ti": w.istitle(), "dig": any(c.isdigit() for c in w),
        "ln": min(len(w), 12), "al": w.isalpha(),
    }


def sent2feats(toks):
    """Ознаки токенів речення: власні + контекст ±3 слова + біграми."""
    n = len(toks)
    base = [word_feats(toks, i) for i in range(n)]
    out = []
    for i in range(n):
        f = dict(base[i])
        for off in (-3, -2, -1, 1, 2, 3):
            j = i + off
            if 0 <= j < n:
                b = base[j]
                f[f"{off}:w"] = b["w"]
                f[f"{off}:sh"] = b["sh"]
                if abs(off) <= 2:
                    f[f"{off}:ti"] = b["ti"]
                    f[f"{off}:s3"] = b["s3"]
            else:
                f[f"{off}:pad"] = True
        if i > 0:
            f["bg:w"] = base[i - 1]["w"] + "|" + base[i]["w"]
            f["bg:sh"] = base[i - 1]["sh"] + "|" + base[i]["sh"]
        if i + 1 < n:
            f["bg+:sh"] = base[i]["sh"] + "|" + base[i + 1]["sh"]
        out.append(f)
    return out
