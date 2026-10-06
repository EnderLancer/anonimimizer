"""Групування згадок однієї особи: «Романишин М.», «Романишина», «Мар'яна Романишин» -> один код ОСОБА_n."""
import re

ENDINGS = sorted(["ами", "ями", "ові", "еві", "ого", "ому", "ими", "ої", "ою", "ею", "ом", "ем", "ам", "ям", "ах", "ях",
                  "ів", "їв", "ий", "ій", "ая", "яя", "а", "я", "у", "ю", "е", "є", "и", "і", "ї", "о", "ь", "й"],
                 key=len, reverse=True)


def stem(w):
    """Грубе відкидання відмінкового закінчення (залишаємо щонайменше 3 літери)."""
    w = w.lower().replace("’", "'").replace("ʼ", "'")
    for e in ENDINGS:
        if w.endswith(e) and len(w) - len(e) >= 3:
            return w[: -len(e)]
    return w


def person_words(s):
    ws = re.findall(r"[^\W\d_][\w'’ʼ-]*", s)
    return [stem(w) for w in ws if len(w) > 1]


def cluster_persons(mentions):
    """mentions: список рядків -> список id груп (одна особа = одна група).

    Спочатку обробляються довші згадки (кілька слів), короткі («Петренко», «Соя Д.») приєднуються
    до групи, що має спільну основу слова. Згадка з >=2 слів приєднується до групи з >=2 слів лише
    за >=2 спільних основ (щоб однакове ім'я не зливало різних людей).
    """
    order = sorted(range(len(mentions)), key=lambda i: -len(person_words(mentions[i])))
    groups, sizes = [], []
    gid = [None] * len(mentions)
    for i in order:
        st = set(person_words(mentions[i]))
        best, bs = None, (0, 0)
        for g, gs in enumerate(groups):
            ov = 0
            for a in st:
                if any(a == b or (len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a))) for b in gs):
                    ov += 1
            need = 2 if len(st) >= 2 and len(gs) >= 2 else 1
            if ov >= need and (ov, sizes[g]) > bs:
                best, bs = g, (ov, sizes[g])
        if best is None:
            groups.append(set(st)); sizes.append(1); gid[i] = len(groups) - 1
        else:
            groups[best] |= st; sizes[best] += 1; gid[i] = best
    return gid
