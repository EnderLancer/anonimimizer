"""Розбір кожної помилки передбачень відносно еталону (relaxed IoU >= 0.5, як в data/evaluate.py).

python analysis/error_analysis.py data/test/court_cases.jsonl results/full.jsonl [--out analysis/out]

Кожен FP/FN потрапляє рівно в один «випадок» (case) — компоненту зв'язності незіставлених фрагментів,
що перетинаються між собою або з уже зіставленими. Для кожного випадку визначаємо категорію:

  missed            еталонний фрагмент, з яким не перетинається жоден передбачений (FN)
  spurious          передбачений фрагмент, що не перетинається з жодним еталонним (FP)
  wrong_type        межі збігаються (IoU >= 0.5), але тип інший (FN + FP)
  wrong_type+bounds тип інший і межі сильно розходяться
  too_wide          той самий тип, передбачений фрагмент містить еталонний, IoU < 0.5
  too_narrow        той самий тип, передбачений фрагмент усередині еталонного, IoU < 0.5
  shifted           той самий тип, перетин без вкладення, IoU < 0.5
  split             один еталонний фрагмент розбито на кілька передбачених
  merged            один передбачений фрагмент накрив кілька еталонних
  extra_fragment    зайвий FP поруч з еталонним, який уже зіставлено з іншим передбаченим
  absorbed          FN, що перетинається з передбаченим, який уже зіставлено з іншим еталонним
  complex           усе інше

Додатково:
  * strict-помилки (relaxed збіг, але межі не до символу) з описом різниці;
  * помилки нумерації осіб (B³): одна особа -> кілька псевдонімів, кілька осіб -> один псевдонім;
  * особи з відкритим ім'ям (deanonymization risk);
  * «oracle»-приріст F1: на скільки зріс би macro F1, якби виправити всі випадки певної категорії.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import evaluate as E  # noqa: E402

CTX = 40


def ctx(text, s, e, n=CTX):
    left = text[max(0, s - n):s].replace("\n", "⏎")
    right = text[e:e + n].replace("\n", "⏎")
    return f"…{left}[[{text[s:e]}]]{right}…".replace("\n", "⏎")


def overlap(a, b):
    return min(a.end, b.end) > max(a.start, b.start)


def boundary_diff(text, g, p):
    """Опис різниці меж, коли тип і перетин збігаються."""
    parts = []
    for side, gs, ps in (("ліва", g.start, p.start), ("права", g.end, p.end)):
        if gs == ps:
            continue
        lo, hi = min(gs, ps), max(gs, ps)
        frag = text[lo:hi]
        wider = (ps < gs) if side == "ліва" else (ps > gs)
        parts.append(f"{side}: {'+' if wider else '-'}{frag!r}")
    return "; ".join(parts)


def boundary_kind(text, g, p):
    """Груба категорія strict-помилки."""
    kinds = []
    for side, gs, ps in (("L", g.start, p.start), ("R", g.end, p.end)):
        if gs == ps:
            continue
        lo, hi = min(gs, ps), max(gs, ps)
        frag = text[lo:hi]
        wider = (ps < gs) if side == "L" else (ps > gs)
        if re.fullmatch(r"[\s.,;:()«»\"'№-]+", frag):
            k = "punct"
        elif re.fullmatch(r"\s*\d[\d\s./-]*\s*", frag):
            k = "digits"
        elif len(frag.split()) == 1:
            k = "1word"
        else:
            k = "words"
        kinds.append(f"{side}{'+' if wider else '-'}{k}")
    return ",".join(kinds)


def analyse(gold_rows, pred_rows):
    preds = {str(r["id"]): r for r in pred_rows}
    cases, strict_rows, numbering_rows, leak_rows = [], [], [], []
    case_id = 0
    # FP/FN по типах після зіставлення, щоб звірити з evaluate.py
    check = {t: Counter() for t in E.TYPES}

    for g_row in gold_rows:
        doc_id, text = str(g_row["id"]), g_row["original"]
        gold = E.spans_from_annotations(g_row["annotations"], len(text), doc_id)
        r = preds.get(doc_id, {"annotations": []})
        pred = E.spans_from_annotations(r.get("annotations", []), len(text), doc_id)

        pairs = E.match(gold, pred, strict=False)
        g2p = {i: j for i, j in pairs}
        p2g = {j: i for i, j in pairs}
        for i, s in enumerate(gold):
            check[s.type]["tp" if i in g2p else "fn"] += 1
        for j, s in enumerate(pred):
            if j not in p2g:
                check[s.type]["fp"] += 1

        # strict-помилки серед relaxed-збігів
        for i, j in pairs:
            g, p = gold[i], pred[j]
            if (g.start, g.end) != (p.start, p.end):
                strict_rows.append(dict(
                    doc=doc_id, type=g.type, gold=text[g.start:g.end], pred=text[p.start:p.end],
                    gold_span=(g.start, g.end), pred_span=(p.start, p.end), iou=round(E.iou(g, p), 3),
                    kind=boundary_kind(text, g, p), diff=boundary_diff(text, g, p),
                    context=ctx(text, min(g.start, p.start), max(g.end, p.end))))

        # компоненти зв'язності незіставлених фрагментів (+ зіставлені сусіди як контекст)
        ug = [i for i in range(len(gold)) if i not in g2p]
        up = [j for j in range(len(pred)) if j not in p2g]
        nodes = [("g", i) for i in ug] + [("p", j) for j in up]
        # додаємо зіставлені фрагменти, що перетинаються з незіставленими
        for i in range(len(gold)):
            if i in g2p and any(overlap(gold[i], pred[j]) for j in up):
                nodes.append(("G", i))
        for j in range(len(pred)):
            if j in p2g and any(overlap(pred[j], gold[i]) for i in ug):
                nodes.append(("P", j))

        def span_of(n):
            return gold[n[1]] if n[0] in "gG" else pred[n[1]]

        parent = {n: n for n in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in nodes:
            for b in nodes:
                if a < b and (a[0].lower() != b[0].lower()) and overlap(span_of(a), span_of(b)):
                    parent[find(a)] = find(b)
        comps = defaultdict(list)
        for n in nodes:
            comps[find(n)].append(n)

        for comp in comps.values():
            fn_nodes = [n for n in comp if n[0] == "g"]
            fp_nodes = [n for n in comp if n[0] == "p"]
            if not fn_nodes and not fp_nodes:
                continue
            mg = [n for n in comp if n[0] == "G"]
            mp = [n for n in comp if n[0] == "P"]
            G = [gold[n[1]] for n in fn_nodes]
            P = [pred[n[1]] for n in fp_nodes]

            if fn_nodes and not fp_nodes and not mp:
                cat = "missed"
            elif fp_nodes and not fn_nodes and not mg:
                cat = "spurious"
            elif len(G) == 1 and len(P) == 1 and not mg and not mp:
                g, p = G[0], P[0]
                v = E.iou(g, p)
                if g.type != p.type:
                    cat = "wrong_type" if v >= 0.5 else "wrong_type+bounds"
                elif p.start <= g.start and p.end >= g.end:
                    cat = "too_wide"
                elif g.start <= p.start and g.end >= p.end:
                    cat = "too_narrow"
                else:
                    cat = "shifted"
            elif len(G) == 1 and len(P) > 1 and not mg and not mp:
                cat = "split" if all(p.type == G[0].type for p in P) else "split+wrong_type"
            elif len(P) == 1 and len(G) > 1 and not mg and not mp:
                cat = "merged" if all(g.type == P[0].type for g in G) else "merged+wrong_type"
            elif fp_nodes and not fn_nodes and mg:
                cat = "extra_fragment"
            elif fn_nodes and not fp_nodes and mp:
                cat = "absorbed"
            else:
                cat = "complex"

            all_spans = G + P + [gold[n[1]] for n in mg] + [pred[n[1]] for n in mp]
            lo, hi = min(s.start for s in all_spans), max(s.end for s in all_spans)
            case_id += 1
            types = sorted({s.type for s in G + P})
            cases.append(dict(
                case=case_id, doc=doc_id, category=cat,
                type="/".join(types),
                gold_types="+".join(s.type for s in G), pred_types="+".join(s.type for s in P),
                n_fn=len(G), n_fp=len(P),
                gold=" | ".join(f"{s.type}:{text[s.start:s.end]}" for s in sorted(G, key=lambda s: s.start)),
                pred=" | ".join(f"{s.type}:{text[s.start:s.end]}" for s in sorted(P, key=lambda s: s.start)),
                matched_gold=" | ".join(f"{gold[n[1]].type}:{text[gold[n[1]].start:gold[n[1]].end]}" for n in mg),
                matched_pred=" | ".join(f"{pred[n[1]].type}:{text[pred[n[1]].start:pred[n[1]].end]}" for n in mp),
                start=lo, end=hi,
                iou=round(E.iou(G[0], P[0]), 3) if len(G) == 1 and len(P) == 1 else None,
                fn_by_type=dict(Counter(s.type for s in G)), fp_by_type=dict(Counter(s.type for s in P)),
                context=ctx(text, lo, hi),
            ))

        # нумерація осіб (B³)
        gp = [s for s in gold if s.type == "ОСОБА"]
        pp = [s for s in pred if s.type == "ОСОБА"]
        ppairs = E.match(gp, pp, strict=False)
        by_gold, by_pred = defaultdict(set), defaultdict(set)
        mentions = defaultdict(list)
        for i, j in ppairs:
            by_gold[gp[i].label].add(pp[j].label)
            by_pred[pp[j].label].add(gp[i].label)
            mentions[(gp[i].label, pp[j].label)].append(text[gp[i].start:gp[i].end])
        for gl, pls in by_gold.items():
            if len(pls) > 1:
                numbering_rows.append(dict(
                    doc=doc_id, kind="одна особа -> кілька псевдонімів", gold_label=gl, pred_labels=sorted(pls),
                    mentions={pl: sorted(set(mentions[(gl, pl)])) for pl in sorted(pls)}))
        for pl, gls in by_pred.items():
            if len(gls) > 1:
                numbering_rows.append(dict(
                    doc=doc_id, kind="кілька осіб -> один псевдонім", pred_label=pl, gold_labels=sorted(gls),
                    mentions={gl: sorted(set(mentions[(gl, pl)])) for gl in sorted(gls)}))

        # витоки імен
        persons = defaultdict(list)
        for s in gold:
            if s.type == "ОСОБА":
                persons[s.label].append(s)
        for label, ms in persons.items():
            open_ = [m for m in ms if not E.covered(text, m, pred)]
            if open_:
                leak_rows.append(dict(doc=doc_id, gold_label=label, leaked=[text[m.start:m.end] for m in open_],
                                      context=[ctx(text, m.start, m.end) for m in open_]))

    return (pd.DataFrame(cases), pd.DataFrame(strict_rows), pd.DataFrame(numbering_rows),
            pd.DataFrame(leak_rows), check)


STATE_BODY = re.compile(
    r"(Головн\w+ управлінн\w+|Управлінн\w+|Міністерств\w+|Департамент\w*|територіальн\w+ центр\w+|"
    r"Державн\w+ \w+ служб\w+|Національн\w+ поліці\w+)", re.I)
COMPANY = re.compile(r"(товариств\w+|ТОВ|ПрАТ|ПАТ|ФОП)\b|\"[^\"]+\"|«[^»]+»", re.I)
DATE = re.compile(r"^\d{1,2}(\.\d{2}\.\d{4}| \w+ \d{4})|^\d{4}-\d{2}-\d{2}|^\d{2}\.\d{2}\.\d{4}")


def reason(r, text):
    """Пояснення випадку: (причина, чия помилка). Чия: model — помилка пайплайну, gold — помилка або
    непослідовність розмітки, ambiguous — розмітка суперечить сама собі, правильної відповіді немає."""
    cat, g, p = r["category"], r["gold"] or "", r["pred"] or ""
    left = text[max(0, r["start"] - 40):r["start"]]
    right = text[r["end"]:r["end"] + 25]
    gt, pt = r["gold_types"], r["pred_types"]
    if cat == "wrong_type" and gt == "ІНФОРМАЦІЯ" and pt == "ОСОБА":
        return "розмітка: ПІБ розмічено як ІНФОРМАЦІЯ", "gold"
    if cat == "wrong_type" and gt == "ІНФОРМАЦІЯ" and pt == "АДРЕСА":
        return "розмітка: адресу розмічено як ІНФОРМАЦІЯ", "gold"
    if cat.startswith("wrong_type") and "частин" in g:
        return "«Військова частина А1234» розмічено як ІНФОРМАЦІЯ (в інших документах лише НОМЕР «А1234»)", "ambiguous"
    if cat == "spurious":
        if pt == "ОСОБА":
            if " проти " in p or re.search(r"(справ\w*|рішенн\w*)\s*[«\"]\s*$", left) or re.match(r"\s*(\)\s*)?(v\.|проти)", right) \
                    or re.search(r"[«\"(]\s*$", left) and re.search(r"v\.|проти", text[r["end"]:r["end"] + 60]):
                return "назва справи ЄСПЛ («X проти України», «X v. Spain»)", "model"
            if "Верховн" in p:
                return "орган («Верховна Рада») як особа", "model"
            if re.search(r"(імені|полковника)[^.]{0,30}$", left):
                return "ім'я в назві установи («імені … полковника Євгенія Пікуса»)", "model"
            if re.match(r"\s*\(код", right):
                return "нерезидент-фізособа, у розмітці пропущено", "gold"
            return "інша не-особа", "model"
        if pt == "ІНФОРМАЦІЯ":
            if STATE_BODY.search(p):
                return "держорган (у цьому документі розмітка його не маскує)", "ambiguous"
            if DATE.search(p.split(":", 1)[1]):
                return "дата, не розмічена (подібна дата в документі розмічена)", "ambiguous"
            return "інше", "model"
        if pt == "АДРЕСА":
            if re.search(r"суд\w*[^.]{0,25}адрес\w*:?\s*$", left, re.I):
                return "адреса суду (публічна)", "model"
            if re.search(r"ЄДРПОУ|Верховної Ради|ДПС|Київстар", left + right):
                return "адреса установи / юрособи (публічна)", "model"
            return "адреса місця події, у розмітці не замаскована", "ambiguous"
        if pt == "НОМЕР":
            v = p.split(":", 1)[1]
            if v.startswith("UA"):
                return "IBAN установи / казначейства", "ambiguous"
            if len(v) <= 3:
                return "коротке число (день місяця / № машиномісця)", "model"
            return "номер документа (дозволу), не розмічений", "ambiguous"
    if cat == "missed":
        v = g.split(":", 1)[1]
        if gt == "ІНФОРМАЦІЯ":
            if DATE.search(v):
                return "дата народження/смерті без маркера «народження»", "model"
            if COMPANY.search(v) or re.search(r"\"|«", left[-3:] + v):
                return "назва приватної компанії (ТОВ/ПрАТ/торгова марка)", "ambiguous"
            return "інше", "model"
        if gt == "НОМЕР":
            if v.startswith("UA"):
                return "IBAN після «рахунок №» відкинуто фільтром казначейських IBAN", "model"
            if re.match(r"[А-ЯІЇЄҐA-Z]{2}-\d+", v):
                return "серія-номер документа з дефісом (СА-452581, НВ-121545)", "model"
            if re.search(r"(код\w*|ідентифікаційн\w+ код\w* юридичної особи)\s*$", left):
                return "код ЄДРПОУ юрособи (8 цифр)", "model"
            if re.search(r"номер об'єкт\w* нерухомого майна\s*$", left):
                return "реєстраційний номер об'єкта нерухомості (12 цифр)", "model"
            if re.search(r"паспорт", left):
                return "номер паспорта (ID-картка, 9 цифр)", "model"
            if re.search(r"посвідчен\w*|Юрія|Марини", left):
                return "номер посвідчення після «ПІБ №»", "model"
            if re.search(r"від \d+ \w+ \d{4} року №\s*$", left):
                return "номер постанови виконавця після «від <дата> №»", "model"
            return "номер після «№» без ключового слова", "model"
        if gt == "АДРЕСА":
            return "адреса без маркера «за адресою / проживає» (після «по», «на адресу»)", "model"
    if cat == "too_narrow":
        if gt == "АДРЕСА":
            return "CRF обірвав адресу на «, м.» (місто, область, індекс не увійшли)", "model"
        return "regex зняв роль «Головуючий» і залишив лише ініціал «В.» (+ обрізано крапку)", "model"
    if cat == "absorbed":
        return "номер ділянки «№ 384551» поглинуто адресою, а адресу обрізано посередині номера", "model"
    if cat == "complex":
        if "DF987654" in g + p:
            return "розмітка: межу між ПІБ і номером поставлено неправильно («Петренко» | «Аліси Володимирівни DF987654»)", "gold"
        return "розмітка: дату «15» відрізано від «листопада 2020 року», яке приклеєно до ПІБ", "gold"
    return "не класифіковано", "unclassified"


def gold_consistency(cases, gold_rows):
    """Для хибних і пропущених фрагментів: скільки разів той самий рядок трапляється в корпусі
    і скільки з цих разів розмітка його маскує (тим самим типом)."""
    texts = {str(r["id"]): r["original"] for r in gold_rows}
    covered = defaultdict(list)
    for r in gold_rows:
        for a in r["annotations"]:
            covered[str(r["id"])].append((a["start"], a["end"], a["pseudonym"].rsplit("_", 1)[0]))
    out = []
    for _, r in cases.iterrows():
        src = r["pred"] if r["category"] == "spurious" else r["gold"] if r["category"] == "missed" else None
        if not isinstance(src, str) or " | " in src:
            out.append("")
            continue
        t, s = src.split(":", 1)
        tot = ann = 0
        for d, text in texts.items():
            for m in re.finditer(re.escape(s), text):
                tot += 1
                ann += any(a <= m.start() and m.end() <= b and tt == t for a, b, tt in covered[d])
        out.append(f"{ann}/{tot}")
    return out


def macro_f1(counts):
    f = []
    for t in E.TYPES:
        c = counts[t]
        f.append(E.prf(c["tp"], c["fp"], c["fn"])["f1"])
    return sum(f) / len(f), dict(zip(E.TYPES, f))


def oracle_gains(cases, check):
    """Приріст macro F1, якщо виправити всі випадки категорії: FN стають TP, FP зникають."""
    base, _ = macro_f1(check)
    rows = []
    for key, grp in list(cases.groupby("category")) + [(f"{c}@{t}", g) for (c, t), g in cases.groupby(["category", "type"])]:
        c = {t: Counter(v) for t, v in check.items()}
        for _, r in grp.iterrows():
            for t, n in r.fn_by_type.items():
                c[t]["fn"] -= n
                c[t]["tp"] += n
            for t, n in r.fp_by_type.items():
                c[t]["fp"] -= n
        f, _ = macro_f1(c)
        rows.append(dict(group=key, cases=len(grp), fn=int(grp.n_fn.sum()), fp=int(grp.n_fp.sum()),
                         f1_if_fixed=round(f, 4), gain=round(f - base, 4)))
    return base, pd.DataFrame(rows).sort_values("gain", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("submission")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gold_rows = E.load_jsonl(a.truth)
    cases, strict, numbering, leaks, check = analyse(gold_rows, E.load_jsonl(a.submission))
    texts = {str(r["id"]): r["original"] for r in gold_rows}
    rs = [reason(r, texts[r["doc"]]) for _, r in cases.iterrows()]
    cases["reason"] = [x[0] for x in rs]
    cases["fault"] = [x[1] for x in rs]
    cases["gold_annotated_in_corpus"] = gold_consistency(cases, gold_rows)
    # 1 - F1_t = (FP_t + FN_t) / (2 TP_t + FP_t + FN_t), тож кожен FP/FN типу t коштує рівно
    # 1 / (4 (2 TP_t + FP_t + FN_t)) macro F1, і сума по всіх випадках дорівнює 1 - macro F1.
    unit = {t: 1 / (len(E.TYPES) * (2 * c["tp"] + c["fp"] + c["fn"])) for t, c in check.items()}
    cases["loss_pts"] = [round(100 * sum(n * unit[t] for d in (r.fn_by_type, r.fp_by_type) for t, n in d.items()), 4)
                         for _, r in cases.iterrows()]
    ref = E.evaluate(E.load_jsonl(a.truth), E.load_jsonl(a.submission))["ner_relaxed"]["per_type"]
    for t in E.TYPES:
        for k in ("tp", "fp", "fn"):
            assert ref[t][k] == check[t][k], (t, k, ref[t][k], check[t][k])
    # кожен FP/FN рівно в одному випадку
    for t in E.TYPES:
        assert sum(d.get(t, 0) for d in cases.fn_by_type) == check[t]["fn"]
        assert sum(d.get(t, 0) for d in cases.fp_by_type) == check[t]["fp"]

    base, gains = oracle_gains(cases, check)
    print(f"macro F1 = {base:.4f}  втрачено {1 - base:.4f}")
    print(f"випадків: {len(cases)}  (FN {cases.n_fn.sum()}, FP {cases.n_fp.sum()}), "
          f"сума loss_pts = {cases.loss_pts.sum():.2f} пункта\n")
    print("ціна одного FP/FN у пунктах F1:", {t: round(100 * u, 4) for t, u in unit.items()}, "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 120)
    print(pd.crosstab(cases.category, cases.type, margins=True), "\n")
    print(gains.to_string(index=False), "\n")
    print(cases.groupby(["fault", "category", "type", "reason"]).agg(cases=("case", "size"), fn=("n_fn", "sum"),
                                                                    fp=("n_fp", "sum"), loss_pts=("loss_pts", "sum")).round(3).to_string(), "\n")
    by_fault = []
    for fault, grp in cases.groupby("fault"):
        c = {t: Counter(v) for t, v in check.items()}
        for _, r in grp.iterrows():
            for t, n in r.fn_by_type.items():
                c[t]["fn"] -= n
                c[t]["tp"] += n
            for t, n in r.fp_by_type.items():
                c[t]["fp"] -= n
        by_fault.append(dict(fault=fault, cases=len(grp), loss_pts=round(grp.loss_pts.sum(), 3), f1_if_fixed=round(macro_f1(c)[0], 4),
                             gain=round(macro_f1(c)[0] - base, 4)))
    print(pd.DataFrame(by_fault).to_string(index=False), "\n")
    print("strict-помилки:", len(strict))
    if len(strict):
        print(pd.crosstab(strict.kind, strict.type, margins=True), "\n")
    print("помилки нумерації осіб:", len(numbering))
    print("відкриті особи:", len(leaks))

    cols = ["case", "doc", "category", "type", "fault", "reason", "gold", "pred", "matched_gold", "matched_pred",
            "iou", "loss_pts", "gold_annotated_in_corpus", "context"]
    cases[cols].to_csv(os.path.join(a.out, "cases.csv"), index=False)
    strict.to_csv(os.path.join(a.out, "strict.csv"), index=False)
    numbering.to_json(os.path.join(a.out, "numbering.json"), orient="records", force_ascii=False, indent=1)
    leaks.to_json(os.path.join(a.out, "leaks.json"), orient="records", force_ascii=False, indent=1)
    gains.to_csv(os.path.join(a.out, "oracle_gains.csv"), index=False)
    write_cases_md(cases, strict, os.path.join(a.out, "cases.md"))


def md(s):
    return str(s).replace("|", "\\|") if isinstance(s, str) else ""


def write_cases_md(cases, strict, path):
    """Усі випадки, згруповані за «чия помилка» -> причина, з ціною в пунктах F1."""
    out = [f"# Усі {len(cases)} випадків ({cases.loss_pts.sum():.2f} пункта F1)\n",
           "Згенеровано `analysis/error_analysis.py`. `loss` — частка втрачених пунктів macro F1.\n"]
    order = cases.groupby("fault").loss_pts.sum().sort_values(ascending=False).index
    for fault in order:
        f = cases[cases.fault == fault]
        out.append(f"\n## {fault}: {len(f)} випадків, {f.loss_pts.sum():.2f} пункта\n")
        for rsn, g in sorted(f.groupby("reason"), key=lambda x: -x[1].loss_pts.sum()):
            out.append(f"\n### {rsn} ({len(g)}, {g.loss_pts.sum():.3f})\n")
            out.append("| # | doc | категорія | еталон | передбачено | loss | у розмітці | контекст |")
            out.append("|---|---|---|---|---|---|---|---|")
            for _, r in g.iterrows():
                out.append(f"| {r.case} | {r.doc} | {r.category} | {md(r.gold)} | {md(r.pred)} | {r.loss_pts:.3f} | "
                           f"{r.gold_annotated_in_corpus} | {md(r.context)} |")
    out.append(f"\n## Strict-помилки ({len(strict)}): relaxed-збіг, але межі не до символу (на підсумковий бал не впливають)\n")
    out.append("| doc | тип | різниця | еталон | передбачено |")
    out.append("|---|---|---|---|---|")
    for _, r in strict.sort_values(["type", "kind"]).iterrows():
        out.append(f"| {r.doc} | {r.type} | {md(r['diff'])} | {md(r.gold)} | {md(r.pred)} |")
    open(path, "w", encoding="utf8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
