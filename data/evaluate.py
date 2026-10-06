#!/usr/bin/env python3
"""Оцінка рішень челенджу «Псевдонімізація персональних даних у текстах».

Запуск:
    python evaluate.py test/court_cases.jsonl submission.jsonl
    python evaluate.py test/court_cases.jsonl submission.jsonl --json      # результат одним JSON
    python evaluate.py test/court_cases.jsonl submission.jsonl --cosine    # потрібен sentence-transformers

Відповідь — JSON Lines, один рядок на рішення, в одному з двох форматів:

    {"id": 116075957, "annotations": [{"start": 19, "end": 42, "pseudonym": "ОСОБА_1"}, ...]}
    {"id": 116075957, "pseudonymized": "У травні 2018 року ОСОБА_1 звернувся до суду ..."}

    start, end — позиції символів у полі original (end не включно).
    pseudonymized — original, у якому персональні дані замінено псевдонімами, а решта тексту не змінена.
    Скрипт сам знаходить у ньому позиції замін. Точним є лише перший формат: якщо два псевдоніми
    стоять поруч через пробіл, межу між ними можна визначити тільки приблизно.

Рішення, якого немає у відповіді, вважається рішенням без знайдених персональних даних.

Як рахуємо:
  1. NER F1. Знайдений фрагмент зараховується, якщо його тип збігається з правильним, а перетин
     фрагментів IoU >= 0.5. Кожен правильний фрагмент зараховується не більше одного разу.
     F1 рахуємо для кожного типу (ОСОБА, АДРЕСА, НОМЕР, ІНФОРМАЦІЯ) і усереднюємо.
     Strict F1 — те саме, але межі мають збігатися до символу.
  2. Deanonymization risk (by name) — частка осіб, у яких хоча б одну згадку імені не приховано.
     Згадка відкрита, якщо хоч одне її слово замасковано менш ніж наполовину.
  3. Consistency — B³ F1: чи однаковий псевдонім мають усі знайдені згадки однієї особи.
  4. Cosine similarity — середня косинусна подібність ембедингів речень оригіналу і тих самих
     речень після замін. Беремо лише речення, де є персональні дані.
  Підсумковий бал = NER macro F1. При рівних балах вище рішення з меншим deanonymization risk.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

TYPES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
PLACEHOLDER = re.compile(r"(ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_(\d+)")
WORD = re.compile(r"[^\W\d_](?:[\w'’ʼ-]*\w)?")
IOU_THRESHOLD = 0.5


class FormatError(ValueError):
    pass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str
    label: str  # псевдонім цілком, наприклад "ОСОБА_3"


def normalize_label(pseudonym: str) -> tuple[str, str]:
    """Повертає тип і псевдонім: " ОСОБА_3\\n" -> ("ОСОБА", "ОСОБА_3"). Латинську I замінюємо кириличною."""
    p = pseudonym.strip().upper().replace("I", "І")
    m = re.fullmatch(r"(ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)(?:_(\w+))?", p)
    if not m:
        raise FormatError(f"невідомий тип псевдоніма: {pseudonym!r}")
    return m.group(1), p if m.group(2) else f"{m.group(1)}_?"


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise FormatError(f"{path}:{n}: некоректний JSON ({e.msg})") from None
            if not isinstance(row, dict) or "id" not in row:
                raise FormatError(f"{path}:{n}: очікується об'єкт з полем 'id'")
            rows.append(row)
    return rows


def spans_from_annotations(anns: list[dict], text_len: int, doc_id) -> list[Span]:
    if not isinstance(anns, list):
        raise FormatError(f"id={doc_id}: 'annotations' має бути списком")
    out = []
    for a in anns:
        try:
            s, e, ps = int(a["start"]), int(a["end"]), a["pseudonym"]
        except (KeyError, TypeError, ValueError):
            raise FormatError(f"id={doc_id}: в анотації мають бути цілі start, end і pseudonym, отримано {a!r}") from None
        if not (0 <= s < e <= text_len):
            raise FormatError(f"id={doc_id}: фрагмент [{s}, {e}) виходить поза межі тексту (довжина {text_len})")
        try:
            t, label = normalize_label(ps)
        except FormatError as err:
            raise FormatError(f"id={doc_id}: {err}") from None
        out.append(Span(s, e, t, label))
    return out


# Кожен псевдонім у тексті відповіді тимчасово замінюємо одним символом із приватної зони Unicode,
# якого не буває у звичайних текстах.
SENTINEL_BASE = 0xF0000


def _strip(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _split_by_types(text: str, s: int, e: int, labels: list[str]) -> list[tuple[int, int]] | None:
    """Ділить text[s:e] по пробілах на стільки частин, скільки псевдонімів стоїть поруч.
    Обираємо поділ, за якого ОСОБА має близько трьох слів без цифр, а НОМЕР містить цифри."""
    from itertools import combinations

    cuts = [m.span() for m in re.finditer(r"\s+", text[s:e])]
    k = len(labels)
    if len(cuts) < k - 1 or sum(1 for _ in combinations(range(len(cuts)), k - 1)) > 20000:
        return None

    def cost(seg: str, label: str) -> float:
        t, digits, words = label.rsplit("_", 1)[0], any(c.isdigit() for c in seg), len(seg.split())
        if t == "ОСОБА":
            return 5 * digits + abs(words - 3)
        if t == "НОМЕР":
            return 5 * (not digits) + 0.5 * (words - 1)
        return 0.0

    best, best_cost = None, float("inf")
    for combo in combinations(range(len(cuts)), k - 1):
        bounds, pos = [], 0
        for c in combo:
            bounds.append((pos, cuts[c][0]))
            pos = cuts[c][1]
        bounds.append((pos, e - s))
        total = sum(cost(text[s + a:s + b], lab) for (a, b), lab in zip(bounds, labels))
        if total < best_cost:
            best, best_cost = bounds, total
    return [(s + a, s + b) for a, b in best]


def spans_from_text(original: str, pseudonymized: str, doc_id=None) -> tuple[list[Span], bool, int]:
    """Знаходить, які фрагменти original замінено псевдонімами в тексті відповіді.

    Тексти порівнюємо через difflib: спершу по словах, потім посимвольно там, де вони розходяться.
    Повертає (фрагменти, ok, ambiguous). ok=False означає, що текст змінено не лише в місцях
    псевдонімів. ambiguous — скільки разів псевдоніми стояли поруч через пробіл і межу між ними
    довелося визначати приблизно.
    """
    from difflib import SequenceMatcher

    labels: list[str] = []

    def to_sentinel(m: re.Match) -> str:
        labels.append(m.group(0))
        return chr(SENTINEL_BASE + len(labels) - 1)

    def is_sent(ch: str) -> bool:
        return ord(ch) >= SENTINEL_BASE

    def label_of(ch: str) -> str:
        return labels[ord(ch) - SENTINEL_BASE]

    pred = PLACEHOLDER.sub(to_sentinel, pseudonymized)

    spans: list[Span] = []
    ok, ambiguous = True, 0

    def emit(s: int, e: int, label: str) -> None:
        s, e = _strip(original, s, e)
        if e > s:
            spans.append(Span(s, e, label.rsplit("_", 1)[0], label))

    def handle(i1: int, i2: int, seg: str) -> None:
        """original[i1:i2] у відповіді замінено на seg."""
        nonlocal ok
        sents = [ch for ch in seg if is_sent(ch)]
        rest = "".join(ch for ch in seg if not is_sent(ch))
        if rest.strip() or (not sents and original[i1:i2].strip()):
            ok = False
        if len(sents) == 1:
            emit(i1, i2, label_of(sents[0]))
        elif len(sents) > 1:  # межу між псевдонімами визначити не вдалося
            ok = False
            emit(i1, i2, label_of(sents[0]))

    tok = re.compile(r"\w+\s*|[^\w\s]\s*|\s+")
    to = [m.span() for m in tok.finditer(original)]
    tp = [m.span() for m in tok.finditer(pred)]
    a = [original[s:e] for s, e in to]
    b = [pred[s:e] for s, e in tp]
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        ci1 = to[i1][0] if i1 < len(to) else len(original)
        ci2 = to[i2 - 1][1] if i2 > i1 else ci1
        cj1 = tp[j1][0] if j1 < len(tp) else len(pred)
        cj2 = tp[j2 - 1][1] if j2 > j1 else cj1
        # спільний початок і кінець не належать до заміни: «ОСОБА_1а» на місці «Іванова» дає «Іванов»
        while ci1 < ci2 and cj1 < cj2 and original[ci1] == pred[cj1]:
            ci1, cj1 = ci1 + 1, cj1 + 1
        while ci2 > ci1 and cj2 > cj1 and original[ci2 - 1] == pred[cj2 - 1]:
            ci2, cj2 = ci2 - 1, cj2 - 1
        seg = pred[cj1:cj2]
        sents = [ch for ch in seg if is_sent(ch)]
        only_ws = not "".join(ch for ch in seg if not is_sent(ch)).strip()
        if only_ws and len(sents) <= 1:
            handle(ci1, ci2, seg)
            continue
        if only_ws:  # кілька псевдонімів поспіль через пробіл, наприклад «ОСОБА_1 ОСОБА_2»
            split = _split_by_types(original, ci1, ci2, [label_of(ch) for ch in sents])
            if split is not None:
                ambiguous += 1
                for (s, e), ch in zip(split, sents):
                    emit(s, e, label_of(ch))
                continue
        # текст змінено не лише в місцях псевдонімів, порівнюємо посимвольно
        sub = SequenceMatcher(None, original[ci1:ci2], seg, autojunk=False)
        for t, x1, x2, y1, y2 in sub.get_opcodes():
            if t != "equal":
                handle(ci1 + x1, ci1 + x2, seg[y1:y2])
    return spans, ok, ambiguous


def iou(a: Span, b: Span) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    if inter <= 0:
        return 0.0
    return inter / (max(a.end, b.end) - min(a.start, b.start))


def match(gold: list[Span], pred: list[Span], strict: bool) -> list[tuple[int, int]]:
    """Пари (правильний, знайдений) з однаковим типом: точний збіг меж (strict) або IoU >= 0.5.
    Кожен фрагмент потрапляє не більше ніж в одну пару, спершу беремо пари з більшим IoU."""
    cand = []
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            if g.type != p.type:
                continue
            if strict:
                if g.start == p.start and g.end == p.end:
                    cand.append((1.0, i, j))
            else:
                v = iou(g, p)
                if v >= IOU_THRESHOLD:
                    cand.append((v, i, j))
    cand.sort(key=lambda x: -x[0])
    used_g, used_p, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            pairs.append((i, j))
    return pairs


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def covered(text: str, span: Span, pred: list[Span]) -> bool:
    """Згадку приховано, якщо в кожному її слові замасковано хоча б половину символів (тип не важливий).
    Пропущене слово вважається витоком, а відкрита остання літера — ні."""
    for w in WORD.finditer(text, span.start, span.end):
        ws, we = w.span()
        cov = [0] * (we - ws)
        for p in pred:
            for i in range(max(ws, p.start), min(we, p.end)):
                cov[i - ws] = 1
        if 2 * sum(cov) < len(cov):
            return False
    return True


def b_cubed(gold_lab: list[str], pred_lab: list[str]) -> tuple[float, float]:
    n = len(gold_lab)
    if n == 0:
        return 0.0, 0.0
    gc, pc, both = Counter(gold_lab), Counter(pred_lab), Counter(zip(gold_lab, pred_lab))
    p = sum(both[(g, q)] / pc[q] for g, q in zip(gold_lab, pred_lab)) / n
    r = sum(both[(g, q)] / gc[g] for g, q in zip(gold_lab, pred_lab)) / n
    return p, r


def render(original: str, spans: list[Span]) -> str:
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        if s.start < pos:
            continue
        out += [original[pos:s.start], s.label]
        pos = s.end
    out.append(original[pos:])
    return "".join(out)


SENT_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|\n+")
ABBREV = re.compile(r"(?:^|[\s(«\"])(?:м|с|смт|вул|просп|пров|пл|буд|кв|корп|обл|р|рр|ст|ч|п|пп|т|ім|див|грн|коп|тис|млн|"
                    r"н|р\.н|св|напр|ін|ін\.)\.$", re.I)


def _is_sentence_end(text: str, m: re.Match) -> bool:
    if "\n" in m.group():
        return True
    nxt = text[m.end():m.end() + 1]
    return not ABBREV.search(text[max(0, m.start() - 6):m.start()]) and (nxt.isupper() or nxt.isdigit() or nxt in "«\"(")


def sentence_pairs(original: str, gold: list[Span], pred: list[Span]) -> list[tuple[str, str]]:
    """Пари (речення оригіналу, те саме речення після замін).

    Речення ділимо тільки в оригіналі й ніколи всередині фрагмента з персональними даними.
    Беремо лише речення з персональними даними: речення без змін завжди дають подібність 1.0.
    """
    spans = gold + pred

    def inside(pos: int) -> bool:
        return any(s.start < pos < s.end for s in spans)

    cuts = [0] + [m.end() for m in SENT_BOUNDARY.finditer(original)
                  if _is_sentence_end(original, m) and not inside(m.start()) and not inside(m.end())]
    cuts.append(len(original))
    pairs = []
    for a, b in zip(cuts, cuts[1:]):
        if not original[a:b].strip() or not any(a <= s.start and s.end <= b for s in spans):
            continue
        local = [Span(s.start - a, s.end - a, s.type, s.label) for s in pred if a <= s.start and s.end <= b]
        pairs.append((original[a:b].strip(), render(original[a:b], local).strip()))
    return pairs


def cosine_similarity(pairs: list[tuple[str, str]], model_name: str) -> float | None:
    """Середня косинусна подібність ембедингів речень оригіналу і тих самих речень після замін."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Для --cosine потрібен пакет sentence-transformers: pip install sentence-transformers")
    if not pairs:
        return None
    model = SentenceTransformer(model_name, device="cpu")
    ea = model.encode([a for a, _ in pairs], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    eb = model.encode([b for _, b in pairs], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return float(sum(float((x * y).sum()) for x, y in zip(ea, eb)) / len(pairs))


def evaluate(gold_rows: list[dict], pred_rows: list[dict], cosine_model: str | None = None) -> dict:
    preds = {}
    warnings = []
    for r in pred_rows:
        key = str(r["id"])
        if key in preds:
            raise FormatError(f"id={key}: рішення трапляється у відповіді вдруге")
        preds[key] = r
    gold_ids = {str(g["id"]) for g in gold_rows}
    extra = [k for k in preds if k not in gold_ids]
    if extra:
        warnings.append(f"{len(extra)} рішень з відповіді немає в тестових даних, їх пропущено: {', '.join(extra[:5])}")

    n_ambiguous, missing = 0, []
    strict_c = {t: Counter() for t in TYPES}
    relax_c = {t: Counter() for t in TYPES}
    persons_total = persons_leaked = 0
    b3_p = b3_r = b3_n = 0.0
    cos_pairs = []

    for g in gold_rows:
        doc_id, original = str(g["id"]), g["original"]
        gold = spans_from_annotations(g["annotations"], len(original), doc_id)
        r = preds.get(doc_id)
        if r is None:
            missing.append(doc_id)
            pred = []
        elif "annotations" in r:
            pred = spans_from_annotations(r["annotations"], len(original), doc_id)
        elif "pseudonymized" in r:
            pred, ok, amb = spans_from_text(original, r["pseudonymized"], doc_id)
            if not ok:
                warnings.append(f"id={doc_id}: текст відповіді змінено не лише в місцях псевдонімів, позиції визначено приблизно")
            if amb:
                n_ambiguous += amb
        else:
            raise FormatError(f"id={doc_id}: у рядку відповіді немає ні annotations, ні pseudonymized")

        # 1. NER
        for strict, counters in ((True, strict_c), (False, relax_c)):
            pairs = match(gold, pred, strict)
            mg, mp = {i for i, _ in pairs}, {j for _, j in pairs}
            for i, s in enumerate(gold):
                counters[s.type]["tp" if i in mg else "fn"] += 1
            for j, s in enumerate(pred):
                if j not in mp:
                    counters[s.type]["fp"] += 1

        # 2. Deanonymization risk (by name)
        persons = defaultdict(list)
        for s in gold:
            if s.type == "ОСОБА":
                persons[s.label].append(s)
        for mentions in persons.values():
            persons_total += 1
            if not all(covered(original, m, pred) for m in mentions):
                persons_leaked += 1

        # 3. Consistency: B³ по згадках ОСОБА, які знайдено
        gp = [s for s in gold if s.type == "ОСОБА"]
        pp = [s for s in pred if s.type == "ОСОБА"]
        pairs = match(gp, pp, strict=False)
        if pairs:
            p, rr = b_cubed([gp[i].label for i, _ in pairs], [pp[j].label for _, j in pairs])
            b3_p += p * len(pairs)
            b3_r += rr * len(pairs)
            b3_n += len(pairs)

        if cosine_model:
            cos_pairs += sentence_pairs(original, gold, pred)

    def block(counters):
        per = {t: prf(c["tp"], c["fp"], c["fn"]) for t, c in counters.items()}
        macro = {k: sum(per[t][k] for t in TYPES) / len(TYPES) for k in ("precision", "recall", "f1")}
        return {"per_type": per, "macro": macro}

    if missing:
        warnings.insert(0, f"{len(missing)} рішень немає у відповіді, вони рахуються як рішення без персональних даних: "
                           f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}")
    bp, br = (b3_p / b3_n, b3_r / b3_n) if b3_n else (0.0, 0.0)
    relaxed = block(relax_c)
    res = {
        "final_score": round(relaxed["macro"]["f1"], 4),
        "ner_relaxed": relaxed,
        "ner_strict": block(strict_c),
        "deanonymization_risk_by_name": persons_leaked / persons_total if persons_total else 0.0,
        "persons_total": persons_total,
        "persons_leaked": persons_leaked,
        "consistency_b3": {"precision": bp, "recall": br, "f1": 2 * bp * br / (bp + br) if bp + br else 0.0},
        "n_docs": len(gold_rows),
        "text_format_ambiguous_boundaries": n_ambiguous,
        "warnings": warnings,
    }
    if cosine_model:
        res["cosine_similarity"] = cosine_similarity(cos_pairs, cosine_model)
    return res


def print_report(res: dict) -> None:
    r = res["ner_relaxed"]["macro"]
    c = res["consistency_b3"]
    print(f"Підсумковий бал                 {res['final_score']:.4f}")
    print(f"NER macro P / R                 {r['precision']:.4f} / {r['recall']:.4f}")
    print(f"NER macro F1 (strict)           {res['ner_strict']['macro']['f1']:.4f}")
    print(f"Deanonymization risk (by name)  {res['deanonymization_risk_by_name']:.4f}   "
          f"(розкрито {res['persons_leaked']} з {res['persons_total']} осіб)")
    print(f"Consistency B³ F1               {c['f1']:.4f}   (P {c['precision']:.4f}, R {c['recall']:.4f})")
    if res.get("cosine_similarity") is not None:
        print(f"Cosine similarity               {res['cosine_similarity']:.4f}")
    for name in ("ner_relaxed", "ner_strict"):
        b = res[name]
        print(f"\n{name}  ({'IoU >= 0.5' if name == 'ner_relaxed' else 'точні межі'})")
        print(f"  {'тип':<12}{'P':>7}{'R':>7}{'F1':>7}{'TP':>7}{'FP':>7}{'FN':>7}")
        for t in TYPES:
            m = b["per_type"][t]
            print(f"  {t:<12}{m['precision']:7.3f}{m['recall']:7.3f}{m['f1']:7.3f}{m['tp']:7d}{m['fp']:7d}{m['fn']:7d}")


def print_warnings(res: dict) -> None:
    out = list(res["warnings"])
    n = res.get("text_format_ambiguous_boundaries")
    if n:
        out.append(f"псевдоніми стоять поруч через пробіл ({n} випадків), межі між ними визначено приблизно; "
                   "точну оцінку дає формат з annotations")
    for w in out[:20]:
        print(w, file=sys.stderr)
    if len(out) > 20:
        print(f"… і ще {len(out) - 20}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("truth", help="файл із правильними відповідями (test/court_cases.jsonl)")
    ap.add_argument("submission", help="ваша відповідь у форматі JSON Lines")
    ap.add_argument("--json", action="store_true", help="вивести результат одним JSON")
    ap.add_argument("--cosine", action="store_true", help="порахувати cosine similarity (потрібен sentence-transformers)")
    ap.add_argument("--cosine-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    help="модель ембедингів для --cosine")
    args = ap.parse_args()
    try:
        res = evaluate(load_jsonl(args.truth), load_jsonl(args.submission), args.cosine_model if args.cosine else None)
    except FormatError as e:
        sys.exit(f"Помилка формату: {e}")
    print_warnings(res)
    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        print_report(res)


if __name__ == "__main__":
    main()
