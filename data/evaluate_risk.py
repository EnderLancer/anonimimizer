#!/usr/bin/env python3
"""Повний deanonymization risk: особа захищена, лише якщо приховано всі згадки її імені
та всі дати, номери й адреси, що стосуються саме її.

Запуск:
    python evaluate_risk.py test/court_cases.jsonl submission.jsonl
    python evaluate_risk.py test/court_cases.jsonl submission.jsonl --json       # результат одним JSON
    python evaluate_risk.py test/court_cases.jsonl submission.jsonl --details    # кожен витік окремо

Формат відповіді той самий, що в evaluate.py (annotations або pseudonymized).

Зв'язки «особа → її дані» беремо з test/person_links.json (його будує link_persons.py): для кожного
псевдоніма НОМЕР / АДРЕСА / ІНФОРМАЦІЯ там записано, яким особам документа він належить, і вид
даних (РНОКПП, паспорт, дата народження, військова частина тощо). Без цього файлу дані вважаються
нічиїми, і метрика збігається з deanonymization risk (by name) з evaluate.py.

Як рахуємо:
  - особа = псевдонім ОСОБА в документі, як в evaluate.py;
  - ідентифікатори особи = усі згадки її імені + усі згадки кожного псевдоніма даних, власником якого
    вона є. Псевдонім може належати кільком особам (спільна адреса), дані без власника (військова
    частина як відповідач, майно юрособи) в ризик не входять;
  - згадку приховано, якщо в кожному її буквено-цифровому токені замасковано щонайменше половину
    символів будь-яким псевдонімом. Це правило evaluate.py, але токеном може бути й число: WORD
    з evaluate.py не бачить жодного слова в «7955136227» і вважав би такий номер прихованим;
  - deanonymization risk = частка осіб, у яких не приховано хоч одну згадку.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402

TOKEN = re.compile(r"[^\W_]+(?:['’ʼ-][^\W_]+)*")
DATA_TYPES = ("НОМЕР", "АДРЕСА", "ІНФОРМАЦІЯ")


def covered_tokens(text: str, span: ev.Span, pred: list[ev.Span]) -> bool:
    """Як evaluate.covered, але токеном є будь-який буквено-цифровий блок, тож числа теж перевіряються."""
    for w in TOKEN.finditer(text, span.start, span.end):
        ws, we = w.span()
        cov = [0] * (we - ws)
        for p in pred:
            for i in range(max(ws, p.start), min(we, p.end)):
                cov[i - ws] = 1
        if 2 * sum(cov) < len(cov):
            return False
    return True


def load_links(path: str) -> dict:
    """Файл зв'язків: {"meta": {...}, "docs": {id: {"data": {псевдонім: {"owners": [...], "kind": ...}}}}}.
    Псевдоніми нормалізуємо так само, як evaluate.py нормалізує gold."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    docs = {}
    for doc_id, d in raw.get("docs", {}).items():
        data = {}
        for pseud, info in d.get("data", {}).items():
            _, label = ev.normalize_label(pseud)
            data[label] = {"owners": [ev.normalize_label(o)[1] for o in info.get("owners", [])],
                           "kind": info.get("kind")}
        docs[str(doc_id)] = data
    return {"meta": raw.get("meta", {}), "docs": docs}


def pred_spans(r: dict | None, original: str, doc_id: str, warnings: list[str]) -> list[ev.Span]:
    """Фрагменти відповіді так само, як їх читає evaluate.evaluate."""
    if r is None:
        return []
    if "annotations" in r:
        return ev.spans_from_annotations(r["annotations"], len(original), doc_id)
    if "pseudonymized" in r:
        pred, ok, _ = ev.spans_from_text(original, r["pseudonymized"], doc_id)
        if not ok:
            warnings.append(f"id={doc_id}: текст відповіді змінено не лише в місцях псевдонімів, позиції визначено приблизно")
        return pred
    raise ev.FormatError(f"id={doc_id}: у рядку відповіді немає ні annotations, ні pseudonymized")


def evaluate_risk(gold_rows: list[dict], pred_rows: list[dict], links: dict | None, details: bool = False) -> dict:
    preds = {}
    warnings = []
    for r in pred_rows:
        key = str(r["id"])
        if key in preds:
            raise ev.FormatError(f"id={key}: рішення трапляється у відповіді вдруге")
        preds[key] = r
    missing = []

    persons_total = persons_leaked = leaked_by_name = leaked_by_data_only = persons_with_data = 0
    official_leaked = 0  # за evaluate.covered, має збігатися з evaluate.py
    name_rule_diff = 0
    leaked_mentions = {t: 0 for t in ev.TYPES}
    leaked_kinds = Counter()
    pseud = Counter()  # псевдоніми даних: total / linked / unlinked / missing
    docs_without_links = []
    unknown_owners = 0
    leaks = []

    for g in gold_rows:
        doc_id, original = str(g["id"]), g["original"]
        gold = ev.spans_from_annotations(g["annotations"], len(original), doc_id)
        if doc_id not in preds:
            missing.append(doc_id)
        pred = pred_spans(preds.get(doc_id), original, doc_id, warnings)

        persons: dict[str, list[ev.Span]] = defaultdict(list)
        data: dict[str, list[ev.Span]] = defaultdict(list)
        for s in gold:
            (persons if s.type == "ОСОБА" else data)[s.label].append(s)

        doc_links = links["docs"].get(doc_id) if links else None
        if links and doc_links is None and data:
            docs_without_links.append(doc_id)
        owners_of: dict[str, set[str]] = {}
        kind_of: dict[str, str | None] = {}
        for lab in data:
            pseud["total"] += 1
            info = (doc_links or {}).get(lab)
            if info is None:
                owners_of[lab] = set()
                kind_of[lab] = None
                pseud["missing" if doc_links is not None else "unlinked"] += 1
                continue
            owners = set(info["owners"])
            unknown_owners += len(owners - persons.keys())
            owners_of[lab] = owners & persons.keys()
            kind_of[lab] = info.get("kind")
            pseud["linked" if owners_of[lab] else "unlinked"] += 1

        hidden = {s: covered_tokens(original, s, pred) for s in gold}
        official = {s: ev.covered(original, s, pred) for s in gold if s.type == "ОСОБА"}  # правило evaluate.py

        for p, mentions in persons.items():
            persons_total += 1
            name_leak = [m for m in mentions if not hidden[m]]
            if not all(official[m] for m in mentions):
                official_leaked += 1
            owned = [lab for lab, owners in owners_of.items() if p in owners]
            if owned:
                persons_with_data += 1
            data_leak = [(lab, m) for lab in owned for m in data[lab] if not hidden[m]]
            if not name_leak and not data_leak:
                continue
            persons_leaked += 1
            if name_leak:
                leaked_by_name += 1
            else:
                leaked_by_data_only += 1
            if details:
                for m in name_leak:
                    leaks.append({"id": doc_id, "person": p, "pseudonym": p, "type": "ОСОБА", "kind": None,
                                  "entity": original[m.start:m.end], "start": m.start, "end": m.end})
                for lab, m in data_leak:
                    leaks.append({"id": doc_id, "person": p, "pseudonym": lab, "type": m.type, "kind": kind_of[lab],
                                  "entity": original[m.start:m.end], "start": m.start, "end": m.end})

        # витоки згадок рахуємо один раз, навіть якщо власників кілька
        for p, mentions in persons.items():
            for m in mentions:
                if not hidden[m]:
                    leaked_mentions["ОСОБА"] += 1
                if hidden[m] != official[m]:
                    name_rule_diff += 1
        for lab, mentions in data.items():
            if not owners_of[lab]:
                continue
            for m in mentions:
                if not hidden[m]:
                    leaked_mentions[m.type] += 1
                    leaked_kinds[kind_of[lab] or "?"] += 1

    if missing:
        warnings.insert(0, f"{len(missing)} рішень немає у відповіді, вони рахуються як рішення без персональних даних: "
                           f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}")
    if links is None:
        warnings.append("файл зв'язків не задано або не знайдено: дані вважаються нічиїми, ризик рахується лише за іменами")
    if docs_without_links:
        warnings.append(f"{len(docs_without_links)} документів з даними немає у файлі зв'язків: {', '.join(docs_without_links[:5])}")
    if pseud["missing"]:
        warnings.append(f"{pseud['missing']} псевдонімів даних немає у файлі зв'язків, їх вважаємо нічиїми")
    if unknown_owners:
        warnings.append(f"{unknown_owners} власників у файлі зв'язків немає серед осіб документа, їх пропущено")
    if name_rule_diff:
        warnings.append(f"{name_rule_diff} згадок імені покрито по-різному за правилом evaluate.py і за токенами з цифрами")

    res = {
        "deanonymization_risk": persons_leaked / persons_total if persons_total else 0.0,
        "deanonymization_risk_by_name": official_leaked / persons_total if persons_total else 0.0,
        "persons_total": persons_total,
        "persons_leaked": persons_leaked,
        "persons_leaked_by_name": leaked_by_name,
        "persons_leaked_by_data_only": leaked_by_data_only,
        "persons_with_data": persons_with_data,
        "leaked_mentions_by_type": leaked_mentions,
        "leaked_by_kind": dict(leaked_kinds.most_common()),
        "data_pseudonyms": {k: pseud[k] for k in ("total", "linked", "unlinked", "missing")},
        "docs_without_links": len(docs_without_links),
        "links_meta": (links or {}).get("meta", {}),
        "n_docs": len(gold_rows),
        "warnings": warnings,
    }
    if details:
        res["leaks"] = leaks
    return res


def print_report(res: dict) -> None:
    n = res["persons_total"]
    print(f"Deanonymization risk (повний)   {res['deanonymization_risk']:.4f}   "
          f"(розкрито {res['persons_leaked']} з {n} осіб)")
    print(f"  через ім'я                    {res['deanonymization_risk_by_name']:.4f}   "
          f"({res['persons_leaked_by_name']} осіб, як в evaluate.py)")
    print(f"  лише через дані               {res['persons_leaked_by_data_only'] / n if n else 0.0:.4f}   "
          f"({res['persons_leaked_by_data_only']} осіб)")
    print(f"Осіб з прив'язаними даними      {res['persons_with_data']} з {n}")
    lm = res["leaked_mentions_by_type"]
    print("Відкритих згадок за типом       " + "  ".join(f"{t} {lm[t]}" for t in ev.TYPES))
    if res["leaked_by_kind"]:
        print("Відкритих даних за видом        " + ", ".join(f"{k} {v}" for k, v in res["leaked_by_kind"].items()))
    d = res["data_pseudonyms"]
    print(f"Зв'язки                         псевдонімів даних {d['total']}: з власником {d['linked']}, "
          f"без власника {d['unlinked']}, без запису {d['missing']}; документів без зв'язків {res['docs_without_links']}")
    for leak in res.get("leaks", []):
        print(f"  {leak['id']}  {leak['person']:<9} {leak['pseudonym']:<13} {leak['kind'] or '':<18} "
              f"[{leak['start']}, {leak['end']})  {leak['entity']!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("truth", help="файл із правильними відповідями (test/court_cases.jsonl)")
    ap.add_argument("submission", help="ваша відповідь у форматі JSON Lines")
    ap.add_argument("--links", help="файл зв'язків особа → дані (за замовчуванням person_links.json поруч із truth)")
    ap.add_argument("--json", action="store_true", help="вивести результат одним JSON")
    ap.add_argument("--details", action="store_true", help="перелічити кожну відкриту згадку")
    args = ap.parse_args()
    links_path = args.links or os.path.join(os.path.dirname(os.path.abspath(args.truth)), "person_links.json")
    links = load_links(links_path) if os.path.exists(links_path) else None
    if args.links and links is None:
        sys.exit(f"файл зв'язків не знайдено: {args.links}")
    try:
        res = evaluate_risk(ev.load_jsonl(args.truth), ev.load_jsonl(args.submission), links, args.details)
    except ev.FormatError as e:
        sys.exit(f"Помилка формату: {e}")
    for w in res["warnings"][:20]:
        print(w, file=sys.stderr)
    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        print_report(res)


if __name__ == "__main__":
    main()
