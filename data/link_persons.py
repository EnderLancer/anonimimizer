#!/usr/bin/env python3
"""Будує зв'язки «особа → її дані» для тестового набору: test/person_links.json.

Для кожного документа Claude читає текст реєстру (`pseudonymized`, у ньому лише плейсхолдери
ОСОБА_1, НОМЕР_2 …, жодних реальних даних) і для кожного псевдоніма НОМЕР / АДРЕСА / ІНФОРМАЦІЯ
каже, яким особам документа він належить і який це вид даних. Відповідь прив'язана до псевдонімів,
а не до позицій у тексті, тож файл переживає перевирівнювання тесту.

Два способи отримати відповіді:
  1. Без API, агентами Claude Code (так зроблено файл у репо):
        python link_persons.py --export tasks/       # tasks/INSTRUCTIONS.md + tasks/<id>.md на документ
        … агенти читають tasks/<id>.md і пишуть answers/<id>.json: {"links": [...]}
        python link_persons.py --import answers/     # перевірка і злиття у test/person_links.json
  2. Через API (потрібні пакет anthropic і ключ в ANTHROPIC_API_KEY):
        python link_persons.py                       # увесь тест
        python link_persons.py --limit 5             # перші 5 документів, подивитися відповіді
        python link_persons.py --only-missing        # добрати документи, яких ще немає у файлі
        python link_persons.py --ids 116175256 116268150
Перевірка без API:
        python link_persons.py --check               # порівняти файл з евристикою за відстанню

Файл: {"meta": {model, source, created, updated, test_sha256},
       "docs": {id: {"persons": [...], "data": {"НОМЕР_1": {"owners": ["ОСОБА_1"], "kind": "rnokpp"}, ...}}}}
Псевдонім без відповіді записується з owners [] і kind null, щоб evaluate_risk.py бачив його як нічий,
а --only-missing не перепитував документ.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402

DEFAULT_MODEL = "claude-opus-5-5"
KINDS = ("rnokpp", "passport", "phone", "iban", "plate", "military_unit", "dob", "death", "email",
         "org", "residence_address", "property_address", "other")

SYSTEM = """Ти розмічаєш судові рішення з Єдиного державного реєстру судових рішень України.
У тексті персональні дані вже замінено плейсхолдерами: ОСОБА_n — фізична особа, НОМЕР_n — номер
(РНОКПП, паспорт, телефон, IBAN, номерний знак, номер військової частини тощо), АДРЕСА_n — адреса,
ІНФОРМАЦІЯ_n — дата народження чи смерті, емейл, назва військової частини чи іншої установи.
Один і той самий плейсхолдер завжди означає одну й ту саму сутність.

Для кожного плейсхолдера НОМЕР / АДРЕСА / ІНФОРМАЦІЯ визнач, яким особам (ОСОБА_n) він належить,
тобто розкриття цих даних допоможе впізнати саме цю особу. Правила:
- Реквізити поруч з іменем належать цій особі: «ОСОБА_1 (АДРЕСА_1, РНОКПП НОМЕР_1)»,
  «ОСОБА_3, АДРЕСА_2, паспорт НОМЕР_4», «ОСОБА_2, ІНФОРМАЦІЯ_1» (дата народження),
  «ОСОБА_5, який помер ІНФОРМАЦІЯ_3», «ОСОБА_1 проживає за адресою АДРЕСА_1».
- «ОСОБА_2 (ОСОБА_2)» або «ОСОБА_4 (ОСОБА_7)» — це одна людина (ім'я та його варіант),
  дані поруч належать їй.
- Номер або назва військової частини (НОМЕР чи ІНФОРМАЦІЯ) належить військовослужбовцю, про
  якого йде справа: він там служить, звільнявся звідти, отримує від неї виплати. Це так і тоді,
  коли частина є відповідачем: «за позовом ОСОБА_1 до військової частини НОМЕР_1» → НОМЕР_1 належить
  ОСОБА_1. Якщо жодна фізична особа з частиною не пов'язана, власника немає.
- Дані можуть належати кільком особам: спільна квартира, спільне місце служби, дата народження,
  на яку посилаються для двох дітей. Перелічи всіх.
- Без власника: адреса чи реквізити юридичної особи, суду, нотаріуса як установи; майно, яке
  належить юрособі; організація як сторона, коли її не пов'язано з конкретною фізичною особою;
  кадастрові номери, номери справ і проваджень, якщо вони трапляються. Порожній список owners.
- Дитина: дата народження дитини належить дитині (ОСОБА_n дитини), а не батькам, навіть якщо вона
  стоїть після імені батьків у реченні «… син ОСОБА_4, ІНФОРМАЦІЯ_1».
- Якщо адреса стосується майна, що належить особі (квартира, земельна ділянка, будинок), її власник —
  ця особа; kind = property_address. Адреса проживання чи реєстрації — residence_address.

kind: rnokpp (ІПН / РНОКПП / ідентифікаційний номер), passport (паспорт, ID-картка, УНЗР), phone,
iban (рахунок, картка), plate (номерний знак, VIN), military_unit (номер або назва військової
частини), dob (дата народження), death (дата смерті), email, org (назва установи чи організації,
не військова частина), residence_address, property_address, other.

Поверни по одному запису для кожного плейсхолдера НОМЕР / АДРЕСА / ІНФОРМАЦІЯ з переліку, навіть
якщо власника немає. owners — лише плейсхолдери ОСОБА_n з переліку осіб документа."""

SCHEMA = {
    "type": "object",
    "properties": {
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pseudonym": {"type": "string"},
                    "owners": {"type": "array", "items": {"type": "string"}},
                    "kind": {"type": "string", "enum": list(KINDS)},
                },
                "required": ["pseudonym", "owners", "kind"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["links"],
    "additionalProperties": False,
}

ANSWER_FORMAT = """Відповідь — JSON-об'єкт {"links": [{"pseudonym": "НОМЕР_1", "owners": ["ОСОБА_1"], "kind": "rnokpp"}, ...]}:
по одному запису на кожен плейсхолдер даних із переліку, owners лише з переліку осіб, kind з переліку вище."""


def pseudonyms(row: dict) -> tuple[list[str], list[str]]:
    """Особи й псевдоніми даних документа в порядку першої появи в тексті реєстру."""
    seen = []
    for m in ev.PLACEHOLDER.finditer(row["pseudonymized"]):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    for a in row["annotations"]:  # на випадок, якщо анотація є, а плейсхолдера в тексті немає
        label = ev.normalize_label(a["pseudonym"])[1]
        if label not in seen:
            seen.append(label)
    persons = [p for p in seen if p.startswith("ОСОБА_")]
    data = [p for p in seen if not p.startswith("ОСОБА_")]
    return persons, data


def user_message(row: dict, persons: list[str], data: list[str]) -> str:
    return (f"Особи документа: {', '.join(persons) or '(немає)'}\n"
            f"Плейсхолдери даних, для яких потрібна відповідь: {', '.join(data)}\n\n"
            f"Текст рішення:\n{row['pseudonymized']}")


def validate(doc_id: str, links: list[dict], persons: list[str], data: list[str], warnings: list[str]) -> dict:
    """Відповідь Claude → {псевдонім: {"owners", "kind"}} лише з псевдонімами й особами документа."""
    out = {}
    allowed = set(persons)
    for item in links:
        try:
            _, label = ev.normalize_label(str(item["pseudonym"]))
        except (ev.FormatError, KeyError, TypeError):
            warnings.append(f"id={doc_id}: невідомий псевдонім {item.get('pseudonym') if isinstance(item, dict) else item!r} у відповіді, пропущено")
            continue
        if label not in data:
            warnings.append(f"id={doc_id}: {label} немає серед даних документа, пропущено")
            continue
        owners = []
        for o in item.get("owners") or []:
            try:
                _, ol = ev.normalize_label(str(o))
            except ev.FormatError:
                ol = o
            if ol not in allowed:
                warnings.append(f"id={doc_id}: власник {o!r} для {label} не є особою документа, пропущено")
                continue
            if ol not in owners:
                owners.append(ol)
        kind = item.get("kind")
        if kind not in KINDS:
            warnings.append(f"id={doc_id}: невідомий kind {kind!r} для {label}, записано other")
            kind = "other"
        out[label] = {"owners": owners, "kind": kind}
    for label in data:
        if label not in out:
            warnings.append(f"id={doc_id}: немає відповіді для {label}, записано як нічий")
            out[label] = {"owners": [], "kind": None}
    return out


def link_document(client, model: str, effort: str, row: dict, persons: list[str], data: list[str],
                  warnings: list[str]) -> dict | None:
    """Один запит до Claude через API. None — документ не вдалося розмітити (добираємо --only-missing)."""
    import anthropic

    doc_id = str(row["id"])
    try:
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_message(row, persons, data)}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except anthropic.RateLimitError as e:
        warnings.append(f"id={doc_id}: rate limit ({e.message}), пропущено")
        return None
    except anthropic.APIStatusError as e:
        warnings.append(f"id={doc_id}: помилка API {e.status_code}: {e.message}, пропущено")
        return None
    except anthropic.APIConnectionError as e:
        warnings.append(f"id={doc_id}: немає з'єднання ({e}), пропущено")
        return None
    if response.stop_reason != "end_turn":
        warnings.append(f"id={doc_id}: stop_reason={response.stop_reason}, пропущено")
        return None
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        links = json.loads(text)["links"]
    except (ValueError, KeyError, TypeError):
        warnings.append(f"id={doc_id}: відповідь не є JSON за схемою, пропущено")
        return None
    usage = response.usage
    return {"persons": persons, "data": validate(doc_id, links, persons, data, warnings),
            "usage": {"input": usage.input_tokens, "output": usage.output_tokens}}


def export_tasks(rows: list[dict], out_dir: str) -> int:
    """Завдання для агентів: INSTRUCTIONS.md з правилами і по файлу на документ із даними."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "INSTRUCTIONS.md"), "w", encoding="utf-8") as f:
        f.write(SYSTEM + "\n\n" + ANSWER_FORMAT + "\n")
    n = 0
    for row in rows:
        persons, data = pseudonyms(row)
        if not data:
            continue
        with open(os.path.join(out_dir, f"{row['id']}.md"), "w", encoding="utf-8") as f:
            f.write(user_message(row, persons, data) + "\n")
        n += 1
    return n


def import_answers(rows: list[dict], in_dir: str, links: dict, warnings: list[str]) -> tuple[int, list[str]]:
    """answers/<id>.json → links["docs"]. Повертає кількість злитих і id без відповіді."""
    done, missing = 0, []
    for row in rows:
        doc_id = str(row["id"])
        persons, data = pseudonyms(row)
        if not data:
            links["docs"][doc_id] = {"persons": persons, "data": {}}
            continue
        path = os.path.join(in_dir, f"{doc_id}.json")
        if not os.path.exists(path):
            missing.append(doc_id)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                answer = json.load(f)
            items = answer["links"] if isinstance(answer, dict) else answer
            if not isinstance(items, list):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            warnings.append(f"id={doc_id}: {path} не є JSON зі списком links, пропущено")
            missing.append(doc_id)
            continue
        links["docs"][doc_id] = {"persons": persons, "data": validate(doc_id, items, persons, data, warnings)}
        done += 1
    return done, missing


def heuristic(row: dict) -> dict[str, set[str]]:
    """Евристика за відстанню з linking_viz/build.py, перенесена на текст реєстру: кожна згадка даних
    належить найближчій попередній особі в межах 150 символів без розриву абзацу, інакше найближчій
    наступній у межах 60. Власники псевдоніма — об'єднання по згадках. Лише для --check."""
    text = row["pseudonymized"]
    ments = [(m.start(), m.end(), m.group(0)) for m in ev.PLACEHOLDER.finditer(text)]
    persons = [m for m in ments if m[2].startswith("ОСОБА_")]
    owners = defaultdict(set)
    for s, e, label in ments:
        if label.startswith("ОСОБА_"):
            continue
        owners.setdefault(label, set())
        prev = [p for p in persons if p[1] <= s and s - p[1] <= 150 and "\n" not in text[p[1]:s]]
        nxt = [p for p in persons if p[0] >= e and p[0] - e <= 60 and "\n" not in text[e:p[0]]]
        if prev:
            owners[label].add(max(prev, key=lambda p: p[1])[2])
        elif nxt:
            owners[label].add(min(nxt, key=lambda p: p[0])[2])
    return owners


def context(row: dict, label: str, width: int = 70) -> str:
    text = row["pseudonymized"]
    m = re.search(re.escape(label) + r"(?!\d)", text)
    if not m:
        return ""
    s, e = max(0, m.start() - width), min(len(text), m.end() + width)
    return re.sub(r"\s+", " ", text[s:e])


def check(rows: list[dict], links: dict) -> None:
    agree, differ = Counter(), Counter()
    for row in rows:
        doc = links["docs"].get(str(row["id"]))
        if doc is None:
            continue
        heur = heuristic(row)
        for label, info in doc["data"].items():
            t = label.rsplit("_", 1)[0]
            llm, h = set(info["owners"]), heur.get(label, set())
            if llm == h:
                agree[t] += 1
                continue
            differ[t] += 1
            print(f"{row['id']}  {label:<13} {info.get('kind') or '?':<18} "
                  f"llm={','.join(sorted(llm)) or '-':<20} евристика={','.join(sorted(h)) or '-'}")
            print(f"    …{context(row, label)}…")
    for t in ev.TYPES[1:]:
        n = agree[t] + differ[t]
        if n:
            print(f"{t}: збіг {agree[t]} з {n} ({agree[t] / n:.0%})")


def save(links: dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(links, f, ensure_ascii=False, indent=1)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", nargs="?", default=os.path.join(here, "test", "court_cases.jsonl"))
    ap.add_argument("--out", help="файл зв'язків (за замовчуванням person_links.json поруч із тестом)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="модель для API або підпис у meta при --import")
    ap.add_argument("--effort", default="high", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--limit", type=int, help="лише перші N документів")
    ap.add_argument("--ids", nargs="*", help="лише ці документи")
    ap.add_argument("--only-missing", action="store_true", help="пропустити документи, що вже є у файлі")
    ap.add_argument("--export", metavar="DIR", help="записати завдання для агентів і вийти")
    ap.add_argument("--import", dest="import_dir", metavar="DIR", help="злити відповіді агентів DIR/<id>.json і вийти")
    ap.add_argument("--check", action="store_true", help="без API: порівняти файл з евристикою за відстанню")
    args = ap.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.test)), "person_links.json")
    rows = ev.load_jsonl(args.test)
    if args.ids:
        want = set(args.ids)
        rows = [r for r in rows if str(r["id"]) in want]
    if args.limit:
        rows = rows[:args.limit]

    links = {"meta": {}, "docs": {}}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            links = json.load(f)
    if args.check:
        check(rows, links)
        return
    if args.export:
        n = export_tasks(rows, args.export)
        print(f"Записано {n} завдань у {args.export} (документи без даних пропущено)", file=sys.stderr)
        return

    with open(args.test, "rb") as f:
        test_sha = hashlib.sha256(f.read()).hexdigest()
    meta = links.setdefault("meta", {})
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    meta.update({"model": args.model, "source": "import" if args.import_dir else "api", "test_sha256": test_sha,
                 "updated": now})
    meta.setdefault("created", now)
    warnings: list[str] = []

    if args.import_dir:
        if args.only_missing:
            rows = [r for r in rows if str(r["id"]) not in links["docs"]]
        done, missing = import_answers(rows, args.import_dir, links, warnings)
        save(links, out_path)
        for w in warnings:
            print(w, file=sys.stderr)
        print(f"Злито {done}, без відповіді {len(missing)}{': ' + ' '.join(missing[:10]) if missing else ''}; "
              f"у файлі {len(links['docs'])} документів: {out_path}", file=sys.stderr)
        return

    try:
        import anthropic
    except ImportError:
        sys.exit("Потрібен пакет anthropic: pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY не задано, клієнт спробує інші джерела облікових даних", file=sys.stderr)
    client = anthropic.Anthropic()
    if args.effort:
        meta["effort"] = args.effort

    tokens = Counter()
    done = skipped = 0
    for i, row in enumerate(rows, 1):
        doc_id = str(row["id"])
        persons, data = pseudonyms(row)
        if not data:
            links["docs"][doc_id] = {"persons": persons, "data": {}}
            continue
        if args.only_missing and doc_id in links["docs"]:
            continue
        res = link_document(client, args.model, args.effort, row, persons, data, warnings)
        if res is None:
            skipped += 1
        else:
            tokens.update(res.pop("usage"))
            links["docs"][doc_id] = res
            done += 1
        save(links, out_path)  # зберігаємо після кожного документа
        print(f"[{i}/{len(rows)}] {doc_id}: осіб {len(persons)}, даних {len(data)}"
              + ("" if res else " — пропущено"), file=sys.stderr)
    for w in warnings:
        print(w, file=sys.stderr)
    print(f"Розмічено {done}, пропущено {skipped}, у файлі {len(links['docs'])} документів; "
          f"токенів на вході {tokens['input']}, на виході {tokens['output']}: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
