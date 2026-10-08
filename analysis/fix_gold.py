"""Виправлені копії лише тих еталонних рішень, де розмітка помилкова.

python analysis/fix_gold.py [data/test/court_cases.jsonl] [data/test/court_cases_fixed.jsonl]

Кожне виправлене рішення має ті самі поля, що й оригінал (annotations, annotations_by_pseudonym і pseudonymized
перебудовано з виправлених анотацій), і додаткове поле `corrections`: що було, що стало, чому це помилка
і на яку норму це спирається. Тексти норм: analysis/legal/.

Нові псевдоніми отримують наступний вільний номер типу в документі, щоб не змінювати решту позначок
(метрика зважає лише на групування згадок, а не на самі номери).

Свідомо НЕ виправлено (це послідовні конвенції набору даних, а не поодинокі помилки; див. LABELING_RULES.md §3):
маскування суддів і виконавців, «А1234» як НОМЕР, e-mail як ІНФОРМАЦІЯ, держоргани/компанії як ІНФОРМАЦІЯ
в частині документів, відкинута крапка ініціала в кінці речення, незамасковані назви справ ЄСПЛ.
"""
import json
import re
import sys
from collections import defaultdict

LAW = "Закон України «Про доступ до судових рішень» № 3262-IV, ст. 7 ч. 1"
COD = "Кодифікатор відомостей, що не можуть бути розголошені в текстах судових рішень (наказ ДСА від 13.06.2006 № 59)"
P2 = f"{COD}, п. 2 (імена фізичних осіб → «ОСОБА_N»)"
P3 = f"{COD}, п. 3 (адреса місця проживання або перебування → «АДРЕСА_N»)"
P4 = f"{COD}, п. 4 (номер документа, ідентифікаційний номер → «НОМЕР_N»)"
P5 = f"{COD}, п. 5 (інша інформація, що дає можливість ідентифікувати особу → «ІНФОРМАЦІЯ_N»)"
NUM = f"{COD}, пп. 2–5: цифрове позначення «відповідає черговості появи цієї відомості у тексті», тобто одна відомість — один номер"

# Кожен запис: doc, дія, що знайти (текст і, за потреби, номер входження), на що замінити, пояснення, норма.
# find: (entity, label) для наявної анотації; occ=None означає «усі входження».
FIXES = [
    # ---------- неправильний тип ----------
    dict(doc="116186753", action="relabel", find=("Соловй", "ІНФОРМАЦІЯ_1"), to_type="ОСОБА",
         why="ПІБ фізичної особи (позивачки) розмічено як ІНФОРМАЦІЯ. Ім'я фізичної особи має тип ОСОБА.",
         ref=[P2, f"{LAW}, п. 5"]),
    dict(doc="116158320", action="relabel", find=("Кушнір", "ІНФОРМАЦІЯ_2"), to_type="ОСОБА",
         why="ПІБ фізичної особи (скаржниці) розмічено як ІНФОРМАЦІЯ. Ім'я фізичної особи має тип ОСОБА.",
         ref=[P2, f"{LAW}, п. 5"]),
    dict(doc="116234614", action="relabel", find=("м. Київ, вул.", "ІНФОРМАЦІЯ_"), to_type="АДРЕСА",
         why="Адресу місця проживання малолітньої дитини і матері розмічено як ІНФОРМАЦІЯ. Адреса проживання має тип АДРЕСА.",
         ref=[P3, f"{LAW}, п. 1 («місце проживання або перебування фізичних осіб із зазначенням адреси»)"]),
    dict(doc="116211933", action="reboundary_relabel", find=("Військов", "ІНФОРМАЦІЯ_"), sub=r"[АA]\d{4}", to_type="НОМЕР",
         why="«Військова частина А1234» розмічено цілком як ІНФОРМАЦІЯ. Персональним є лише умовне найменування «А1234»; "
             "у всьому наборі воно 314 разів розмічене як НОМЕР без слів «військова частина» (і в цьому ж документі нижче).",
         ref=[P4, "Узгодженість набору даних: 314 із 318 згадок номера військової частини мають тип НОМЕР. "
                  "Закон не вимагає маскувати реквізити військової частини (ст. 7 ч. 1 стосується фізичних осіб); "
                  "виправлення лише вирівнює тип за конвенцією набору"]),
    dict(doc="116133531", action="reboundary_relabel", find=("Військов", "ІНФОРМАЦІЯ_"), sub=r"[АA]\d{4}", to_type="НОМЕР",
         why="Те саме, що в 116211933: «Військова частина А1234» цілком як ІНФОРМАЦІЯ замість номера «А1234» як НОМЕР.",
         ref=[P4, "Узгодженість набору даних (314 із 318)"]),
    # ---------- неправильні межі ----------
    dict(doc="116234786", action="resplit", spans=[("15 листопада 2020 року", "ІНФОРМАЦІЯ"), ("Шевченко Тамари Миколаївни", "ОСОБА")],
         replaces=[("15", "ІНФОРМАЦІЯ_"), ("листопада 2020 року Шевченко Тамари Миколаївни", "ОСОБА_")],
         why="Межу поставлено посеред дати: «15» — ІНФОРМАЦІЯ, а «листопада 2020 року» приклеєно до ПІБ. "
             "Дата смерті «15 листопада 2020 року» — ІНФОРМАЦІЯ, ім'я «Шевченко Тамари Миколаївни» — ОСОБА без дати.",
         ref=[P2, P5]),
    dict(doc="116234813", action="resplit", spans=[("Петренко Аліси Володимирівни", "ОСОБА"), ("DF987654", "НОМЕР")],
         replaces=[("Петренко", "ОСОБА_"), ("Аліси Володимирівни DF987654", "НОМЕР_")], near="трудової книжки",
         why="Межу між ПІБ і номером трудової книжки зсунуто: «Петренко» — ОСОБА, «Аліси Володимирівни DF987654» — НОМЕР. "
             "Ім'я та по батькові — частина імені особи, номер документа — окремо.",
         ref=[P2, P4]),
    dict(doc="116177353", action="resplit", spans=[("Шидерової Н. С.", "ОСОБА"), ("y.pryimak@notary.ua", "ІНФОРМАЦІЯ")],
         replaces=[("Шидерової", "ОСОБА_"), ("Н. С. y.pryimak@notary.ua", "ІНФОРМАЦІЯ_")], near="адвоката",
         why="Ініціали адвоката «Н. С.» приклеєно до адреси e-mail. Ініціали — частина імені (ОСОБА), "
             "e-mail — окрема відомість (тип ІНФОРМАЦІЯ залишено за конвенцією набору; Кодифікатор відносить e-mail до НОМЕР).",
         ref=[P2, P4, f"{LAW}, п. 1 («адреси електронної пошти»)"]),
    dict(doc="116234574", action="trim", find=("адвокат Мастістий І. А.", "ОСОБА_"), new_text="Мастістий І. А.",
         why="До імені включено процесуальну роль «адвокат». Замінюється лише ім'я; через це слово «адвокат» "
             "рахувалося як незамасковане слово імені (витік за метрикою deanonymization risk).",
         ref=[P2 + ': приклад «"Іванов Іван Іванович" замінюється на: "ОСОБА_1"» — лише ім\'я']),
    dict(doc="116177397", action="trim", find=(", розташованого за адресою:", "АДРЕСА_"), new_text_re=r"\d{5},.*",
         why="До адреси включено «, розташованого за адресою: ». Замінюється лише сама адреса.",
         ref=[P3 + ': приклад «"Вул. Київська, кв. 1, буд. 1, м. Київ, 01001" замінюється на: "АДРЕСА_1"»']),
    dict(doc="116234753", action="trim", find=("місто Одеса, вулиця Троїцька", "АДРЕСА_"), strip=",",
         why="До адреси включено кому, що належить реченню.", ref=[P3]),
    dict(doc="116258571", action="trim", find=("Мельник Ірини Борисівни,", "ОСОБА_"), strip=",",
         why="До імені включено кому, що належить реченню.", ref=[P2]),
    dict(doc="116258571", action="trim", find=("Ганжу Петра Романовича.", "ОСОБА_"), strip=".",
         why="До повного імені включено крапку в кінці речення (це не крапка ініціала).", ref=[P2]),
    dict(doc="116258635", action="trim", find=("Ярошенка Максима Олеговича.", "ОСОБА_"), strip=".",
         why="До повного імені включено крапку в кінці речення (це не крапка ініціала).", ref=[P2]),
    dict(doc="116205945", action="extend", find=("Підлубного Е. С", "ОСОБА_"), add=".",
         why="Відкинуто крапку ініціала «С.» посеред речення («Підлубного Е. С. та витребування…»). "
             "Набір відкидає крапку ініціала лише в кінці речення; тут речення триває.",
         ref=[P2 + ": ініціали — частина імені"]),
    # ---------- нумерація ----------
    dict(doc="116234778", action="merge", find=("Гварамадзе Леван Зурабович", "ОСОБА_3"), into="ОСОБА_2",
         why="Одна особа (відповідач Гварамадзе Леван Зурабович) має два псевдоніми: ОСОБА_2 у родовому відмінку і ОСОБА_3 у називному.",
         ref=[NUM]),
    dict(doc="116205945", action="merge", find=("Підлуб", "ОСОБА_7"), into="ОСОБА_6",
         why="Одна особа (виконувач обов'язків голови Дніпровської міської ради Підлубний Е. С.) має два псевдоніми: ОСОБА_6 і ОСОБА_7. "
             "Примітка: це посадова особа, що діє в межах повноважень, тож за п. 2 Кодифікатора її ім'я взагалі не мало б маскуватися; "
             "маскування залишено за конвенцією набору, виправлено лише нумерацію.",
         ref=[NUM, P2 + " (виняток для посадових осіб)"]),
    dict(doc="116206025", action="split", find=("Тимошенко Максим Олександрович", "ОСОБА_10"), to_type="ОСОБА",
         why="Дві різні особи в тексті («ФОП Тимченко Олександр Миколайович» і «ФОП Тимошенко Максим Олександрович») мають один псевдонім ОСОБА_10. "
             "Ймовірно, у джерелі це одна особа, але синтетичне поле original підставило два різні імена; розмітка має відповідати тексту.",
         ref=[NUM]),
    # ---------- пропущені відомості ----------
    dict(doc="116231420", action="add", text="Туйчієвій Одінахон", to_type="ОСОБА",
         why="Ім'я фізичної особи-нерезидента, якій вручається документ, не замасковано, хоча її код у тому ж реченні замасковано як НОМЕР.",
         ref=[f"{LAW}, п. 5", P2, f"{COD}, додаток 1, № 59 «Інша фізична особа, відомості щодо якої містяться у тексті судового рішення»"]),
    dict(doc="116258650", action="add", text="770867935101", to_type="НОМЕР",
         why="Реєстраційний номер об'єкта нерухомого майна спадкодавиці (4 згадки) не замасковано. За ним у Державному реєстрі речових прав "
             "встановлюється власник; в іншому рішенні набору (116177341) такий самий номер розмічено як НОМЕР.",
         ref=[f"{LAW}, п. 5", P4]),
]


def strip_ws(label):
    return label.strip()


def matches(a, entity, label):
    return entity in a["entity"] and strip_ws(a["pseudonym"]).startswith(label)


def next_label(anns, typ):
    nums = [int(strip_ws(a["pseudonym"]).rsplit("_", 1)[1]) for a in anns if strip_ws(a["pseudonym"]).startswith(typ + "_")]
    return f"{typ}_{max(nums, default=0) + 1}"


def ann(text, s, e, label):
    return {"start": s, "end": e, "entity": text[s:e], "pseudonym": label}


def short(a):
    return {k: a[k] for k in ("start", "end", "entity", "pseudonym")}


def apply(doc, fixes):
    text = doc["original"]
    anns = [dict(a, pseudonym=strip_ws(a["pseudonym"])) for a in doc["annotations"]]
    ws_fixed = sum(1 for a in doc["annotations"] if a["pseudonym"] != strip_ws(a["pseudonym"]))
    corrections = []
    for f in fixes:
        act = f["action"]
        before, after = [], []
        if act in ("relabel", "reboundary_relabel", "merge", "split"):
            hit = [a for a in anns if matches(a, *f["find"])]
            assert hit, f
            rest = [a for a in anns if a not in hit]
            mapping = {}  # кожна окрема відомість (старий псевдонім) отримує свій новий номер
            for a in hit:
                before.append(short(a))
                if act == "reboundary_relabel":
                    m = re.search(f["sub"], a["entity"])
                    s, e = a["start"] + m.start(), a["start"] + m.end()
                else:
                    s, e = a["start"], a["end"]
                if act == "merge":
                    new = f["into"]
                elif a["pseudonym"] in mapping:
                    new = mapping[a["pseudonym"]]
                else:
                    # та сама відомість уже має позначку цього типу (напр. «А1234» як НОМЕР) — беремо її
                    same = [b["pseudonym"] for b in rest if b["pseudonym"].startswith(f["to_type"] + "_")
                            and b["entity"] == text[s:e] and f["to_type"] != "ОСОБА"]
                    new = same[0] if same else next_label(rest + [ann(text, 0, 0, v) for v in mapping.values()], f["to_type"])
                    mapping[a["pseudonym"]] = new
                a.update(ann(text, s, e, new))
                after.append(short(a))
        elif act == "resplit":
            olds = []
            for ent, lab in f["replaces"]:
                cand = [a for a in anns if a["entity"] == ent and a["pseudonym"].startswith(lab)]
                if "near" in f:
                    cand = [a for a in cand if f["near"] in text[max(0, a["start"] - 60):a["start"]]] or cand
                assert cand, (f, ent)
                olds.append(cand[0])
            lo = min(a["start"] for a in olds)
            hi = max(a["end"] for a in olds)
            region = text[lo:hi]
            for a in olds:
                before.append(short(a))
                anns.remove(a)
            for ent, typ in f["spans"]:
                i = region.find(ent)
                assert i >= 0, (f, ent)
                # тип, що вже був серед старих позначок, зберігає свій номер
                keep = [a["pseudonym"] for a in olds if a["pseudonym"].startswith(typ + "_")]
                same = [a["pseudonym"] for a in anns if a["entity"] == ent and a["pseudonym"].startswith(typ + "_")]
                label = (same or keep or [next_label(anns, typ)])[0]
                na = ann(text, lo + i, lo + i + len(ent), label)
                anns.append(na)
                after.append(short(na))
        elif act in ("trim", "extend"):
            hit = [a for a in anns if matches(a, *f["find"])]
            assert hit, f
            for a in hit:
                b0 = short(a)
                s, e = a["start"], a["end"]
                if "new_text" in f:
                    s += a["entity"].index(f["new_text"])
                    e = s + len(f["new_text"])
                elif "new_text_re" in f:
                    m = re.search(f["new_text_re"], a["entity"])
                    s, e = s + m.start(), s + m.end()
                elif "strip" in f:
                    while e > s and text[e - 1] in f["strip"]:
                        e -= 1
                elif "add" in f:
                    assert text[e:e + len(f["add"])] == f["add"]
                    e += len(f["add"])
                if (s, e) != (a["start"], a["end"]):
                    a.update(ann(text, s, e, a["pseudonym"]))
                    before.append(b0)
                    after.append(short(a))
            assert after, f
        elif act == "add":
            label = next_label(anns, f["to_type"])
            for m in re.finditer(re.escape(f["text"]), text):
                assert not any(a["start"] < m.end() and m.start() < a["end"] for a in anns), f
                na = ann(text, m.start(), m.end(), label)
                anns.append(na)
                after.append(short(na))
            assert after, f
        corrections.append({"action": act, "before": before, "after": after,
                            "description": f["why"], "legal_reference": f["ref"]})
    if ws_fixed:
        corrections.append({"action": "normalize_label", "before": [], "after": [],
                            "description": f"У {ws_fixed} позначках псевдонім містив пробіл або перенос рядка (напр. 'ОСОБА_1 '); прибрано. "
                                           "Це артефакт формату, на оцінку не впливає.",
                            "legal_reference": []})
    anns.sort(key=lambda a: (a["start"], a["end"]))
    for x, y in zip(anns, anns[1:]):
        assert y["start"] >= x["end"], (doc["id"], x, y)
    by = defaultdict(list)
    for a in anns:
        by[a["pseudonym"]].append({"start": a["start"], "end": a["end"], "entity": a["entity"]})
    out, pos = [], 0
    for a in anns:
        out += [text[pos:a["start"]], a["pseudonym"]]
        pos = a["end"]
    out.append(text[pos:])
    return {"id": doc["id"], "pseudonymized": "".join(out), "original": text, "annotations": anns,
            "annotations_by_pseudonym": dict(by), "corrections": corrections}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/test/court_cases.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/test/court_cases_fixed.jsonl"
    docs = {str(r["id"]): r for r in map(json.loads, open(src, encoding="utf8"))}
    by_doc = defaultdict(list)
    for f in FIXES:
        by_doc[f["doc"]].append(f)
    with open(dst, "w", encoding="utf8") as g:
        for d in docs:  # порядок як в оригіналі
            if d in by_doc:
                fixed = apply(docs[d], by_doc[d])
                g.write(json.dumps(fixed, ensure_ascii=False) + "\n")
    print(f"{len(by_doc)} рішень, {len(FIXES)} виправлень -> {dst}")


if __name__ == "__main__":
    main()
