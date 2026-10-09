from silver.exclude import ExcludedPool, case_number
from silver.filter import doc_kind

BODY = ("Суд касаційної інстанції дослідив матеріали справи та встановив, що позивач ОСОБА_1 звернувся до відповідача "
        "з вимогою про стягнення заборгованості за договором позики, укладеним між сторонами у місті Полтава, "
        "а відповідач ОСОБА_2 проти позову заперечував і просив відмовити у його задоволенні повністю. ")
POOL = ExcludedPool([("100", "УХВАЛА\nм. Київ\nсправа № 296/5017/18\nпровадження № 61-14884ск23\n" + BODY * 3)])


def test_case_number_is_read_from_the_heading():
    assert case_number("ПОСТАНОВА\nСправа №296/5017/18\nВерховний Суд") == case_number("справа № 296/5017/18 ") == "296/5017/18"
    assert case_number("Справа № 2-а-15/11") == case_number("Справа № 2-a-15/11")  # Cyrillic and Latin «а»
    assert case_number("x" * 2000 + "справа № 1/2/23") is None


def test_pool_document_same_case_and_near_copy_are_excluded():
    assert POOL.reason("100", "будь-який текст") == "test_pool"
    assert POOL.reason("200", "ПОСТАНОВА\nсправа № 296/5017/18\nінший текст рішення апеляційного суду") == "test_case"
    copy = "УХВАЛА\nсправа № 111/22/23\n" + (BODY * 3).replace("ОСОБА_1", "ОСОБА_7") + " Додатковий абзац про судові витрати."
    assert POOL.reason("300", copy) == "test_near_copy"
    other = "РІШЕННЯ\nсправа № 5/5/23\nПозивачка просила розірвати шлюб, зареєстрований у 2015 році, та визначити місце проживання дитини. " * 4
    assert POOL.reason("400", other) is None


def test_doc_kind():
    assert doc_kind("ПОСТАНОВА\nВерховний Суд у складі колегії суддів") == "ВС"
    assert doc_kind("x" * 700 + " проходить службу у військовій частині НОМЕР_1, ст. 122 КАС") == "військові"
    assert doc_kind("x" * 700 + " за ст. 130 КУпАП, протокол про адміністративне правопорушення") == "КУпАП"
    assert doc_kind("x" * 700) == "інше"


def test_excerpt_keeps_the_reasoning_part_and_shifts_labels():
    from silver.excerpt import cut_points, excerpt

    head = "УХВАЛА\nсправа № 1/2/23\nВерховний Суд у складі: головуючого Іщука Т. П.,\nВСТАНОВИВ:\n"
    body = "Позивач Коваль Іван Петрович звернувся до суду з позовом про стягнення коштів. " * 20
    tail = "\nСудді: Т. П. Іщук"
    text = head + body + tail
    assert cut_points(text) == (len(head) - 1, len(head + body) + 1)
    judge, party = text.index("Іщука Т. П."), text.index("Коваль Іван Петрович")
    labels = [{"start": judge, "end": judge + 11, "type": "ОСОБА"}, {"start": party, "end": party + 20, "type": "ОСОБА"}]
    cuts = set()
    for i in range(40):
        out = excerpt({"id": str(i), "text": text, "labels": labels, "masks": []})
        cuts.add((out["cut"]["head"], out["cut"]["signature"]))
        assert [out["text"][l["start"]:l["end"]] for l in out["labels"]] == (["Коваль Іван Петрович"] if out["cut"]["head"] else ["Іщука Т. П.", "Коваль Іван Петрович"])
        assert out["text"] == out["text"].strip()
    assert (True, True) in cuts and len(cuts) > 1
    assert cut_points("В С Т А Н О В И В :\n" + body)[0] == 19


def test_split_keeps_cases_and_fills_of_held_out_decisions_out_of_train():
    from silver.split import DEV_BUCKETS, _bucket, split_docs

    def doc(i, case, fill=1, corpus="vs5k"):
        return {"id": str(i), "fill": fill, "corpus": corpus, "case": case, "labels": [{"type": "НОМЕР", "source": "placeholder"}]}

    cases = [f"{n}/1/23" for n in range(400)]
    docs = [doc(i, c, fill) for i, c in enumerate(cases) for fill in (1, 2)]
    docs += [doc(1000 + i, c, corpus="edrsr") for i, c in enumerate(cases[:50])]  # another decision of the same case
    train, dev, _ = split_docs(docs)
    assert dev and all(d["fill"] == 1 and _bucket(d["case"]) < DEV_BUCKETS for d in dev)
    assert len({d["id"] for d in dev}) == len(dev)
    assert not {d["case"] for d in dev} & {d["case"] for d in train}
    assert sum(d["fill"] == 2 for d in train) == sum(d["fill"] == 1 and d["corpus"] == "vs5k" for d in train) > 300
