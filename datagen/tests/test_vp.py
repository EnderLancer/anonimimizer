"""Open enforcement-proceeding numbers: the forms the pattern finds and the person test that labels or masks them."""
import time

import pytest
import regex as re

from silver import patterns as P
from silver.label import label_doc
from silver.numbers import VP_PERSON_HEAD, VP_PERSON_NEAR, label_numbers

NAME = "Коваль Іван Петрович"
FILLER = "Суд дослідив матеріали справи та заслухав пояснення сторін. "  # neutral text to set distances


def _found(text: str) -> list[str]:
    return [n for m in P.VP.finditer(text) for n in m.captures("n")]


def _person(text: str, source: str = "placeholder") -> dict:
    s = text.index(NAME)
    return {"start": s, "end": s + len(NAME), "type": "ОСОБА", "pseudonym": "ОСОБА_1" if source == "placeholder" else None,
            "source": source}


def _doc(text: str, labels: list[dict] | None = None, masks: list[dict] | None = None) -> dict:
    return {"id": "1", "text": text, "text_pseudonymized": text, "labels": labels or [], "masks": masks or []}


def _vp(doc: dict) -> tuple[list[str], list[str]]:
    t = doc["text"]
    return ([t[l["start"]:l["end"]] for l in doc["labels"] if l["source"] == "vp"],
            [t[m["start"]:m["end"]] for m in doc["masks"] if m["reason"] == "vp_no_person"])


@pytest.mark.parametrize("text,expected", [
    ("відкрито виконавче провадження № 73659776, накладено арешт", ["73659776"]),
    ("закінчення виконавчого провадження №61871019, на підставі", ["61871019"]),
    ("провадження ВП № 73939285 з примусового виконання", ["73939285"]),
    ("про існування ВП№68236237 про стягнення", ["68236237"]),
    # every number of a list after one cue
    ("виконавчі провадження № 74591493 та № 74591949 закінчено", ["74591493", "74591949"]),
    ("у виконавчому провадженні №6395898, №63958373, №63958006 та №63971432 щодо", ["6395898", "63958373", "63958006", "63971432"]),
    ("матеріалів виконавчого провадження №№69655704, 69474005, 69653024, 58731828 та постанов", ["69655704", "69474005", "69653024", "58731828"]),
    ("у таких виконавчих провадженнях: 74906985, 74957903; - скасувати арешт", ["74906985", "74957903"]),
    # the cue without «№»
    ("в рамках ВП 62884789 від 22.09.2016 року", ["62884789"]),
    ("перебуває виконавче провадження 68561892 щодо стягнення", ["68561892"]),
    # «за №», «по №»
    ("відкрите виконавче провадження за № 74890886 з примусового виконання", ["74890886"]),
    ("про закінчення виконавчого провадження по № 73323201. Відновити", ["73323201"]),
    # «N» for «№», also with the mangled sign of some source texts
    ("зведеного виконавчого провадження N 74808786, а саме", ["74808786"]),
    ("у виконавчому провадженні ВП N?72485916 з примусового", ["72485916"]),
    # АСВП and ЗВП as cues
    ("у виконавчому провадженні АСВП № 8757608 з примусового", ["8757608"]),
    ("по цій справі) - № АСВП: 68293414 від 21.01.2022, № АСВП: 68301385 від", ["68293414", "68301385"]),
    ("матеріали виконавчого провадження АСВП 25636216 та пояснення", ["25636216"]),
    ("зареєстроване в системі АСВП за №9029729 щодо стягнення", ["9029729"]),
    ("боржника ВП №70939393 ЗВП №36035661 від 24.04.2024", ["70939393", "36035661"]),
    # a date between the cue and the number
    ("про відкриття виконавчого провадження від 23.07.2021 № 66181350, виданої", ["66181350"]),
    ("виконавчого провадження від 22 грудня 2023 року ВП № 73662722 та вимогу", ["73662722"]),
    # 7 to 10 digits
    ("ВП № 7637954 про стягнення; ВП № 1234567890.", ["7637954", "1234567890"]),
    # a company code earlier in the sentence does not block a proceeding number
    ("ТОВ «Фірма» (код ЄДРПОУ 12345678), ВП № 87654321", ["87654321"]),
    # «код» blocks only when nothing but link words stand between it and «ВП»
    ("постанова з кодом доступу ВП № 12345678 надіслана", ["12345678"]),
    ("ідентифікаційний код боржника у ВП № 87654321 зазначено", ["87654321"]),
    # «ВП» glued to the number
    ("винесено постанову ВП75258642 про відкриття", ["75258642"]),
    # without «№» an amount or a phone number is not part of the list, the numbers around it are
    ("виконавче провадження № 12345678 та 2500000 грн виконавчого збору", ["12345678"]),
    ("ВП № 12345678, 0501234567 (телефон виконавця)", ["12345678"]),
    ("у ВП 62884789 стягнуто 1500000 гривень", ["62884789"]),
])
def test_pattern_matches(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize("text", [
    "ГУ ДПС у Дніпропетровській області (код ЄДРПОУ ВП 44118658) на користь",   # ВП = відокремлений підрозділ
    "відокремлений підрозділ Держмитслужби, Код ЄДРПОУ ВП: 43971364), рівнозначній",
    "вул. Сімферопольська, 17-а, код ЄДРПОУ ВП - 44118658)",
    "податковий орган (код за ЄДРПОУ ВП 44118658), код в ЄДР-ВП 43971364, код ЄДРПОУ – ВП 44118658, код ВП 43143039",
    "відокремлений підрозділ, ЄДРПОУ ВП44118658",
    "у справі ВП № 755/19637/17 від 01.02.2018",          # a case number
    "арешт коштів боржника: ВП №3676/7, ВП №36/141",      # old-style numbers with a slash
    "ВП № 1234567/18 та виконавче провадження 7654321/2",
    "виконавче провадження № 123456 закінчено",           # 6 digits
    "ВП № 12345678901 та наказ № 1234567890123",          # 11 and more digits
    "довідка ВПО 12345678 видана",                        # not the cue
    "Закон України «Про виконавче провадження» від 02.06.2016 № 1404-VIII",
    "закінчено виконавче провадження, стягнуто 1434048,11 грн",
    "стягнуто у виконавчому провадженні 1500000 гривень",   # an amount in whole hryvnias
    "в межах виконавчого провадження 25000000 грн та ВП 3500000 тис. грн",
    "телефон відділу виконавчого провадження: 0671234567",  # a phone number
    "виконавче провадження відкрито 12.03.2020, справа № 2-2722/08",
])
def test_pattern_does_not_match(text):
    assert _found(text) == []


def test_pattern_is_not_much_slower_than_a_plain_cue():
    # a lookbehind in front of the cue alternation ran at every character and made the pattern 10 times slower than this
    # plain one (1.6 times now); best of 5 runs on 350,000 characters, the limit leaves room for a busy machine
    plain = re.compile(r"(?:\bВП|\b(?i:виконавч\w+\s+провадженн\w+))\s*№\s*(\d{7,10})\b")
    text = (FILLER + "Державним виконавцем відкрито виконавче провадження № 61234567 про стягнення з боржника коштів, "
            "код ЄДРПОУ 37972475. ") * 2000

    def best(rx) -> float:
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            assert sum(1 for _ in rx.finditer(text)) == 2000
            times.append(time.perf_counter() - t0)
        return min(times)

    assert best(P.VP) < 5 * best(plain)


def test_labelled_with_person_within_the_window():
    gap = "х" * (VP_PERSON_NEAR - 40)
    text = "й" * (VP_PERSON_HEAD + 10) + f" боржник {NAME} {gap} у ВП № 61234567 та №65432109."
    labels, masks, _ = label_numbers(text, [_person(text)], [])
    assert [text[l["start"]:l["end"]] for l in labels] == ["61234567", "65432109"]
    assert all(l["type"] == "НОМЕР" and l["source"] == "vp" and l["pseudonym"] is None for l in labels)
    assert masks == []


def test_labelled_with_person_after_the_number():
    text = "й" * (VP_PERSON_HEAD + 10) + f" у виконавчому провадженні 61234567 про стягнення з {NAME} боргу"
    assert _vp(label_doc(_doc(text, labels=[_person(text)]))) == (["61234567"], [])


def test_masked_without_person():
    text = "ТОВ «Альфа» оскаржує дії виконавця у ВП № 61234567 щодо стягнення на користь АТ «Банк»."
    assert _vp(label_doc(_doc(text))) == ([], ["61234567"])


def test_masked_when_person_is_far_and_not_in_heading():
    gap = "х" * (VP_PERSON_NEAR + 5)
    text = "й" * (VP_PERSON_HEAD + 10) + f" боржник {NAME} {gap} у ВП № 61234567."
    assert _vp(label_doc(_doc(text, labels=[_person(text)]))) == ([], ["61234567"])


def test_labelled_when_person_stands_only_in_heading():
    text = f"УХВАЛА за скаргою {NAME} на дії виконавця. " + FILLER * 60 + "Арешт накладено у ВП № 61234567."
    assert text.index("61234567") > VP_PERSON_HEAD + VP_PERSON_NEAR
    assert _vp(label_doc(_doc(text, labels=[_person(text)]))) == (["61234567"], [])


def test_open_names_of_officials_do_not_count():
    # the judge of the heading and the executor next to the number are left open by the clerks: they are not parties
    text = ("Суддя Гордійчук С. О. розглянув скаргу ТОВ «Альфа». " + FILLER * 5
            + "Державним виконавцем Петренко О.В. відкрито виконавче провадження № 61234567.")
    d = label_doc(_doc(text))
    assert [l["source"] for l in d["labels"] if l["type"] == "ОСОБА"]  # the open names are labelled as persons
    assert _vp(d) == ([], ["61234567"])


def test_open_name_of_a_party_counts():
    # a debtor the clerk left open is a natural person all the same; such a decision may have no placeholder at all
    head = "Суддя Гордійчук С. О., позивач ТОВ «Альфа», відповідач: Фізична особа-підприємець Харченко Віктор Леонідович. "
    text = head + FILLER * 60 + "Постановою від 01.02.2023 закінчено ВП 70709870."
    assert text.index("70709870") > VP_PERSON_HEAD + VP_PERSON_NEAR
    assert _vp(label_doc(_doc(text))) == (["70709870"], [])
    assert _vp(label_doc(_doc(text.replace("Фізична особа-підприємець ", "представник Банку ")))) == ([], ["70709870"])
    # an entrepreneur who is a third party of a dispute between companies is not the debtor (decision 119839903)
    third = text.replace("відповідач: Фізична", "відповідач ТОВ «Бета», третя особа на стороні відповідача Фізична")
    assert _vp(label_doc(_doc(third))) == ([], ["70709870"])
    for party in ("ФОП", "боржник", "стягувача -"):
        text = "й" * (VP_PERSON_HEAD + 10) + f" {party} Харченко Віктор Леонідович, відкрито ВП № 70709870."
        assert _vp(label_doc(_doc(text))) == (["70709870"], []), party


def test_hidden_third_party_in_heading_counts():
    # known cost (decisions 120114987, 121902057): the proceeding is between legal entities, the hidden person is a third
    # party. Kept, because in a complaint against the executor the third party is as often the debtor
    text = (f"ТОВ «Альфа» до відділу ДВС, третя особа - {NAME}, про визнання дій протиправними. " + FILLER * 60
            + "Арешт накладено у ВП № 61234567.")
    assert _vp(label_doc(_doc(text, labels=[_person(text)]))) == (["61234567"], [])


def test_same_number_gets_one_decision_in_a_document():
    # the person is near the first occurrence only: the second one must not stay open in the exported text
    far = "х" * (VP_PERSON_NEAR + 100)
    text = "й" * (VP_PERSON_HEAD + 10) + f" боржник {NAME} у ВП № 61234567. {far} Закінчено ВП № 61234567 та ВП № 65432109."
    assert _vp(label_doc(_doc(text, labels=[_person(text)]))) == (["61234567", "61234567"], ["65432109"])


def test_list_of_three_numbers():
    text = f"{NAME} просить замінити стягувача у виконавчих провадженнях №72740621, №72741630 та №72744531."
    assert _vp(label_doc(_doc(text, labels=[_person(text)]))) == (["72740621", "72741630", "72744531"], [])
    assert _vp(label_doc(_doc(text.replace(NAME, "ТОВ «Альфа»")))) == ([], ["72740621", "72741630", "72744531"])


@pytest.mark.parametrize("value", ["КВ/5469", "2954501730", "58365"])
def test_list_goes_on_after_a_filled_placeholder(value):
    # «виконавчі провадження: № НОМЕР_1, № 58365604, …»: the open numbers must not depend on the shape of the fill value
    text = f"{NAME} оскаржує виконавчі провадження: № {value}, № 58365604, № 61493933 та 61876187, а також наказ № 1234567."
    s = text.index(value)
    filled = {"start": s, "end": s + len(value), "type": "НОМЕР", "pseudonym": "НОМЕР_1", "source": "placeholder"}
    assert _vp(label_doc(_doc(text, labels=[_person(text), filled]))) == (["58365604", "61493933", "61876187"], [])
    text = text.replace(NAME, "ТОВ «Альфа»")
    filled = {**filled, "start": text.index(value), "end": text.index(value) + len(value)}
    assert _vp(label_doc(_doc(text, labels=[filled]))) == ([], ["58365604", "61493933", "61876187"])


def test_filled_placeholder_without_cue_starts_no_list():
    text = f"{NAME}, договір № КВ/5469, № 58365604 та рахунок 61493933."
    s = text.index("КВ/5469")
    filled = {"start": s, "end": s + 7, "type": "НОМЕР", "pseudonym": "НОМЕР_1", "source": "placeholder"}
    assert _vp(label_doc(_doc(text, labels=[_person(text), filled]))) == ([], [])


def test_number_inside_filled_placeholder_is_left_alone():
    text = f"{NAME} є боржником у ВП № 61234567 та № 65432109."
    s = text.index("61234567")
    filled = {"start": s, "end": s + 8, "type": "НОМЕР", "pseudonym": "НОМЕР_1", "source": "placeholder"}
    d = label_doc(_doc(text, labels=[_person(text), filled]))
    assert [(text[l["start"]:l["end"]], l["source"]) for l in d["labels"] if l["type"] == "НОМЕР"] == [
        ("61234567", "placeholder"), ("65432109", "vp")]
    # without a person the placeholder keeps its label and gets no mask on top
    text = text.replace(NAME, "ТОВ «Альфа»")
    s = text.index("61234567")
    d = label_doc(_doc(text, labels=[{**filled, "start": s, "end": s + 8}]))
    assert [l["source"] for l in d["labels"] if l["type"] == "НОМЕР"] == ["placeholder"]
    assert _vp(d) == ([], ["65432109"])


def test_number_inside_existing_mask_is_left_alone():
    text = f"{NAME} є боржником у ВП № 61234567."
    s = text.index("61234567")
    old = {"start": s - 5, "end": s + 8, "reason": "fill_unsure"}
    labels, masks, _ = label_numbers(text, [_person(text)], [old])
    assert labels == [] and masks == []


def test_label_doc_keeps_the_mask_through_merge():
    text = "ТОВ «Альфа» оскаржує дії у ВП № 61234567 та у виконавчому провадженні 65432109 від 01.02.2023."
    d = label_doc(_doc(text))
    assert [(text[m["start"]:m["end"]], m["reason"]) for m in d["masks"]] == [
        ("61234567", "vp_no_person"), ("65432109", "vp_no_person")]
    assert not [l for l in d["labels"] if l["type"] == "НОМЕР"]


def test_other_number_layers_are_unchanged():
    text = "РНОКПП 3127985494, паспорт серії КВ 123456, код ЄДРПОУ ВП 44118658, код ЄДРПОУ 37972475"
    d = label_doc(_doc(text))
    assert [(text[l["start"]:l["end"]], l["source"]) for l in d["labels"]] == [("3127985494", "rnokpp"), ("КВ 123456", "passport")]
    assert d["masks"] == []
