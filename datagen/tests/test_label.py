import pytest

from silver.label import label_doc
from silver.merge import merge
from silver import patterns as P


def _doc(text, labels=None, masks=None):
    return {"id": "1", "text": text, "text_pseudonymized": text, "labels": labels or [], "masks": masks or []}


def _spans(doc, source=None):
    return [(doc["text"][l["start"]:l["end"]], l["type"], l["source"]) for l in doc["labels"] if source is None or l["source"] == source]


@pytest.mark.parametrize("text,expected", [
    ("розглянувши справу за участю судді Іщук Т. П., секретаря Дубас Т.В.,", ["Іщук Т. П.", "Дубас Т.В."]),
    ("за участю адвоката Сенчука А.І. та прокурора Мороза А.О.", ["Сенчука А.І.", "Мороза А.О."]),
    ("у складі: головуючого судді Томаша В.І., суддів Лузан Л.В., Комаса О.В.,", ["Томаша В.І.", "Лузан Л.В.", "Комаса О.В."]),
    ("приватним нотаріусом Київського міського нотаріального округу Петренко О.В. посвідчено", ["Петренко О.В."]),
    ("Фізична особа-підприємець Протеняк Олександр Михайлович припинив діяльність", ["Протеняк Олександр Михайлович"]),
])
def test_persons_found(text, expected):
    d = label_doc(_doc(text))
    assert [s for s, t, _ in _spans(d) if t == "ОСОБА"] == expected


@pytest.mark.parametrize("text", [
    "проживає по вул. Лесі Українки, 5",
    "Кодекс України Про адміністративні правопорушення",
    "рішення Верховного Суду України від 1 січня",
    "школа ім. Т. Г. Шевченка",
])
def test_persons_not_found(text):
    d = label_doc(_doc(text))
    assert not [s for s, t, _ in _spans(d) if t == "ОСОБА"], _spans(d)


def test_rnokpp_and_edrpou():
    d = label_doc(_doc("РНОКПП 3127985494, код ЄДРПОУ 37972475, ІПН: 2970309241"))
    assert sorted(s for s, t, src in _spans(d) if src == "rnokpp") == ["2970309241", "3127985494"]


def test_enforcement_proceeding_numbers():
    d = label_doc(_doc("у ВП № 61234567 та у виконавчому провадженні №65432109; справа № 2-2722/08, наказ № 1234567890123"))
    assert [s for s, t, src in _spans(d) if src == "vp"] == ["61234567", "65432109"]
    assert all(t == "НОМЕР" for _, t, src in _spans(d) if src == "vp")


def test_passport_needs_context():
    d = label_doc(_doc("паспорт громадянина України серії КВ 123456, виданий ... спецупакування № АБ030649"))
    assert [s for s, t, src in _spans(d) if src == "passport"] == ["КВ 123456"]
    assert any(m["reason"] == "series_no_ctx" for m in d["masks"])


def test_iban_person_vs_institution():
    person = "стягнути з Коваля Івана Петровича на користь Шевченко Ольги Іванівни на рахунок UA473221280000026731954859945 кошти"
    inst = "судовий збір на рахунок UA908999980313111256000026001, отримувач: ГУК у м. Києві, код ЄДРПОУ 37993783"
    d1, d2 = label_doc(_doc(person)), label_doc(_doc(inst))
    assert [s for s, t, src in _spans(d1) if t == "НОМЕР"] == ["UA473221280000026731954859945"]
    assert not [s for s, t, src in _spans(d2) if t == "НОМЕР"]
    assert not any(m["reason"] == "iban_unknown" for m in d2["masks"])


def test_dob_only_with_cue():
    d = label_doc(_doc("Коваль Іван Петрович, 12.01.1980 року народження, помер 18.03.2020. 29.12.2023 року надійшла заява; "
                       "запис про народження № 24 від 3 листопада 1992 року"))
    assert [s for s, t, src in _spans(d) if src == "dob"] == ["12.01.1980", "18.03.2020"]


def test_email_personal_vs_gov():
    d = label_doc(_doc("позивач Коваль Іван Петрович, електронна пошта: ivan.koval@gmail.com, суд: inbox@dp.arbitr.gov.ua, "
                       "ТОВ «Фірма», e-mail: firma2010@ukr.net, інше: info@firma.ua"))
    assert [s for s, t, src in _spans(d) if src == "email"] == ["ivan.koval@gmail.com"]
    assert sorted(m["reason"] for m in d["masks"]) == ["email_other", "email_other"]


def test_merge_same_type_union_and_conflict_mask():
    text = "Коваль Іван Петрович 1234567890"
    labels = [
        {"start": 0, "end": 20, "type": "ОСОБА", "pseudonym": "ОСОБА_1", "source": "placeholder"},
        {"start": 0, "end": 6, "type": "ОСОБА", "pseudonym": None, "source": "ner"},
        {"start": 21, "end": 31, "type": "НОМЕР", "pseudonym": None, "source": "rnokpp"},
        {"start": 15, "end": 25, "type": "ІНФОРМАЦІЯ", "pseudonym": None, "source": "dob"},
    ]
    out, masks = merge(text, labels, [])
    assert [(l["start"], l["end"], l["type"]) for l in out] == [(0, 20, "ОСОБА"), (21, 31, "НОМЕР")]
    assert masks == []  # conflict mask overlaps kept labels and is dropped


def test_placeholder_regex_shapes():
    assert P.NAME_ANY.fullmatch("Соя Д.")
    assert P.NAME_ANY.fullmatch("Романишин М. М.")
    assert P.NAME_ANY.fullmatch("О.М. Таварткіладзе")
    assert P.NAME_ANY.fullmatch("Попова Сергія Олексійовича")


@pytest.mark.parametrize("text,expected", [
    ("Коваль Іван Петрович, мешкає за адресою: 07341, Київська обл., Вишгородський р-н, Жукин, Лісова 3, за ч. 1 ст. 130 Кодексу",
     ["07341, Київська обл., Вишгородський р-н, Жукин, Лісова 3"]),
    ("Коваль Іван Петрович, зареєстрований за адресою: проспект Берестейський, 55/2, м. Київ, 03113, у Солом'янському районі",
     ["проспект Берестейський, 55/2, м. Київ, 03113"]),
    ("Коваль Іван Петрович, яка проживає разом з ним за вказаною адресою, чим вчинив домашнє насильство", []),
    ("Коваль Іван Петрович не проживає за зазначеною адресою протягом тривалого періоду", []),
])
def test_address_grammar(text, expected):
    d = label_doc(_doc(text))
    assert [s for s, t, src in _spans(d) if src == "address"] == expected


def test_passport_rejects_banknotes_and_acts():
    d = label_doc(_doc("паспорт громадянина України МА730156; купюри по 100 гривень серії УЛ 926431; "
                       "державного акта на землю серія ЯО 556104; протоколу серії ГП 108315"))
    assert [s for s, t, src in _spans(d) if src == "passport"] == ["МА730156"]


def test_card_needs_context():
    d = label_doc(_doc("у кримінальному провадженні № 1202078120000434; картка 4149 4991 2076 3107 належить Ковалю Івану Петровичу"))
    assert [s for s, t, src in _spans(d) if src == "card"] == ["4149 4991 2076 3107"]
    assert not [m for m in d["masks"] if m["reason"] == "card_unknown"]


def test_dob_needs_comma_after_person():
    d = label_doc(_doc("Радченко Михайло Андрійович 11 лютого 2002 року склав заповіт. Коваль Іван Петрович, 12.01.1980, громадянин"))
    assert [s for s, t, src in _spans(d) if src == "dob"] == ["12.01.1980"]


def test_caps_name_needs_dictionary_first_name():
    d = label_doc(_doc("посвідченого нотаріусом Костянтинівської ДНК; Суддя апеляційного суду Сергій БОЛОТІН"))
    assert [s for s, t, src in _spans(d) if t == "ОСОБА"] == ["Сергій БОЛОТІН"]


def test_render_pseudonymized_replaces_open_names():
    from silver.export import assign_pseudonyms, render_pseudonymized
    text = "Позов Коваля Івана Петровича. Головуючий суддя Гордійчук С. О. Судді: Боймиструк С.В. Шимків С.С."
    doc = label_doc(_doc(text, labels=[{"start": 6, "end": 28, "type": "ОСОБА", "pseudonym": "ОСОБА_1", "source": "placeholder"}]))
    anns = assign_pseudonyms(doc)
    out = render_pseudonymized(text, anns)
    for name in ("Коваля", "Гордійчук", "Боймиструк", "Шимків"):
        assert name not in out
    assert out.count("ОСОБА_") == 4


@pytest.mark.parametrize("text,expected", [
    ('голова правління кредитної спілки "Центр-кредит" Кравчук С.П. подав заяву', ["Кравчук С.П."]),
    ("помилково визначено суддю «Соколову В. В.» замість", ["Соколову В. В."]),
    ('ТОВ з іноземними інвестиціями "Дока Україна Т.О.В." звернулось', []),
])
def test_names_next_to_quotes(text, expected):
    d = label_doc(_doc(text))
    assert [s for s, t, _ in _spans(d) if t == "ОСОБА"] == expected
