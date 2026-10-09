import random

import pytest

from silver.fill.orgs import OrgGenerator, real_org
from silver.orgnames import (clean_company, company_case, edr_company, harvest_bodies, harvest_companies, load_names,
                             treasury_kind, write_table)


@pytest.mark.parametrize("raw, expected", [
    ("Комтех Логістік", "Комтех Логістік"), (" Дельта-Пром ", "Дельта-Пром"), ("Б’юті Груп", "Б'юті Груп"),
    ("ІНФОРМАЦІЯ_1", None), ("Фирма Эталон", None), ("Фірма ТОВ Зоря", None), ("12", None),
    ("Дуже довга назва з шести окремих слів", None),
])
def test_clean_company(raw, expected):
    assert clean_company(raw) == expected


def test_harvest_companies_keeps_the_usual_spelling_and_skips_nested_quotes():
    texts = ["позов ТОВ «Комтех Логістік» до ПрАТ \"Дельта-Пром\" та Товариства з обмеженою відповідальністю «Комтех Логістік»",
             "ТОВ «КОМТЕХ ЛОГІСТІК» і ТОВ «ІНФОРМАЦІЯ_1», а також ТОВ «Фірма «Академпромбуд»"]
    found = harvest_companies(texts)
    assert found["Комтех Логістік"] == 3 and found["Дельта-Пром"] == 1
    assert not any("ІНФОРМАЦІЯ" in n or "Академпромбуд" in n or n.isupper() for n in found)


def test_harvest_bodies_takes_nominative_names_seen_in_two_decisions():
    a = "Головне управління Пенсійного фонду України в Одеській області подало скаргу. Позов до Головного управління Держпраці у Львівській області."
    b = "Відповідач - Головне управління Пенсійного фонду України в Одеській області, який заперечив."
    c = "Департамент патрульної поліції Національної поліції України не погодився."
    found = harvest_bodies([a, b, c])
    assert set(found) == {"Головне управління Пенсійного фонду України в Одеській області"}


@pytest.mark.parametrize("name, expected", [
    ("Управління соціального захисту населення Павлоградської районної державної адміністрації", "social"),
    ("Управління праці та соціального захисту населення Гірницької райадміністрації", "social"),
    ("Вінницька обласна прокуратура", "prosecutor"), ("Київська міська прокуратура", "prosecutor"),
    ("Херсонський апеляційний суд", "court"), ("Господарський суд Львівської області", "court"),
    ("Прокуратура Автономної Республіки Крим та міста Севастополя", None),
    ("Комунальний заклад \"Ліцей № 3\"", None), ("Виконавчий комітет Стрийської міської ради", None),
    ("Територіальний центр соціального обслуговування (надання соціальних послуг) міста Вараш", None),
])
def test_treasury_kind(name, expected):
    assert treasury_kind(name) == expected


@pytest.mark.parametrize("upper, expected", [
    ("КОМТЕХ ЛОГІСТІК", "Комтех Логістік"), ("БМ ТРЕЙД", "БМ Трейд"), ("ДЕЛЬТА-ПРОМ ЛТД", "Дельта-Пром ЛТД"),
    ("ЧЕРКАСЬКА ВИРОБНИЧА КОМПАНІЯ", "Черкаська Виробнича Компанія"),
])
def test_company_case(upper, expected):
    assert company_case(upper) == expected


def test_edr_company_takes_registered_companies_only():
    line = ("<SUBJECT><RECORD>1</RECORD><NAME>ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ &quot;ФОРЕСТПРОМТРЕЙД&quot;</NAME><SHORT_NAME/>"
            "<OPF>ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ</OPF><EDRPOU>33444728</EDRPOU><STAN>зареєстровано</STAN></SUBJECT>")
    assert edr_company(line) == ("33444728", "Форестпромтрейд")
    assert edr_company(line.replace("зареєстровано", "припинено")) is None
    assert edr_company(line.replace("<OPF>ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ</OPF>", "<OPF>ГРОМАДСЬКА ОРГАНІЗАЦІЯ</OPF>")) is None
    assert edr_company(line.replace("ФОРЕСТПРОМТРЕЙД", "ФІРМА &quot;ФОРЕСТ&quot;")) is None


@pytest.mark.parametrize("name, case, expected", [
    ("Вінницька обласна прокуратура", "gent", "Вінницької обласної прокуратури"),
    ("Херсонський апеляційний суд", "gent", "Херсонського апеляційного суду"),
    ("Управління соціального захисту населення Павлоградської районної державної адміністрації", "datv",
     "Управлінню соціального захисту населення Павлоградської районної державної адміністрації"),
    ("Головне управління Пенсійного фонду України в Одеській області", "gent", "Головного управління Пенсійного фонду України в Одеській області"),
    ("Департамент патрульної поліції", "accs", "Департамент патрульної поліції"),
])
def test_real_org_declines_only_its_head(name, case, expected):
    assert real_org(name).render(case) == expected
    assert real_org(name).render() == name


def test_table_round_trip(tmp_path):
    rows = [("court", "company", "Комтех"), ("edr", "company", "Форест"), ("court", "company", "Альфа"), ("court", "company", "Альфа")]
    write_table(rows, tmp_path / "t.parquet")
    assert load_names(tmp_path / "t.parquet") == {("court", "company"): ["Альфа", "Комтех"], ("edr", "company"): ["Форест"]}
    assert load_names(tmp_path / "missing.parquet") == {}


NAMES = {("court", "company"): ["Судова Назва"], ("edr", "company"): ["Реєстрова Назва"], ("court", "body"): ["Головне управління Держпраці у Львівській області"],
         ("treasury", "social"): ["Управління соціального захисту населення Стрийської районної державної адміністрації"],
         ("treasury", "prosecutor"): ["Вінницька обласна прокуратура"], ("treasury", "court"): ["Херсонський апеляційний суд"]}


def test_registers_are_used_only_where_they_fit():
    g = OrgGenerator(random.Random(1), names=NAMES)
    plain = {g.company_name() for _ in range(300)}
    assert "Судова Назва" in plain and not any("реєстрова" in n.lower() for n in plain)  # the register: commercial cases only
    commercial = {g.company_name(commercial=True) for _ in range(300)}
    assert {"Судова Назва", "Реєстрова Назва", "РЕЄСТРОВА НАЗВА"} <= commercial
    civil = {g.civil_body().render() for _ in range(400)}
    assert "Головне управління Держпраці у Львівській області" in civil
    assert not civil & {"Вінницька обласна прокуратура", "Херсонський апеляційний суд", NAMES[("treasury", "social")][0]}
    social = {g.civil_body(registers=True).render() for _ in range(400)}
    assert {"Вінницька обласна прокуратура", NAMES[("treasury", "social")][0]} <= social
    # data v6: a court is never a labelled value - decisions name courts in the open everywhere
    assert "Херсонський апеляційний суд" not in social and not any(" суд" in n for n in social)
    assert len(plain) > 50 and len(civil) > 50  # composed names remain
    assert g.used["company_court"] and g.used["company_edr"] and g.used["company_composed"] and g.used["body_composed"]


def test_without_the_table_every_name_is_composed():
    g = OrgGenerator(random.Random(1), names={})
    for _ in range(200):
        g.institution(military=True), g.institution(commercial=True), g.institution(social=True)
    assert set(g.used) <= {"body_composed", "company_composed"}
