import random
import re

import pytest

from silver.fill.addresses import TABLES, USE, AddressGenerator
from silver.fill.context import number_subtype
from silver.fill.numbers import NumberGenerator
from silver.fill.region import doc_region
from silver.fill.run import Filler
from silver.io import RESOURCES
from silver.normalize import PLACEHOLDER, find_placeholders, normalize


def test_fill_roundtrip_offsets():
    raw = ("Позивач ОСОБА_1 , ІНФОРМАЦІЯ_1 , РНОКПП НОМЕР_1 , мешкає за адресою: АДРЕСА_1 , звернувся до суду з позовом "
           "до ОСОБА_2 . Представник ОСОБА_1 адвокат Іщук Т. П. просив стягнути з ОСОБА_2 на користь ОСОБА_1 кошти. "
           "Електронна адреса ОСОБА_2 : ІНФОРМАЦІЯ_2 . Народилась ІНФОРМАЦІЯ_3 .")
    text = normalize(raw)
    doc = {"id": "t", "text": text, "placeholders": find_placeholders(text)}
    out = Filler(seed=1).fill(doc)
    assert not PLACEHOLDER.search(out["text"])
    assert not re.search(r"\S ,", out["text"])
    for l in out["labels"]:
        frag = out["text"][l["start"]:l["end"]]
        assert frag == frag.strip() and frag
    persons = {l["pseudonym"] for l in out["labels"] if l["type"] == "ОСОБА"}
    assert persons == {"ОСОБА_1", "ОСОБА_2"}
    assert out["n_persons"] == 2
    types = {l["type"] for l in out["labels"]}
    assert types == {"ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ"}
    email = [out["text"][l["start"]:l["end"]] for l in out["labels"] if l["pseudonym"] == "ІНФОРМАЦІЯ_2"][0]
    assert "@" in email
    rnokpp = [out["text"][l["start"]:l["end"]] for l in out["labels"] if l["pseudonym"] == "НОМЕР_1"][0]
    assert re.fullmatch(r"\d{10}", rnokpp)


def test_fill_is_deterministic():
    text = normalize("Позов ОСОБА_1 до ОСОБА_2 , ІНФОРМАЦІЯ_1 .")
    doc = {"id": "t", "text": text, "placeholders": find_placeholders(text)}
    a = Filler(seed=7).fill(doc)
    b = Filler(seed=7).fill(doc)
    assert a == b


def _fill(raw: str, seed: int = 1) -> dict:
    text = normalize(raw)
    return Filler(seed=seed).fill({"id": "t", "text": text, "placeholders": find_placeholders(text)})


def _values(out: dict, prefix: str) -> list[str]:
    return [out["text"][l["start"]:l["end"]] for l in out["labels"] if l["pseudonym"].startswith(prefix)]


def test_address_parts_come_from_one_directory_row():
    gen = AddressGenerator(random.Random(3))
    rows: dict[tuple, set] = {}  # region, settlement, index, street -> house numbers
    for r in gen.rows:
        rows.setdefault((r[0], r[4], r[5], r[7]), set()).update(r[8].split(","))
    ctx = gen.doc_context("Рішенням Малиновського районного суду м. Одеси, Одеська область")
    for _ in range(300):
        a = gen.new_address(ctx)
        houses = rows[(a["region"], a["np"], a["zip"], a["street"])]
        assert a["house"].upper() in houses
        assert gen.render(a) and "None" not in gen.render(a) + gen.render(a, variant=True)


def test_addresses_follow_the_region_and_settlements_of_the_document():
    gen = AddressGenerator(random.Random(4))
    ctx = gen.doc_context("Постановою Одеського апеляційного суду … мешканця с. Нерубайського Одеського району Одеської області")
    assert ctx["region"] == "Одеська" and [k[3] for k in ctx["places"]] == ["Нерубайське"]
    made = [gen.new_address(ctx) for _ in range(400)]
    assert sum(a["region"] == "Одеська" for a in made) > 320
    assert sum(a["np"] == "Нерубайське" for a in made) > 200
    # the city of Kyiv has no region and no district around it, only districts inside it
    assert not any(a["with_region"] or (a["district"] and not a["city_district"]) for a in made if a["region"] == "Київ")


@pytest.mark.skipif("osm" not in USE or not (RESOURCES / TABLES["osm"]).exists(), reason="scripts/fetch_osm.py was not run")
def test_openstreetmap_rows_add_city_districts():
    gen = AddressGenerator(random.Random(6))
    ctx = gen.doc_context("Печерський районний суд міста Києва, Київський апеляційний суд")
    made = [gen.new_address(ctx) for _ in range(3000)]
    kyiv = [a for a in made if a["np"] == "Київ"]
    share = sum(a["src"] == "osm" for a in kyiv) / len(kyiv)
    assert 0.2 < share < 0.4
    districts = {a["district"] for a in kyiv if a["city_district"]}
    assert districts and districts <= {"Голосіївський", "Дарницький", "Деснянський", "Дніпровський", "Оболонський", "Печерський",
                                       "Подільський", "Святошинський", "Солом’янський", "Шевченківський"}
    assert all(a["src"] == "osm" for a in kyiv if a["city_district"])
    a = next(a for a in kyiv if a["city_district"])
    assert f"{a['district']} р" in gen.render(a) and "обл" not in gen.render(a)
    assert all(not a["with_zip"] for a in made if not a["zip"])


def test_one_address_style_per_document():
    out = _fill("Зареєстрований: АДРЕСА_1 , проживає: АДРЕСА_2 , майно: АДРЕСА_3 , АДРЕСА_4 , АДРЕСА_5 . Харківська область.", seed=11)
    values = _values(out, "АДРЕСА")
    assert len(values) == 5
    assert len({("вул." in v or "пров." in v or "м. " in v or "с. " in v) for v in values}) == 1


@pytest.mark.parametrize("text,expected", [
    ("ВЕРХОВНИЙ СУД\nм. Київ\nрішення Шевченківського районного суду м. Львова, Львівський апеляційний суд", "Львівська"),
    ("Київський апеляційний суд, Печерський районний суд міста Києва", "Київ"),
    ("відділ ДВС у Броварському районі Київської області", "Київська"),
    ("стягнення заборгованості за кредитним договором", None),
])
def test_doc_region(text, expected):
    assert doc_region(text) == expected


def test_one_date_format_per_document():
    raw = "ОСОБА_1 , ІНФОРМАЦІЯ_1 , та ОСОБА_2 , ІНФОРМАЦІЯ_2 , народився ІНФОРМАЦІЯ_3 , помер ІНФОРМАЦІЯ_4 ."
    kinds = {"iso": r"\d{4}-\d{2}-\d{2}", "numeric": r"\d{2}\.\d{2}\.\d{4}", "words": r"\d{1,2} [а-яіїє]+ \d{4}.*"}
    seen = set()
    for seed in range(12):
        dates = _values(_fill(raw, seed), "ІНФОРМАЦІЯ")
        styles = {next(k for k, rx in kinds.items() if re.fullmatch(rx, d)) for d in dates}
        assert len(styles) == 1, dates
        seen |= styles
    assert seen == set(kinds)


@pytest.mark.parametrize("left,expected", [
    ("ідентифікаційний номер транспортного засобу (VIN) ", "vin"),
    ("номер кузова НОМЕР_1, реєстраційний номер ", "vehicle"),
    ("РНОКПП НОМЕР_1, паспорт № ", "passport"),
    ("паспорт громадянина України серії ", "svidotstvo"),
    ("у виконавчому провадженні № ", "provadzhennia"),
    ("постанову про арешт коштів боржника у ВП № ", "provadzhennia"), ("за даними АСВП ", "provadzhennia"),
    ("код ЄДРПОУ ВП ", "generic"), ("РНОКПП НОМЕР_1, ВПО № ", "rnokpp"),
])
def test_number_subtype_takes_the_closest_cue(left, expected):
    assert number_subtype(left, "") == expected


def test_number_shapes():
    gen = NumberGenerator(random.Random(2))
    assert all(re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", gen.generate("vin")) for _ in range(50))
    assert all(re.fullmatch(r"\d{8,10}|[А-ЯІ]{2}/\d{4}", gen.generate("provadzhennia")) for _ in range(50))
    plates = {gen.generate("vehicle") for _ in range(200)}
    assert any(re.match(r"[A-Z]{2}", p) for p in plates) and any(re.match(r"[А-ЯІ]{2}", p) for p in plates)


def test_second_fill_of_a_document_has_other_values():
    text = normalize("Позов ОСОБА_1 до ОСОБА_2 , проживає: АДРЕСА_1 , РНОКПП НОМЕР_1 . Одеська область.")
    doc = {"id": "t", "text": text, "placeholders": find_placeholders(text)}
    first, again, second = Filler(seed=7).fill(doc), Filler(seed=7).fill(doc, fill=1), Filler(seed=7).fill(doc, fill=2)
    assert first == again and (first["fill"], second["fill"]) == (1, 2)
    assert first["text"] != second["text"] and first["text_pseudonymized"] == second["text_pseudonymized"]
    assert [l["pseudonym"] for l in first["labels"]] == [l["pseudonym"] for l in second["labels"]]
