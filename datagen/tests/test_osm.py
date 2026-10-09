import pytest

from silver.osm import district_name, norm_house, parse_street, region_name


@pytest.mark.parametrize("raw,expected", [
    ("вулиця Богдана Хмельницького", ("вул.", "Богдана Хмельницького")),
    ("Садова вулиця", ("вул.", "Садова")),
    ("Хрещатик", ("вул.", "Хрещатик")),
    ("пр-т Науки", ("просп.", "Науки")),
    ("1-й провулок Шевченка", ("пров.", "1-й Шевченка")),
    ("В'їзд Кузнецький", ("в’їзд", "Кузнецький")),
    ("вулиця 8 Березня", ("вул.", "8 Березня")),
    ("улица Ленина", None),            # Russian
    ("Первомайская улица", None),
    ("набережна Дніпровська", None),   # a kind of way the generator has no wording for
    ("вулиця", None),
    ("12", None),
])
def test_parse_street(raw, expected):
    assert parse_street(raw) == expected


def test_norm_house():
    assert [norm_house(h) for h in ["12", "12а", "12-А", "12 Б", "12/3"]] == ["12", "12А", "12-А", "12Б", "12/3"]
    assert [norm_house(h) for h in ["12к2", "12-14", "1234", "буд. 5"]] == [None] * 4


def test_region_and_district_names():
    assert region_name("Київська область") == "Київська" and region_name("Київ") == "Київ"
    assert region_name("Автономна Республіка Крим") is None
    assert district_name("Броварський район") == "Броварський" and district_name("Бровари") is None
