from silver.normalize import find_placeholders, normalize


def test_space_before_punct_removed():
    t = normalize("ОСОБА_1 , ІНФОРМАЦІЯ_1 , мешкає за адресою: АДРЕСА_1 , РНОКПП - НОМЕР_1 ,\n")
    assert " ," not in t and " ;" not in t
    assert t.startswith("ОСОБА_1, ІНФОРМАЦІЯ_1,")


def test_placeholders_located_after_normalization():
    t = normalize("Позов ОСОБА_12 до ОСОБА_3 .")
    ph = find_placeholders(t)
    assert [(p["type"], p["idx"]) for p in ph] == [("ОСОБА", 12), ("ОСОБА", 3)]
    for p in ph:
        assert t[p["start"]:p["end"]] == f"{p['type']}_{p['idx']}"
