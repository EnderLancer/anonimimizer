import json
import sys

from pseudo import DATA

sys.path.insert(0, str(DATA))
import evaluate as ev  # noqa: E402
import evaluate_risk as er  # noqa: E402

TEXT = ("Позивач Іванов Іван Іванович (РНОКПП 1234567890) звернувся з позовом до Петренко Петра Петровича, "
        "м. Київ, вул. Хрещатик, 1, кв. 2. Іванов І.І. просив стягнути борг.")


def span(entity: str, label: str, occurrence: int = 0) -> ev.Span:
    start = -1
    for _ in range(occurrence + 1):
        start = TEXT.index(entity, start + 1)
    return ev.Span(start, start + len(entity), label.rsplit("_", 1)[0], label)


def ann(s: ev.Span) -> dict:
    return {"start": s.start, "end": s.end, "pseudonym": s.label}


GOLD = [span("Іванов Іван Іванович", "ОСОБА_1"), span("Іванов І.І.", "ОСОБА_1"), span("1234567890", "НОМЕР_1"),
        span("Петренко Петра Петровича", "ОСОБА_2"), span("м. Київ, вул. Хрещатик, 1, кв. 2", "АДРЕСА_1")]
GOLD_ROW = {"id": 1, "original": TEXT, "annotations": [ann(s) for s in GOLD]}


def links_file(tmp_path, data: dict) -> dict:
    p = tmp_path / "person_links.json"
    p.write_text(json.dumps({"meta": {"model": "test"}, "docs": {"1": {"persons": ["ОСОБА_1", "ОСОБА_2"], "data": data}}},
                            ensure_ascii=False), encoding="utf-8")
    return er.load_links(str(p))


LINKS = {"НОМЕР_1": {"owners": ["ОСОБА_1"], "kind": "rnokpp"}, "АДРЕСА_1": {"owners": ["ОСОБА_2"], "kind": "residence_address"}}


def test_covered_tokens_checks_digits_unlike_evaluate_py():
    text = "РНОКПП 7955136227, тел."
    s = ev.Span(7, 17, "НОМЕР", "НОМЕР_1")
    assert ev.covered(text, s, []) is True  # WORD з evaluate.py не бачить числа
    assert er.covered_tokens(text, s, []) is False
    assert er.covered_tokens(text, s, [ev.Span(7, 17, "НОМЕР", "НОМЕР_1")])
    assert er.covered_tokens(text, s, [ev.Span(7, 12, "ОСОБА", "ОСОБА_9")])  # половина символів, тип не важливий
    assert not er.covered_tokens(text, s, [ev.Span(7, 11, "НОМЕР", "НОМЕР_1")])  # менше половини
    assert er.covered_tokens(text, s, [ev.Span(7, 16, "НОМЕР", "НОМЕР_1")])  # відкрита остання цифра


def test_covered_tokens_agrees_with_evaluate_on_names():
    s = span("Іванов Іван Іванович", "ОСОБА_1")
    for pred in ([], [s], [ev.Span(s.start, s.start + 6, "ОСОБА", "ОСОБА_1")], [ev.Span(s.start, s.end - 1, "ОСОБА", "ОСОБА_1")]):
        assert er.covered_tokens(TEXT, s, pred) == ev.covered(TEXT, s, pred)


def test_person_leaks_through_her_own_data_only(tmp_path):
    # усі імена сховано, номер сховано, адресу ні: розкрита лише ОСОБА_2, власниця адреси
    pred = [{"id": 1, "annotations": [ann(s) for s in GOLD if s.label != "АДРЕСА_1"]}]
    res = er.evaluate_risk([GOLD_ROW], pred, links_file(tmp_path, LINKS), details=True)
    assert res["deanonymization_risk_by_name"] == 0.0
    assert res["deanonymization_risk"] == 0.5
    assert (res["persons_leaked_by_name"], res["persons_leaked_by_data_only"], res["persons_with_data"]) == (0, 1, 2)
    assert res["leaked_mentions_by_type"] == {"ОСОБА": 0, "АДРЕСА": 1, "НОМЕР": 0, "ІНФОРМАЦІЯ": 0}
    assert res["leaked_by_kind"] == {"residence_address": 1}
    assert [(l["person"], l["pseudonym"], l["kind"]) for l in res["leaks"]] == [("ОСОБА_2", "АДРЕСА_1", "residence_address")]
    assert res["data_pseudonyms"] == {"total": 2, "linked": 2, "unlinked": 0, "missing": 0}


def test_without_links_the_metric_is_the_name_risk():
    pred = [{"id": 1, "annotations": [ann(s) for s in GOLD if s.type == "ОСОБА" and s.label != "ОСОБА_2"]}]
    res = er.evaluate_risk([GOLD_ROW], pred, None)
    assert res["deanonymization_risk"] == res["deanonymization_risk_by_name"] == 0.5
    assert res["persons_with_data"] == 0
    assert res["data_pseudonyms"] == {"total": 2, "linked": 0, "unlinked": 2, "missing": 0}
    assert any("нічиїми" in w for w in res["warnings"])


def test_shared_data_counts_for_every_owner(tmp_path):
    links = links_file(tmp_path, {"АДРЕСА_1": {"owners": ["ОСОБА_1", "ОСОБА_2"], "kind": "residence_address"},
                                  "НОМЕР_1": {"owners": [], "kind": "rnokpp"}})
    pred = [{"id": 1, "annotations": [ann(s) for s in GOLD if s.type == "ОСОБА"]}]
    res = er.evaluate_risk([GOLD_ROW], pred, links)
    assert res["deanonymization_risk"] == 1.0
    assert res["leaked_mentions_by_type"]["АДРЕСА"] == 1  # згадку рахуємо один раз
    assert res["leaked_mentions_by_type"]["НОМЕР"] == 0  # нічий номер у ризик не входить
    assert res["data_pseudonyms"]["unlinked"] == 1


def test_links_are_normalized_and_unknown_owners_dropped(tmp_path):
    links = links_file(tmp_path, {" номер_1 ": {"owners": ["особа_1", "ОСОБА_7"], "kind": "rnokpp"}})
    pred = [{"id": 1, "annotations": [ann(s) for s in GOLD if s.type == "ОСОБА"]}]
    res = er.evaluate_risk([GOLD_ROW], pred, links)
    assert res["persons_leaked_by_data_only"] == 1
    assert res["data_pseudonyms"] == {"total": 2, "linked": 1, "unlinked": 0, "missing": 1}
    assert any("власників" in w for w in res["warnings"]) and any("немає у файлі" in w for w in res["warnings"])


def test_missing_submission_row_counts_as_fully_leaked(tmp_path):
    res = er.evaluate_risk([GOLD_ROW], [], links_file(tmp_path, LINKS))
    assert res["deanonymization_risk"] == res["deanonymization_risk_by_name"] == 1.0
    assert res["warnings"][0].startswith("1 рішень немає у відповіді")


def test_pseudonymized_text_format_is_accepted(tmp_path):
    out = ev.render(TEXT, GOLD)
    res = er.evaluate_risk([GOLD_ROW], [{"id": 1, "pseudonymized": out}], links_file(tmp_path, LINKS))
    assert res["deanonymization_risk"] == 0.0
