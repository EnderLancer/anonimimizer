import json
import sys

from pseudo import DATA

sys.path.insert(0, str(DATA))
import link_persons as lp  # noqa: E402

ROW = {"id": 7, "pseudonymized": "Позивач ОСОБА_1 (РНОКПП НОМЕР_1, АДРЕСА_1) до ОСОБА_2. Знову ОСОБА_1.",
       "annotations": [{"pseudonym": "ОСОБА_1 "}, {"pseudonym": "НОМЕР_1"}, {"pseudonym": "АДРЕСА_1"},
                       {"pseudonym": "ОСОБА_2"}, {"pseudonym": "ІНФОРМАЦІЯ_1"}]}  # ІНФОРМАЦІЯ_1 лише в анотаціях


def test_pseudonyms_follow_text_order_and_include_annotation_only_labels():
    assert lp.pseudonyms(ROW) == (["ОСОБА_1", "ОСОБА_2"], ["НОМЕР_1", "АДРЕСА_1", "ІНФОРМАЦІЯ_1"])


def test_validate_keeps_only_document_persons_and_fills_missing():
    warnings = []
    out = lp.validate("7", [{"pseudonym": " номер_1", "owners": ["особа_1", "ОСОБА_9", "ОСОБА_1"], "kind": "rnokpp"},
                            {"pseudonym": "АДРЕСА_1", "owners": [], "kind": "garage"},
                            {"pseudonym": "АДРЕСА_5", "owners": ["ОСОБА_1"], "kind": "other"}],
                      ["ОСОБА_1", "ОСОБА_2"], ["НОМЕР_1", "АДРЕСА_1", "ІНФОРМАЦІЯ_1"], warnings)
    assert out == {"НОМЕР_1": {"owners": ["ОСОБА_1"], "kind": "rnokpp"},
                   "АДРЕСА_1": {"owners": [], "kind": "other"},
                   "ІНФОРМАЦІЯ_1": {"owners": [], "kind": None}}
    assert len(warnings) == 4  # чужий власник, невідомий kind, зайвий псевдонім, пропущений псевдонім


def test_export_and_import_round_trip(tmp_path):
    tasks, answers = tmp_path / "tasks", tmp_path / "answers"
    assert lp.export_tasks([ROW, {"id": 8, "pseudonymized": "Лише ОСОБА_1.", "annotations": [{"pseudonym": "ОСОБА_1"}]}],
                           str(tasks)) == 1
    assert (tasks / "INSTRUCTIONS.md").read_text(encoding="utf-8").startswith(lp.SYSTEM)
    assert "Плейсхолдери даних, для яких потрібна відповідь: НОМЕР_1, АДРЕСА_1, ІНФОРМАЦІЯ_1" in (tasks / "7.md").read_text(encoding="utf-8")
    answers.mkdir()
    (answers / "7.json").write_text(json.dumps({"links": [{"pseudonym": "НОМЕР_1", "owners": ["ОСОБА_1"], "kind": "rnokpp"},
                                                          {"pseudonym": "АДРЕСА_1", "owners": ["ОСОБА_1"], "kind": "residence_address"},
                                                          {"pseudonym": "ІНФОРМАЦІЯ_1", "owners": [], "kind": "org"}]}),
                                    encoding="utf-8")
    links, warnings = {"meta": {}, "docs": {}}, []
    done, missing = lp.import_answers([ROW, {"id": 8, "pseudonymized": "Лише ОСОБА_1.", "annotations": [{"pseudonym": "ОСОБА_1"}]},
                                       {"id": 9, "pseudonymized": "ОСОБА_1, НОМЕР_1.", "annotations": [{"pseudonym": "ОСОБА_1"}, {"pseudonym": "НОМЕР_1"}]}],
                                      str(answers), links, warnings)
    assert (done, missing, warnings) == (1, ["9"], [])
    assert links["docs"]["7"]["data"]["АДРЕСА_1"] == {"owners": ["ОСОБА_1"], "kind": "residence_address"}
    assert links["docs"]["8"] == {"persons": ["ОСОБА_1"], "data": {}}  # документ без даних не потребує відповіді


def test_heuristic_links_nearest_preceding_person():
    assert lp.heuristic(ROW) == {"НОМЕР_1": {"ОСОБА_1"}, "АДРЕСА_1": {"ОСОБА_1"}}
