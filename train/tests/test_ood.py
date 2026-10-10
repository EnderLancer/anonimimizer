import json
import sys

from pseudo import DATA

sys.path.insert(0, str(DATA))
import build_ood as bo  # noqa: E402
import evaluate_ood as eo  # noqa: E402

TEXT = ("Віктор Ющенко прибув на вул. Шевченка, 5. Позивний «Шаман» мовчав. Ющенка зустрів Шевченко О. І.\n"
        "Телефон +38 (067) 123-45-67, email ivan@example.com. Народився 25 березня 1975 року.")


def test_occurrences_match_whole_words_and_respect_except():
    assert bo.occurrences(TEXT, "Ющенко") == [(7, 13)]  # не «Ющенка»
    assert bo.occurrences(TEXT, "Шевченко") == [(TEXT.index("Шевченко О."), TEXT.index("Шевченко О.") + 8)]
    assert bo.occurrences(TEXT, "Шевченка") == [(TEXT.index("Шевченка"), TEXT.index("Шевченка") + 8)]
    assert bo.occurrences(TEXT, "Шевченка", ["вул. Шевченка"]) == []
    assert bo.occurrences("Мар'яна і Мар’яна", "Мар'яна") == [(0, 7), (10, 17)]  # апострофи уніфіковано
    assert bo.occurrences("Жансема-художника і Vallejo's site", "Жансема") == [(0, 7)]  # дефіс не склеює
    assert bo.occurrences("Жансема-художника і Vallejo's site", "Vallejo") == [(20, 27)]  # англійське 's
    assert bo.occurrences("Ді\u0301дух", "Ді") == []  # знак наголосу — частина слова
    assert bo.occurrences(TEXT, "+38 (067) 123-45-67") == [(TEXT.index("+38"), TEXT.index("+38") + 19)]


def test_spans_longest_wins_and_pseudonyms_follow_entities(tmp_path):
    warnings = []
    ents = [{"text": "Віктор Ющенко", "type": "ОСОБА", "id": "p1"}, {"text": "Ющенко", "type": "ОСОБА", "id": "p1"},
            {"text": "Ющенка", "type": "ОСОБА", "id": "p1"}, {"text": "Шевченко О. І.", "type": "ОСОБА", "id": "p2"},
            {"text": "Шаман", "type": "ОСОБА", "kind": "callsign", "id": "p3"},
            {"text": "вул. Шевченка, 5", "type": "АДРЕСА", "kind": "event"},
            {"text": "+38 (067) 123-45-67", "type": "НОМЕР", "kind": "phone"},
            {"text": "ivan@example.com", "type": "ІНФОРМАЦІЯ", "kind": "email"},
            {"text": "25 березня 1975", "type": "ІНФОРМАЦІЯ", "kind": "dob", "id": "p1"},
            {"text": "немає такого", "type": "НОМЕР", "kind": "other"}]
    norm = [bo._normalize_entity(e, "d", warnings) for e in ents]
    spans = bo.spans_for(TEXT, norm, "d", warnings)
    assert [TEXT[s["start"]:s["end"]] for s in spans] == ["Віктор Ющенко", "вул. Шевченка, 5", "Шаман", "Ющенка", "Шевченко О. І.",
                                                           "+38 (067) 123-45-67", "ivan@example.com", "25 березня 1975"]
    anns, pseudo = bo.pseudonymize(TEXT, spans)
    assert [a["pseudonym"] for a in anns] == ["ОСОБА_1", "АДРЕСА_1", "ОСОБА_2", "ОСОБА_1", "ОСОБА_3", "НОМЕР_1", "ІНФОРМАЦІЯ_1", "ІНФОРМАЦІЯ_2"]
    assert anns[2]["kind"] == "callsign" and "kind" not in anns[0]
    assert pseudo.startswith("ОСОБА_1 прибув на АДРЕСА_1. Позивний «ОСОБА_2» мовчав. ОСОБА_1 зустрів ОСОБА_3\nТелефон НОМЕР_1")
    assert warnings == ["d: не знайдено в тексті НОМЕР 'немає такого'"]


def test_normalize_entity_rejects_bad_type_and_fixes_kind():
    warnings = []
    assert bo._normalize_entity({"text": "x", "type": "ORG"}, "d", warnings) is None
    e = bo._normalize_entity({"text": "x", "type": "номер", "kind": "garage", "person": "p9"}, "d", warnings)
    assert e == {"text": "x", "type": "НОМЕР", "kind": "other", "id": "p9", "except": []}
    assert len(warnings) == 2


def test_truncate_prefers_paragraph_then_sentence():
    para = "а" * 500 + "\n" + "б" * 2000 + "\n" + "в" * 1000
    cut, truncated = bo.truncate(para)
    assert truncated and cut == "а" * 500 + "\n" + "б" * 2000
    sent = "Речення перше. " * 100 + "Довге " * 600
    cut, truncated = bo.truncate(sent, limit=1600)
    assert truncated and cut.endswith("Речення перше.") and len(cut) <= 1600
    assert bo.truncate("коротко") == ("коротко", False)


def test_clean_strips_html_and_normalizes_whitespace():
    raw = "<p><img src='x.jpg'></p><p>Перший   абзац &laquo;тут&raquo;.</p><script>alert(1)</script><p>Другий<br>рядок</p>"
    assert bo.clean(raw) == "Перший абзац «тут».\n\nДругий\nрядок"
    assert bo.usable("short") == "short"
    assert bo.usable("Это русский текст " * 40) == "russian"
    assert bo.usable("Суд ВСТАНОВИВ таке у справі № 1 " * 20) == "court"
    assert bo.usable("Звичайний український текст про погоду і місто. " * 20) is None


def test_export_and_import_round_trip(tmp_path):
    cand = tmp_path / "cand.jsonl"
    rows = [{"id": "news:1", "original": TEXT, "source": "news", "subset": "war", "origin": "kobza", "url": "u", "title": None,
             "truncated": False, "license": "CC0-1.0 (Kobza)"},
            {"id": "wiki:2", "original": "Без персональних даних, лише гори і ріки.", "source": "wiki", "subset": "random"}]
    cand.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tasks = tmp_path / "tasks"
    bo.export(type("A", (), {"candidates": str(cand), "out_dir": str(tasks), "batches": 2})())
    assert (tasks / "INSTRUCTIONS.md").read_text(encoding="utf-8").startswith("# Інструкція розмітки")
    assert "=== ТЕКСТ ===\n" + TEXT + "\n=== КІНЕЦЬ ===" in (tasks / "news_1.md").read_text(encoding="utf-8")
    assert json.loads((tasks / "batches.json").read_text(encoding="utf-8")) == {"0": ["news:1"], "1": ["wiki:2"]}
    answers = tmp_path / "answers"
    answers.mkdir()
    (answers / "news_1.json").write_text(json.dumps({"entities": [{"text": "Віктор Ющенко", "type": "ОСОБА", "id": "p1"},
                                                                  {"text": "ivan@example.com", "type": "ІНФОРМАЦІЯ", "kind": "email"}]}),
                                         encoding="utf-8")
    (answers / "wiki_2.json").write_text('{"entities": []}', encoding="utf-8")
    warnings = []
    out = bo.import_answers(rows, str(answers), warnings)
    assert warnings == []
    assert [r["id"] for r in out] == ["news:1", "wiki:2"]
    assert out[0]["annotations"] == [{"start": 0, "end": 13, "entity": "Віктор Ющенко", "pseudonym": "ОСОБА_1"},
                                     {"start": TEXT.index("ivan@"), "end": TEXT.index("ivan@") + 16, "entity": "ivan@example.com",
                                      "pseudonym": "ІНФОРМАЦІЯ_1", "kind": "email"}]
    assert "truncated" not in out[0] and out[0]["license"] == "CC0-1.0 (Kobza)"
    assert out[1]["annotations"] == [] and out[1]["pseudonymized"] == out[1]["original"]
    assert "| news | 1 | 1 | 0 | 0 | 1 |" in bo.report(out)
    assert "Документів без сутностей: 1." in bo.report(out)


def test_evaluate_by_source_with_extra_metrics():
    gold = [{"id": "news:1", "source": "news", "original": "Іван Петренко і Марія Коваль. Тел. 0671234567.",
             "annotations": [{"start": 0, "end": 13, "pseudonym": "ОСОБА_1"}, {"start": 16, "end": 28, "pseudonym": "ОСОБА_2"},
                             {"start": 35, "end": 45, "pseudonym": "НОМЕР_1", "kind": "phone"}]},
            {"id": "wiki:2", "source": "wiki", "original": "Гори і ріки без імен, зовсім нічого.", "annotations": []}]
    pred = [{"id": "news:1", "annotations": [{"start": 0, "end": 13, "pseudonym": "ОСОБА_1"}, {"start": 16, "end": 28, "pseudonym": "ОСОБА_1"}]},
            {"id": "wiki:2", "annotations": [{"start": 0, "end": 4, "pseudonym": "ОСОБА_1"}]}]
    res = eo.evaluate_all(gold, {"m": pred}, "source", n_boot=20, seed=1)
    assert res["groups"] == ["news", "wiki"]
    m = res["submissions"]["m"]
    assert m["by_group"]["news"]["ner_relaxed"]["per_type"]["ОСОБА"]["f1"] == 1.0
    assert m["by_group"]["news"]["ner_relaxed"]["per_type"]["НОМЕР"]["recall"] == 0.0
    assert m["by_group"]["wiki"]["ner_relaxed"]["per_type"]["ОСОБА"]["precision"] == 0.0
    assert m["false_positives"]["ОСОБА"] == {"docs": 1, "false_positives": 1, "per_1k_chars": 1000 / len(gold[1]["original"])}
    assert m["recall_by_kind"] == {"НОМЕР/phone": {"total": 1, "found": 0, "recall": 0.0},
                                   "ОСОБА/—": {"total": 2, "found": 2, "recall": 1.0}}
    lo, hi = m["macro_f1_ci95"]
    assert 0.0 <= lo <= hi <= 1.0
    text = eo.render(res)
    assert "| news | 1 | 1.0 (1.0 / 1.0) |" in text and "| усього | 2 |" in text
    assert "ОСОБА/— 1.0 (2/2)" in text
