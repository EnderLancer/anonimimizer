"""Regression checks for production annotation reuse and court signatures."""
import unittest
from unittest.mock import patch

import run
import ner_postprocess as NP
from regex_persons import persons
from ner_adapters import EntitySpan
from additional_tasks.audit.audit_registry import candidates, raw_candidates
from additional_tasks.audit.service_filter import mark_service_candidates

EXAMPLES = [
    "Суддя Первомайського міськрайонного суду\nХарківської області" + " " * 64 +
    "С.В.Библів\n\nВідмітки про виконання судового наказу",
    "судді Янченко А.В., присяжник Квашука О.Є., Мазуренко А.П., розглянувши",
    "Суддя А.В. Янченко\n\nПрисяжні: О.Є. Квашук\n\nА.П. Мазуренко",
    "Суддя Дубровицького\nрайонного суду" + " " * 77 + "К.М. Отупор",
    "у складі колегії суддів: головуючого - судді Чабаненко С.В. (доповідач),\n"
    "                               суддів: Білак С.В., Юрко І.В.,",
]


class PipelineAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tagger = run.load_tagger()

    def test_same_final_annotations_as_production(self):
        text = "Позивач Петренко Іван Іванович, РНОКПП: 1234567890, email person@example.com"
        start = text.index("Петренко")
        ner = [EntitySpan(start, start + len("Петренко Іван"), "Петренко Іван", "ОСОБА", 0.9)]
        for mode in ("regex", "full", "regex_crf"):
            with self.subTest(mode=mode):
                spans = ner if mode == "full" else []
                tagger = self.tagger if mode != "regex" else None
                expected = run.annotate(text, spans, "full" if mode == "regex_crf" else mode, tagger)
                actual = candidates(text, tagger, spans, mode)
                self.assertEqual([{key: c[key] for key in ("start", "end", "pseudonym")} for c in actual], expected)
                self.assertTrue(all(c["sources"] == ["pipeline"] for c in actual))
                self.assertTrue(all(text[c["start"]:c["end"]] == c["value"] for c in actual))

    def test_crf_person_errors_do_not_enter_production_audit(self):
        text = EXAMPLES[0]
        start = text.index("С.В.")
        end = text.index(" про")
        with patch("predict.predict_doc", return_value=[(start, end, "ОСОБА")]):
            production = candidates(text, self.tagger, mode="regex_crf")
            diagnostic = raw_candidates(text, self.tagger)
        self.assertEqual([c["value"] for c in production], ["С.В.Библів"])
        self.assertTrue(any("Відмітки" in c["value"] for c in diagnostic))

    def test_reported_signatures_are_service_names(self):
        expected_counts = [1, 3, 3, 1, 3]
        for text, count in zip(EXAMPLES, expected_counts):
            with self.subTest(text=text):
                result = candidates(text, self.tagger, mode="regex_crf")
                people = [c for c in result if c["type"] == "ОСОБА"]
                self.assertEqual(len(people), count)
                self.assertTrue(all(c["excluded"] for c in people), people)

    def test_juror_list_does_not_hide_participant_or_unrelated_prose(self):
        for text in (
            "Присяжні: О.Є. Квашук\n\nВідповідач А.П. Мазуренко",
            "Присяжні: О.Є. Квашук\n\nСуд викликав А.П. Мазуренко",
            "Заявник — присяжний А.П. Мазуренко",
        ):
            with self.subTest(text=text):
                people = candidates(text, mode="regex")
                self.assertFalse(next(c for c in people if "Мазуренко" in c["value"])["excluded"])

    def test_an_overbroad_ner_span_is_not_hidden_as_a_service_name(self):
        text = EXAMPLES[0]
        start, end = text.index("С.В."), text.index(" про")
        result = mark_service_candidates(text, [{"start": start, "end": end, "value": text[start:end], "type": "ОСОБА"}])
        # Reusing production logic cannot guarantee that NER itself is correct.
        # Never hide unrelated text merely because a title appears nearby.
        self.assertFalse(result[0]["excluded"])

    def test_merged_ner_judge_and_juror_lists(self):
        for text, value, role in (
            (EXAMPLES[1], "Квашука О.Є., Мазуренко А.П.", "присяжний"),
            (EXAMPLES[2], "Квашук\n\nА.П. Мазуренко", "присяжний"),
            (EXAMPLES[4], "Білак С.В., Юрко І.В.", "суддя"),
        ):
            with self.subTest(value=value):
                start = text.index(value)
                span = EntitySpan(start, start + len(value), value, "ОСОБА", 0.99)
                result = candidates(text, ner=[span], mode="model_regex")
                candidate = next(c for c in result if c["value"] == value)
                self.assertTrue(candidate["excluded"])
                self.assertEqual(candidate["role"], role)

    def test_merged_person_list_with_participant_stays_visible(self):
        text = "Судді Білак С.В., Юрко І.В.; відповідач Мазуренко А.П."
        value = "Білак С.В., Юрко І.В.; відповідач Мазуренко А.П."
        start = text.index(value)
        result = candidates(text, ner=[EntitySpan(start, len(text), value, "ОСОБА", 0.99)], mode="model_regex")
        self.assertFalse(next(c for c in result if c["value"] == value)["excluded"])

    def test_existing_pseudonyms_are_not_new_findings(self):
        text = "Позивач ОСОБА_1, email person@example.com"
        start = text.index("ОСОБА_1")
        span = EntitySpan(start, start + len("ОСОБА_1"), "ОСОБА_1", "ОСОБА", 1.0)
        result = candidates(text, ner=[span], mode="model_regex")
        self.assertFalse(any("ОСОБА_1" in c["value"] for c in result))
        self.assertTrue(any(c["value"] == "person@example.com" for c in result))

    def test_new_court_signatures_are_service_names(self):
        examples = (
            ("Суддя П`ятихатського районного суду\nДніпропетровської області Ю.В.Кудрявцева", ["Ю.В.Кудрявцева"]),
            ("Верховний Суд у складі судді-доповідача Касаційного адміністративного суду Мартинюк Н.М.,", ["Мартинюк Н.М."]),
            ("Судді | Л.І. Бившева |\n| Р.Ф. Ханова В.В. Хохуляк |", ["Л.І. Бившева", "Р.Ф. Ханова", "В.В. Хохуляк"]),
            ("Судді: І. О. Дундар\n\nЄ. В. Краснощоков\n\nВ. І. Крат", ["І. О. Дундар", "Є. В. Краснощоков", "В. І. Крат"]),
        )
        for text, expected in examples:
            with self.subTest(text=text):
                result = candidates(text, mode="regex")
                self.assertEqual([c["value"] for c in result], expected)
                self.assertTrue(all(c["excluded"] for c in result))
                self.assertEqual([text[s:e] for s, e, _ in persons(text)], expected)

    def test_merged_model_names_are_split_in_shared_postprocessing(self):
        for text, expected in (
            ("Р.Ф. Ханова В.В. Хохуляк", ["Р.Ф. Ханова", "В.В. Хохуляк"]),
            ("Краснощокова Є. В., Крата В. І.,", ["Краснощокова Є. В.", "Крата В. І."]),
        ):
            with self.subTest(text=text):
                spans = NP.process([EntitySpan(0, len(text), text, "ОСОБА", 0.9)], text)
                self.assertEqual([s.text for s in spans], expected)
                # A second processing pass must not merge separate names again.
                self.assertEqual([s.text for s in NP.process(spans, text)], expected)
                final = run.annotate(text, spans, "full", self.tagger)
                audited = candidates(text, self.tagger, spans, "full")
                self.assertEqual([{key: c[key] for key in ("start", "end", "pseudonym")} for c in audited], final)

    def test_model_person_span_with_prose_is_not_split(self):
        text = "Петренко І.П. повідомив про Коваль О.В."
        spans = NP.process([EntitySpan(0, len(text), text, "ОСОБА", 0.9)], text)
        self.assertEqual([s.text for s in spans], [text])

    def test_person_parts_still_assemble_and_addresses_are_unaffected(self):
        text = "Олена Соловей"
        spans = NP.process([EntitySpan(0, 5, "Олена", "ОСОБА", 0.9), EntitySpan(6, len(text), "Соловей", "ОСОБА", 0.9)], text)
        self.assertEqual([s.text for s in spans], [text])
        text = "вул. Київська, буд. 15"
        spans = [EntitySpan(0, 13, text[:13], "АДРЕСА", 0.9), EntitySpan(19, len(text), text[19:], "АДРЕСА", 0.9)]
        self.assertEqual(NP.process(spans, text), NP.assemble(spans, text))

    def test_judge_lists_do_not_swallow_unrelated_people_or_prose(self):
        texts = (
            "Судді: І. О. Дундар\n\nПозивач Є. В. Краснощоков",
            "Судді | Л.І. Бившева |\n| Р.Ф. Ханова | відповідач В.В. Хохуляк |",
            "Судді: І. О. Дундар\n\nВідомості про Є. В. Краснощоков",
        )
        for text in texts:
            with self.subTest(text=text):
                result = candidates(text, mode="regex")
                self.assertFalse(result[-1]["excluded"])

    def test_apostrophe_court_title_does_not_hide_a_participant_judge(self):
        text = "Відповідач — суддя П`ятихатського районного суду\nДніпропетровської області Ю.В.Кудрявцева"
        result = candidates(text, mode="regex")
        self.assertFalse(result[0]["excluded"])


if __name__ == "__main__":
    unittest.main()
