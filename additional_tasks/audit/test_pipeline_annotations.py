"""Regression checks for production annotation reuse and court signatures."""
import unittest
from unittest.mock import patch

import run
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


if __name__ == "__main__":
    unittest.main()
