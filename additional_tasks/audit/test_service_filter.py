"""Run: python -m unittest additional_tasks.audit.test_service_filter"""
import tempfile
import unittest
from pathlib import Path

from additional_tasks.audit.audit_registry import candidates, write_report
from additional_tasks.audit.service_filter import mark_service_candidates
from ner_adapters import EntitySpan


def detect(text, values, kind="ОСОБА"):
    spans = []
    for value in values:
        start = text.index(value)
        spans.append({"start": start, "end": start + len(value), "type": kind, "value": value})
    return mark_service_candidates(text, spans)


class ServiceFilterTests(unittest.TestCase):
    def test_roles_and_forms(self):
        for role in ("суддя", "суддя-доповідач", "головуючого судді", "адвокатом", "приватним нотаріусом",
                     "прокурора", "секретарем судового засідання"):
            with self.subTest(role=role):
                result = detect(role + " Петренко Іван Іванович", ["Петренко Іван Іванович"])
                self.assertTrue(result[0]["excluded"])
                self.assertTrue(result[0]["role"])

    def test_institution_and_judge_list(self):
        text = "суддів Київського районного суду: Петренко І. П., Іваненко О. В.; позивач Коваль М. І."
        result = detect(text, ["Петренко І. П.", "Іваненко О. В.", "Коваль М. І."])
        self.assertEqual([c["excluded"] for c in result], [True, True, False])

    def test_suffix(self):
        result = detect("Петренко Іван Іванович, приватний нотаріус", ["Петренко Іван Іванович"])
        self.assertTrue(result[0]["excluded"])

    def test_judge_after_court_city(self):
        text = "Суддя Шевченківського районного суду м. Чернівці \nПетренко В.Г. розглянувши позов ОСОБА_1"
        result = candidates(text)
        self.assertTrue(next(c for c in result if c["value"] == "Петренко В.Г.")["excluded"])

    def test_participants_and_nearby_titles(self):
        for prefix in ("відповідач — адвокат ", "позивач нотаріус ",
                       "суддя заслухав пояснення ", "адвокат представляє інтереси "):
            with self.subTest(prefix=prefix):
                self.assertFalse(detect(prefix + "Коваль Іван Іванович", ["Коваль Іван Іванович"])[0]["excluded"])

    def test_additional_participant_roles_and_cases(self):
        roles = ("Заявник", "заявника", "заявнику", "заявником", "заявники", "заявників",
                 "Заявниця", "заявниці", "заявницю", "заявницею",
                 "Стягувач", "стягувача", "стягувачу", "стягувачем", "стягувачі", "стягувачів",
                 "Скаржник", "скаржника", "скаржнику", "скаржником", "скаржниця", "скаржницею",
                 "Третя особа", "третьої особи", "третій особі", "третю особу", "третьою особою",
                 "треті особи", "третіх осіб", "третім особам", "третіми особами")
        for role in roles:
            with self.subTest(role=role):
                text = f"{role} — адвокат Петренко Іван Іванович."
                result = detect(text, ["Петренко Іван Іванович"])
                self.assertFalse(result[0]["excluded"])
                self.assertIsNone(result[0]["role"])

    def test_participant_context_across_layout_and_city_abbreviation(self):
        prefixes = ("Позивач — адвокат\n", "Позивач — адвокат\r\n",
                    "Позивач — адвокат (м. Київ) ", "Позивач — адвокат\n(м. Київ)\n",
                    "Заявник | адвокат | ", "Третя\nособа — нотаріус\n")
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                text = prefix + "Петренко Іван Іванович подав позов."
                result = candidates(text)
                person = next(c for c in result if c["value"] == "Петренко Іван Іванович")
                self.assertFalse(person["excluded"])

    def test_partial_ner_keeps_participant_context(self):
        text = "Позивач — адвокат (м. Київ)\nПетренко Іван Іванович подав позов."
        value = "Іван Іванович"
        s = text.index(value)
        result = candidates(text, ner=[EntitySpan(s, s + len(value), value, "ОСОБА", 0.99)])
        self.assertFalse(next(c for c in result if c["sources"] == ["ner"])["excluded"])

    def test_representatives_of_participants(self):
        prefixes = ("Представник позивача — адвокат ", "Представник відповідача – адвокат ",
                    "Представником позивача — адвокатом ", "Представниця відповідача — адвокат ",
                    "Представник інтересів заявника — адвокат ", "Представник третьої особи — адвокат ",
                    "Представник\nпозивача — адвокат (м. Київ) ", "Адвокат позивача ")
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                text = prefix + "Петренко Іван Іванович."
                result = detect(text, ["Петренко Іван Іванович"])
                self.assertTrue(result[0]["excluded"])
                self.assertEqual(result[0]["role"], "адвокат / захисник")

    def test_party_and_their_lawyer_are_distinct_people(self):
        text = "Позивач Коваль Іван Іванович, адвокат Петренко Олег Петрович."
        result = detect(text, ["Коваль Іван Іванович", "Петренко Олег Петрович"])
        self.assertEqual([c["excluded"] for c in result], [False, True])

    def test_participant_context_stops_at_sentence_boundary(self):
        text = "Позивач подав позов. Адвокат Петренко Іван Іванович представляє його."
        self.assertTrue(detect(text, ["Петренко Іван Іванович"])[0]["excluded"])

    def test_table_signatures_preserve_offsets(self):
        examples = (("Суддя | |О.В. Ліпинський", "О.В. Ліпинський"),
                    ("Суддя |Машевська О.П.", "Машевська О.П."),
                    ("Суддя:||С.М. Ченцова", "С.М. Ченцова"),
                    ("Представник позивача – адвокат Пінчук К.О.", "Пінчук К.О."))
        for text, value in examples:
            with self.subTest(text=text):
                result = candidates(text)
                person = next(c for c in result if c["value"] == value)
                self.assertTrue(person["excluded"])
                self.assertEqual(person["start"], text.index(value))
                self.assertEqual(person["end"], text.index(value) + len(value))
                self.assertEqual(text[person["start"]:person["end"]], value)

    def test_new_participant_roles_prevent_repeat_propagation(self):
        text = ("Адвокат Петренко Іван Іванович.\n"
                "Заявник — адвокат (м. Київ)\nПетренко Іван Іванович.\n"
                "Петренко Іван Іванович подав заяву.")
        result = candidates(text)
        self.assertEqual([c["excluded"] for c in result if c["type"] == "ОСОБА"], [True, False, False])

    def test_exact_repeats_and_partial_ner(self):
        text = "Суддя Петренко Іван Іванович.\nПетренко Іван Іванович підписав рішення.\nПетренко подав заяву."
        result = candidates(text)
        people = [c for c in result if c["type"] == "ОСОБА"]
        self.assertEqual(len(people), 2)
        self.assertTrue(all(c["excluded"] for c in people))
        self.assertIn("Повтор", people[1]["exclusion_reason"])
        result = detect(text, ["Петренко Іван Іванович", "Іван Іванович", "Петренко"])
        self.assertEqual([c["excluded"] for c in result], [True, True, True])
        # An isolated surname elsewhere is never propagated from the full name.
        s = text.rindex("Петренко")
        result = mark_service_candidates(text, [
            {"start": 6, "end": 30, "type": "ОСОБА", "value": text[6:30]},
            {"start": s, "end": s + 8, "type": "ОСОБА", "value": "Петренко"}])
        self.assertFalse(result[1]["excluded"])

    def test_conflicting_repeat(self):
        text = "Суддя Петренко І. П.; відповідач Петренко І. П.;\nПетренко І. П."
        result = candidates(text)
        self.assertEqual([c["excluded"] for c in result if c["type"] == "ОСОБА"], [True, False, False])

    def test_ner_partial_and_private_address(self):
        text = "Суддя Петренко Іван Іванович. Адреса проживання: вул. Миру, 10. ОСОБА_1"
        value = "Іван Іванович"
        s = text.index(value)
        result = candidates(text, ner=[EntitySpan(s, s + len(value), value, "ОСОБА", 0.99)])
        self.assertTrue(next(c for c in result if c["sources"] == ["ner"])["excluded"])
        self.assertTrue(any(c["type"] == "АДРЕСА" and not c["excluded"] for c in result))
        self.assertFalse(any("ОСОБА_" in c["value"] for c in result))

    def test_addresses(self):
        for prefix, expected in (("Адреса суду: ", True), ("Суд встановив, що проживає за адресою: ", False),
                                 ("Адреса проживання нотаріуса: ", False)):
            with self.subTest(prefix=prefix):
                self.assertEqual(detect(prefix + "вул. Миру, 10", ["вул. Миру, 10"], "АДРЕСА")[0]["excluded"], expected)

    def test_official_hosts(self):
        for value, expected in (("https://reyestr.court.gov.ua/Review/123", True),
                                ("inbox@ki.court.gov.ua", True), ("https://court.gov.ua.evil.example", False),
                                ("person@example.com", False), ("http://[broken", False)):
            with self.subTest(value=value):
                self.assertEqual(detect(value, [value], "ІНФОРМАЦІЯ")[0]["excluded"], expected)

    def test_accounts_and_numbers(self):
        treasury = "UA12" + "899998" + "0" * 19
        personal = "UA12" + "300001" + "0" * 19
        self.assertTrue(detect(treasury, [treasury], "НОМЕР")[0]["excluded"])
        self.assertFalse(detect(personal, [personal], "НОМЕР")[0]["excluded"])
        self.assertTrue(detect("номер справи № 123456", ["123456"], "НОМЕР")[0]["excluded"])
        self.assertFalse(detect("РНОКПП: 1234567890", ["1234567890"], "НОМЕР")[0]["excluded"])

    def test_institutional_contacts(self):
        self.assertTrue(detect("Телефон канцелярії: +380501234567", ["+380501234567"], "НОМЕР")[0]["excluded"])
        self.assertFalse(detect("Телефон адвоката: +380501234567", ["+380501234567"], "НОМЕР")[0]["excluded"])

    def test_report_contains_exclusions_and_safe_json(self):
        text = "Суддя Петренко І. П. </script>"
        result = candidates(text)
        self.assertTrue(any(c["excluded"] for c in result))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.html"
            write_report(path, {"metadata": {}, "documents": [{"text": text, "candidates": result}]})
            html = path.read_text()
        self.assertIn('id="show-service"', html)
        self.assertIn('"excluded": true', html)
        self.assertIn(r"\u003c/script>", html)


if __name__ == "__main__":
    unittest.main()
