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

    def test_service_addresses_from_wechsel_report(self):
        examples = (
            ("ГОСПОДАРСЬКИЙ СУД\nДНІПРОПЕТРОВСЬКОЇ ОБЛАСТІ\n", "вул. Володимира Винниченка 1, м. Дніпро, 49027"),
            ("Справа № 445/421/15-к\n\nПеремишлянський районний суд ", "Львівської області\nвул.Галицька,67а"),
            ("Копію направити Шепетівській окружній прокуратурі (", "вул. Сергія Оврашка, 43, м. Шепетівка, 30400"),
            ("Засідання відбудеться у приміщенні Господарського суду Київської області (", "м. Київ, вул. С.Петлюри, 16/108"),
            ("Судове засідання в залі суду на 28 лютого 2024 року на 09.30 год. за адресою: ", "вул. Шевченка, 29"),
            ("Львівського інституту судових експертиз Міністерства юстиції України, який розташований за адресою: ", "79024, м. Львів, вул. Липинського, 54"),
            ("Своєю робочою адресою адвокат ОСОБА_6 зазначає: ", "01001, м. Київ, вул. Заньковецької, будинок 4, офіс 3"),
            ("Відповідач – Студениківська сільська рада Бориспільського району Київської області, юридична адреса: ", "вул.Переясласька, 19"),
            ("Позивач: Приватне підприємство «ДНК», адреса: ", "вул. Промислова, буд. 23"),
        )
        for prefix, value in examples:
            with self.subTest(prefix=prefix):
                text = prefix + value
                result = detect(text, [value], "АДРЕСА")[0]
                self.assertTrue(result["excluded"])
                self.assertTrue(result["exclusion_reason"])
                self.assertEqual(text[result["start"]:result["end"]], value)

    def test_nearby_institution_does_not_exclude_personal_or_property_address(self):
        prefixes = (
            "Суд встановив, що ОСОБА_1 проживає за адресою: ",
            "Адреса проживання судді: ",
            "Прокуратура повідомила про обшук за адресою: ",
            "АТ «БАНК КРЕДИТ ДНІПРО», склад рулонів, що розташований за адресою: ",
            "Позивач — адвокат Петренко Іван Іванович (",
            "Суд розташований за адресою: АДРЕСА_1. Позивач мешкає за адресою: ",
            "Адреса суду: АДРЕСА_1; місце події: ",
            "Начальник окружної прокуратури ОСОБА_3 (",
            "АТ «ПУМБ», близько 09 год. 55 хв., знаходячись у приміщенні магазину «Київстар», за адресою: ",
            "Департаменту патрульної поліції капітан поліції Петренко Іван Іванович (місцезнаходження: ",
        )
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                self.assertFalse(detect(prefix + "вул. Миру, 10", ["вул. Миру, 10"], "АДРЕСА")[0]["excluded"])

    def test_contained_address_fragments_share_exclusion(self):
        text = "Судове засідання відбудеться у Верховному Суді за адресою: м. Київ, вул. Миру, 10."
        result = detect(text, ["м. Київ, вул. Миру, 10", "м. Київ", "вул. Миру, 10"], "АДРЕСА")
        self.assertTrue(all(c["excluded"] for c in result))

    def test_same_address_is_classified_per_occurrence(self):
        text = "Адреса суду: вул. Миру, 10. Адреса проживання: вул. Миру, 10."
        result = candidates(text)
        self.assertEqual([c["excluded"] for c in result if c["type"] == "АДРЕСА"], [True, False])

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

    def test_service_number_categories_and_layout(self):
        examples = (
            ("Справа № 123/456/24-к", "123/456/24-к", "Номер судової справи"),
            ("у справі\n№\u00a0123/456/24", "123/456/24", "Номер судової справи"),
            ("Судове провадження № 123456", "123456", "Номер судового провадження"),
            ("Провадження № 2-кп/123/456/24", "2-кп/123/456/24", "Номер судового провадження"),
            ("Наказ № 123456", "123456", "Номер ухвали, постанови або наказу"),
            ("Ухвала № 123456", "123456", "Номер ухвали, постанови або наказу"),
            ("Постанова\n№ 123456", "123456", "Номер ухвали, постанови або наказу"),
            ("Тел. суду: (044) 123-45-67", "(044) 123-45-67", "Телефон або факс установи за явним контекстом"),
            ("Факс прокуратури: 044 123 45 67", "044 123 45 67", "Телефон або факс установи за явним контекстом"),
        )
        for text, value, reason in examples:
            with self.subTest(text=text):
                c = detect(text, [value], "НОМЕР")[0]
                self.assertTrue(c["excluded"])
                self.assertEqual(c["exclusion_reason"], reason)
                self.assertEqual(text[c["start"]:c["end"]], value)

    def test_private_and_ambiguous_numbers_remain_visible(self):
        examples = (
            ("РНОКПП: 1234567890", "1234567890"),
            ("Паспорт: АА123456", "АА123456"),
            ("Картка: 1234 5678 9012 3456", "1234 5678 9012 3456"),
            ("VIN: ABC12345678901234", "ABC12345678901234"),
            ("IMEI: 123456789012345", "123456789012345"),
            ("ДНЗ: АА1234ВВ", "АА1234ВВ"),
            ("Виконавче провадження № 123/456/24", "123/456/24"),
            ("ВП № 12345678", "12345678"),
            ("Кримінальне провадження № 123/456/24", "123/456/24"),
            ("Провадження № 12345678", "12345678"),
            ("Кадастровий номер: 1234567890:12:345:6789", "1234567890:12:345:6789"),
            ("Договір № 123456", "123456"),
            ("Номер запису № 123456", "123456"),
            ("Квитанція № 123456", "123456"),
            ("Суд повідомив телефон позивача: +380501234567", "+380501234567"),
            ("Справа № НОМЕР_1. РНОКПП: 1234567890", "1234567890"),
            ("Справа № 123/456/24. 1234567890", "1234567890"),
            ("Справа. № 123456", "123456"),
            ("Справжній номер: 123456", "123456"),
            ("Справа 12.10.2024", "12"),
        )
        for text, value in examples:
            with self.subTest(text=text):
                self.assertFalse(detect(text, [value], "НОМЕР")[0]["excluded"])

    def test_number_fragments_and_repeats_are_occurrence_local(self):
        text = "Справа № 1234567890. РНОКПП: 1234567890."
        spans = []
        for start, end in ((text.index("1234567890"), text.index("1234567890") + 10),
                           (text.index("1234567890") + 2, text.index("1234567890") + 6),
                           (text.rindex("1234567890"), text.rindex("1234567890") + 10)):
            spans.append({"start": start, "end": end, "value": text[start:end], "type": "НОМЕР"})
        self.assertEqual([c["excluded"] for c in mark_service_candidates(text, spans)], [True, True, False])

    def test_crossing_number_candidate_is_not_excluded(self):
        text = "Справа № 123456, паспорт АА123456"
        self.assertFalse(detect(text, ["123456, паспорт АА123456"], "НОМЕР")[0]["excluded"])

    def test_treasury_account_partial_ner_and_personal_account(self):
        account = "UA12" + "899998" + "0" * 19
        text = "Рахунок: " + account
        self.assertTrue(detect(text, [account[8:18]], "НОМЕР")[0]["excluded"])
        spaced = " ".join((account[:4], account[4:10], account[10:]))
        self.assertTrue(detect(spaced, [spaced], "НОМЕР")[0]["excluded"])
        self.assertTrue(detect("Рахунок – " + account + "Отримувач: Казначейство", [account], "НОМЕР")[0]["excluded"])
        self.assertFalse(detect(account + "1", [account], "НОМЕР")[0]["excluded"])
        personal = "UA12" + "300001" + "0" * 19
        self.assertFalse(detect("Суд повідомив IBAN: " + personal, [personal], "НОМЕР")[0]["excluded"])

    def test_audit_filters_partial_ner_number_after_collection(self):
        text = "Справа № 123/456/24. РНОКПП: 1234567890"
        s = text.index("456")
        result = candidates(text, ner=[EntitySpan(s, s + 3, "456", "НОМЕР", 0.99)])
        self.assertTrue(next(c for c in result if c["sources"] == ["ner"])["excluded"])
        self.assertFalse(next(c for c in result if c["value"] == "1234567890")["excluded"])

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
