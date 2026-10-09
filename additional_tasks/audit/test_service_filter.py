"""Run: python -m unittest additional_tasks.audit.test_service_filter"""
import re
import tempfile
import unittest
from pathlib import Path

from additional_tasks.audit.audit_registry import raw_candidates, write_report
from additional_tasks.audit.service_filter import mark_service_candidates
from additional_tasks.audit.service_filter import service_information_reason
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
        result = raw_candidates(text)
        self.assertTrue(next(c for c in result if c["value"] == "Петренко В.Г.")["excluded"])

    def test_complete_court_descriptions_with_geography(self):
        examples = (
            ("Києво-Святошинського районного суду Київської області ", "Омельченко Мирослава Миколаївна"),
            ("Кам`янець-Подільського\n       міськрайонного суду Хмельницької області" + " " * 32, "Шевцова Л.М."),
            ("Вільнянського районного суду Запорізької області ", "Мануйлова Н. Ю."),
            ("Дніпропетровського окружного адміністративного суду ", "Кальник В.В."),
        )
        for court, name in examples:
            for apostrophe in ("`", "'", "’", "ʼ"):
                court_variant = court.replace("`", apostrophe)
                text = "Суддя " + court_variant + name + ","
                with self.subTest(text=text):
                    result = raw_candidates(text)
                    person = next(c for c in result if c["value"] == name)
                    self.assertTrue(person["excluded"])
                    self.assertEqual(person["role"], "суддя")
                    self.assertEqual(text[person["start"]:person["end"]], name)
                    # Model fragments must inherit the full name's role.
                    self.assertTrue(detect(text, [name.split()[0]])[0]["excluded"])
                    for party in ("Позивач", "Відповідач", "Свідок"):
                        participant = raw_candidates(party + " — " + text)
                        self.assertFalse(next(c for c in participant if c["value"] == name)["excluded"])
                    for prose in ("заслухав пояснення ", "повідомив про ", "Свідок "):
                        narrative = "Суддя " + court_variant + prose + name
                        self.assertFalse(detect(narrative, [name])[0]["excluded"])

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
                result = raw_candidates(text)
                person = next(c for c in result if c["value"] == "Петренко Іван Іванович")
                self.assertFalse(person["excluded"])

    def test_partial_ner_keeps_participant_context(self):
        text = "Позивач — адвокат (м. Київ)\nПетренко Іван Іванович подав позов."
        value = "Іван Іванович"
        s = text.index(value)
        result = raw_candidates(text, ner=[EntitySpan(s, s + len(value), value, "ОСОБА", 0.99)])
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
                result = raw_candidates(text)
                person = next(c for c in result if c["value"] == value)
                self.assertTrue(person["excluded"])
                self.assertEqual(person["start"], text.index(value))
                self.assertEqual(person["end"], text.index(value) + len(value))
                self.assertEqual(text[person["start"]:person["end"]], value)

    def test_new_participant_roles_prevent_repeat_propagation(self):
        text = ("Адвокат Петренко Іван Іванович.\n"
                "Заявник — адвокат (м. Київ)\nПетренко Іван Іванович.\n"
                "Петренко Іван Іванович подав заяву.")
        result = raw_candidates(text)
        self.assertEqual([c["excluded"] for c in result if c["type"] == "ОСОБА"], [True, False, False])

    def test_judge_signatures_with_model_title_and_partial_initials(self):
        examples = (
            ("Постанова оскарженню не підлягає.\n\nСуддя " + " \xa0" * 16 + " Н.В.Домарєва",
             "Суддя " + " \xa0" * 16 + " Н.В.Домарєва", "Н.В.Домарєва"),
            ("Суддя" + "\xa0" * 95 + "С.І. Крамар",
             "Суддя" + "\xa0" * 95 + "С.І. Крамар", "С.І. Крамар"),
            ("Ухвала підписана суддею, оскарженню не підлягає.\n\nСуддя Т.А. Марків",
             "Суддя Т.А.", "Т.А. Марків"),
            ("Суддя Г. М. Мачульський\n\nСуддя Л. І. Рогач",
             "Суддя Л. І.", "Л. І. Рогач"),
        )
        for text, model_value, name in examples:
            with self.subTest(name=name):
                s = text.index(model_value)
                span = EntitySpan(s, s + len(model_value), model_value, "ОСОБА", 0.99)
                result = raw_candidates(text, ner=[span])
                for value in (model_value, name):
                    c = next(c for c in result if c["value"] == value)
                    self.assertTrue(c["excluded"])
                    self.assertEqual(c["role"], "суддя")
                    self.assertEqual(text[c["start"]:c["end"]], value)

    def test_signature_name_survives_model_span_with_trailing_prose(self):
        text = "Суддя" + "\xa0" * 95 + "С.І. Крамар\n\nВіддруков. 4 прим."
        model_value = text[:text.index(". 4")]
        result = raw_candidates(text, ner=[EntitySpan(0, len(model_value), model_value, "ОСОБА", 0.99)])
        name = next(c for c in result if c["value"] == "С.І. Крамар")
        self.assertTrue(name["excluded"])
        # The extra prose is not proven to be service person data.
        self.assertFalse(next(c for c in result if c["value"] == model_value)["excluded"])

    def test_model_title_does_not_override_participant_or_crossing_person(self):
        for prefix in ("Відповідач — суддя ", "Позивач — суддя\n"):
            text = prefix + "Л. І. Рогач"
            model_value = text[text.index("суддя"):]
            s = text.index(model_value)
            result = raw_candidates(text, ner=[EntitySpan(s, len(text), model_value, "ОСОБА", 0.99)])
            self.assertTrue(all(not c["excluded"] for c in result if c["type"] == "ОСОБА"))
        text = "Суддя Л. І. Рогач; відповідач Коваль Іван Іванович"
        result = raw_candidates(text, ner=[EntitySpan(0, len(text), text, "ОСОБА", 0.99)])
        self.assertTrue(next(c for c in result if c["value"] == "Л. І. Рогач")["excluded"])
        self.assertFalse(next(c for c in result if c["value"] == text)["excluded"])
        self.assertFalse(next(c for c in result if c["value"] == "Коваль Іван Іванович")["excluded"])

    def test_exact_repeats_and_partial_ner(self):
        text = "Суддя Петренко Іван Іванович.\nПетренко Іван Іванович підписав рішення.\nПетренко подав заяву."
        result = raw_candidates(text)
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
        result = raw_candidates(text)
        self.assertEqual([c["excluded"] for c in result if c["type"] == "ОСОБА"], [True, False, False])

    def test_ner_partial_and_private_address(self):
        text = "Суддя Петренко Іван Іванович. Адреса проживання: вул. Миру, 10. ОСОБА_1"
        value = "Іван Іванович"
        s = text.index(value)
        result = raw_candidates(text, ner=[EntitySpan(s, s + len(value), value, "ОСОБА", 0.99)])
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
        result = raw_candidates(text)
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

    def test_service_information_categories(self):
        examples = (
            ("E-mail прокуратури: office@example.com", "office@example.com"),
            ("Телефон канцелярії: +380501234567", "+380501234567"),
            ("Електронна пошта міністерства:\n office@example.com", "office@example.com"),
            ("Офіційний сайт: https://minjust.gov.ua/", "https://minjust.gov.ua/"),
            ("Веб-адреса: http://reyestr.court.gov.ua/.", "http://reyestr.court.gov.ua/."),
            ("Сайт: court.gov.ua/", "court.gov.ua/"),
            ("https://zakon.rada.gov.ua/laws/show/1402-19#Text", "https://zakon.rada.gov.ua/laws/show/1402-19#Text"),
            ("Дата ухвалення рішення: 12.03.2024", "12.03.2024"),
            ("Ухвала від 12 березня 2024 року", "12 березня 2024 року"),
            ("Дата судового засідання — 2024-03-12", "2024-03-12"),
            ("Судове засідання призначено на 12.03.2024", "12.03.2024"),
            ("Орган, що видав документ: Міністерство юстиції України", "Міністерство юстиції України"),
            ("Найменування установи: Міністерство юстиції України", "Міністерство юстиції України"),
            ("Верховний Суд\nУХВАЛА", "Верховний Суд"),
            ("Міністерство юстиції України\nРеквізити установи", "Міністерство юстиції України"),
        )
        for text, value in examples:
            with self.subTest(text=text):
                c = detect(text, [value], "ІНФОРМАЦІЯ")[0]
                self.assertTrue(c["excluded"])
                self.assertTrue(c["exclusion_reason"])
                self.assertEqual(text[c["start"]:c["end"]], value)

    def test_private_and_ambiguous_information(self):
        examples = (
            ("Дата народження: 12.03.1980", "12.03.1980"),
            ("Дата смерті: 12.03.2024", "12.03.2024"),
            ("Ухвала від 12.03.1980 року народження", "12.03.1980"),
            ("Народився 12.03.1980, ухвала від 12.03.2024", "12.03.2024"),
            ("Зустріч відбулася 12.03.2024", "12.03.2024"),
            ("Особистий email судді: inbox@ki.court.gov.ua", "inbox@ki.court.gov.ua"),
            ("Email позивача: inbox@ki.court.gov.ua", "inbox@ki.court.gov.ua"),
            ("inbox@ki.court.gov.ua — електронна пошта позивача", "inbox@ki.court.gov.ua"),
            ("Акаунт: https://court.gov.ua/profile/123", "https://court.gov.ua/profile/123"),
            ("Email адвоката: office@example.com", "office@example.com"),
            ("Телефон адвоката: +380501234567", "+380501234567"),
            ("https://court.gov.ua/?person=123", "https://court.gov.ua/?person=123"),
            ("https://court.gov.ua.evil.example", "https://court.gov.ua.evil.example"),
            ("court.gov.ua.evil.example", "court.gov.ua.evil.example"),
            ("court.gov.ua?person=123", "court.gov.ua?person=123"),
            ("court.gov.ua:8080", "court.gov.ua"),
            ("https://evil.example/?next=court.gov.ua", "court.gov.ua"),
            ("https://court.gov.ua/?email=inbox@ki.court.gov.ua", "inbox@ki.court.gov.ua"),
            ("court.gov.ua?email=inbox@ki.court.gov.ua", "inbox@ki.court.gov.ua"),
            ("https://court.gov.ua/#person123", "https://court.gov.ua/#person123"),
            ("https://minjust.gov.ua.evil.example/", "https://minjust.gov.ua.evil.example/"),
            ("https://minjust.gov.ua/profile/123", "https://minjust.gov.ua/profile/123"),
            ("https://zakon.rada.gov.ua/laws/show/1402-19?person=123", "https://zakon.rada.gov.ua/laws/show/1402-19?person=123"),
            ("Найменування установи: Військовий навчальний центр «Десна»", "Військовий навчальний центр «Десна»"),
            ("Нікнейм: @example_user", "@example_user"),
            ("Виконавче провадження № 12345678", "12345678"),
        )
        for text, value in examples:
            with self.subTest(text=text):
                self.assertFalse(detect(text, [value], "ІНФОРМАЦІЯ")[0]["excluded"])

    def test_information_contained_fragments_and_crossing_spans(self):
        examples = (
            ("Сайт https://reyestr.court.gov.ua/Review/123", "https://reyestr.court.gov.ua/Review/123", "court.gov.ua"),
            ("E-mail прокуратури: office@example.com", "office@example.com", "example"),
            ("Ухвала від 12.03.2024", "12.03.2024", "2024"),
            ("Орган, що видав документ: Міністерство юстиції України", "Міністерство юстиції України", "юстиції"),
        )
        for text, full, fragment in examples:
            with self.subTest(text=text):
                result = detect(text, [full, fragment], "ІНФОРМАЦІЯ")
                self.assertTrue(all(c["excluded"] for c in result))
                self.assertEqual(result[0]["exclusion_reason"], result[1]["exclusion_reason"])
                self.assertIsNone(service_information_reason(text, 0, len(text)))

    def test_information_repeats_are_occurrence_local(self):
        text = "Ухвала від 12.03.2024. Дата народження: 12.03.2024"
        spans = [{"start": m, "end": m + 10, "value": text[m:m + 10], "type": "ІНФОРМАЦІЯ"}
                 for m in (text.index("12.03.2024"), text.rindex("12.03.2024"))]
        self.assertEqual([c["excluded"] for c in mark_service_candidates(text, spans)], [True, False])
        text = "Email прокуратури: office@example.com; Email позивача: office@example.com"
        spans = [{"start": m, "end": m + len("office@example.com"), "value": "office@example.com", "type": "ІНФОРМАЦІЯ"}
                 for m in (text.index("office@example.com"), text.rindex("office@example.com"))]
        self.assertEqual([c["excluded"] for c in mark_service_candidates(text, spans)], [True, False])

    def test_information_context_does_not_cross_sentence(self):
        text = "Дата народження: ІНФОРМАЦІЯ_1. Ухвала від 12.03.2024"
        self.assertTrue(detect(text, ["12.03.2024"], "ІНФОРМАЦІЯ")[0]["excluded"])
        text = "Email прокуратури: ІНФОРМАЦІЯ_1. Контакт: office@example.com"
        self.assertFalse(detect(text, ["office@example.com"], "ІНФОРМАЦІЯ")[0]["excluded"])

    def test_institution_contact_examples_from_registry(self):
        examples = (
            ("вул. Володимира Винниченка 1, м. Дніпро, 49027\n"
             "E-mail: inbox@dp.arbitr.gov.ua, тел. (056) 377-18-49, fax (056) 377-38-63",
             ["inbox@dp.arbitr.gov.ua", "arbitr.gov.ua"]),
            ("Шепетівська окружна прокуратура (вул. Сергія Оврашка, 43, м. Шепетівка, 30400; "
             "е-пошта: shepetiv_oprok@khmel.gp.gov.ua)", ["shepetiv_oprok@khmel.gp.gov.ua"]),
            ("3 - позивачу - Ленковецька сільська рада (вул. Коськівецька, 55, "
             "с. Ленківці Шепетівського району Хмельницької області, 30443; "
             "е-пош. lenkovecka-rada@ukr.net)", ["lenkovecka-rada@ukr.net"]),
            ("ГОСПОДАРСЬКИЙ СУД ЖИТОМИРСЬКОЇ ОБЛАСТІ\n\n"
             "майдан Путятинський, 3/65, м. Житомир, 10002, тел. (0412) 48 16 20,\n"
             "e-mail: inbox@zt.arbitr.gov.ua, web: https://zt.arbitr.gov.ua,\nкод ЄДРПОУ 03499916",
             ["inbox@zt.arbitr.gov.ua", "https://zt.arbitr.gov.ua"]),
            ("Е-пошта прокуратури: office@example.com", ["office@example.com"]),
            ("Ленковецька сільська рада (тел. (038) 123-45-67; е-пошта: office@ukr.net)",
             ["office@ukr.net"]),
        )
        for text, values in examples:
            with self.subTest(text=text):
                result = detect(text, values, "ІНФОРМАЦІЯ")
                self.assertTrue(all(c["excluded"] for c in result))
                self.assertTrue(all(c["exclusion_reason"] for c in result))

    def test_institution_contact_ownership_does_not_hide_private_contacts(self):
        examples = (
            ("Особистий email судді: inbox@zt.arbitr.gov.ua", "inbox@zt.arbitr.gov.ua"),
            ("https://zt.arbitr.gov.ua.evil.example", "https://zt.arbitr.gov.ua.evil.example"),
            ("inbox@zt.arbitr.gov.ua.evil.example", "inbox@zt.arbitr.gov.ua.evil.example"),
            ("Ленковецька сільська рада повідомила email позивача: person@ukr.net", "person@ukr.net"),
            ("Ленковецька сільська рада (особистий е-пош. person@ukr.net)", "person@ukr.net"),
            ("Ленковецька сільська рада (е-пошта позивача: person@ukr.net)", "person@ukr.net"),
            ("Ленковецька сільська рада (е-пошта: person@ukr.net — особистий контакт)", "person@ukr.net"),
            ("Адвокат ОСОБА_1 (е-пошта: person@ukr.net)", "person@ukr.net"),
            ("Ленковецька сільська рада (е-пошта: office@ukr.net", "office@ukr.net"),
            ("Ленковецька сільська рада (адреса: АДРЕСА_1); е-пош. person@ukr.net", "person@ukr.net"),
        )
        for text, value in examples:
            with self.subTest(text=text):
                self.assertFalse(detect(text, [value], "ІНФОРМАЦІЯ")[0]["excluded"])

    def test_same_email_has_separate_institutional_and_personal_owners(self):
        text = "Позивач — Ленковецька сільська рада (е-пош. contact@ukr.net); е-пошта позивача ОСОБА_1: contact@ukr.net"
        value = "contact@ukr.net"
        spans = [{"start": s, "end": s + len(value), "type": "ІНФОРМАЦІЯ", "value": value}
                 for s in (text.index(value), text.rindex(value))]
        self.assertEqual([c["excluded"] for c in mark_service_candidates(text, spans)], [True, False])

    def test_civil_authorities_are_service_even_as_parties(self):
        examples = (
            ("Позивач: Головне управління Державної податкової служби в Івано-Франківській області,", "Головне управління Державної податкової служби в Івано-Франківській області"),
            ("1) Головного управління Держгеокадастру у Донецькій області;", "Головного управління Держгеокадастру у Донецькій області"),
            ("в особі Управління Східного офісу Держаудитслужби в Донецькій області до відповідачів:", "Управління Східного офісу Держаудитслужби в Донецькій області"),
            ("перевіривши матеріали касаційної скарги Головного управління ДПС у Харківській області", "Головного управління ДПС у Харківській області"),
            ("Відповідач: Головному управлінню ДПС у Полтавській області", "Головному управлінню ДПС у Полтавській області"),
        )
        for text, value in examples:
            with self.subTest(text=text):
                result = detect(text, [value], "ІНФОРМАЦІЯ")[0]
                self.assertTrue(result["excluded"])
                self.assertTrue(result["exclusion_reason"])

    def test_civil_authority_regions_do_not_cover_personal_data(self):
        text = "Головне управління ДПС у Харківській області, працівник Петренко Іван Іванович, email person@example.com, народився 12.03.1980"
        values = ["Головне управління ДПС у Харківській області", "ДПС", "Харківській області", "person@example.com", "12.03.1980", "Петренко Іван Іванович", "Головне управління ДПС у Харківській області, працівник Петренко Іван Іванович"]
        result = detect(text, values, "ІНФОРМАЦІЯ")
        self.assertEqual([c["excluded"] for c in result], [True, True, True, False, False, False, False])
        for value in ("Управління ТОВ «Приватна компанія» у Харківській області", "ДПС-банк", "Приватний навчальний центр", "Військова частина А1234"):
            with self.subTest(value=value):
                self.assertFalse(detect(value, [value], "ІНФОРМАЦІЯ")[0]["excluded"])

    def test_public_authority_classes_and_case_forms(self):
        values = (
            "Могилів-Подільського районного територіального центру комплектування та соціальної підтримки",
            "Міністерства внутрішніх справи України",
            "Головного управління Пенсійного фонду України",
            "Міністерству соціальної політики України",
            "міністерством юстиції України",
            "Головним управління Пенсійного фонду України",
            "Головне управління Пенсійного фонду України в Харківській області",
            "Територіального управління Державної судової адміністрації",
            "територіальних центрах комплектування та соціальної підтримки",
            "ТЦК та СП",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertTrue(detect("Позов до " + value + ";", [value], "ІНФОРМАЦІЯ")[0]["excluded"])
        for text in ("Позов до Міністерства юстиції України", "Міністерства юстиції України — відповідач"):
            self.assertTrue(detect(text, ["Міністерства юстиції України"], "ІНФОРМАЦІЯ")[0]["excluded"])

    def test_authority_keywords_alone_do_not_prove_service_name(self):
        values = (
            "головне управління", "міністерство", "територіальний центр",
            "Головне управління приватної компанії у Харківській області",
            "Управління ТОВ у Харківській області",
            "Головне управління компанії Державний кредит у Харківській області",
            "Військовий навчальний центр «Десна»", "Військова частина А1234",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertFalse(detect(value, [value], "ІНФОРМАЦІЯ")[0]["excluded"])

    def test_authority_names_do_not_hide_nearby_private_data(self):
        name = "Головного управління Пенсійного фонду України"
        text = "Позивач: " + name + ", працівник Петренко Іван Іванович, особистий email person@example.com, дата народження: 12.03.1980"
        values = [name, "Пенсійного фонду", "person@example.com", "12.03.1980", "Петренко Іван Іванович", name + ", працівник Петренко Іван Іванович"]
        self.assertEqual([c["excluded"] for c in detect(text, values, "ІНФОРМАЦІЯ")], [True, True, False, False, False, False])
        name = "Могилів-Подільського районного територіального центру комплектування та соціальної підтримки"
        text = name + ", військовослужбовець ОСОБА_1, person@example.com"
        self.assertEqual([c["excluded"] for c in detect(text, [name, "person@example.com"], "ІНФОРМАЦІЯ")], [True, False])

    def test_audit_filters_information_after_collection(self):
        text = "Дата ухвалення рішення: 12.03.2024; email позивача: private@example.com"
        start = text.index("2024")
        result = raw_candidates(text, ner=[EntitySpan(start, start + 4, "2024", "ІНФОРМАЦІЯ", 0.99)])
        self.assertTrue(next(c for c in result if c["sources"] == ["ner"])["excluded"])
        self.assertFalse(next(c for c in result if c["value"] == "private@example.com")["excluded"])

    def test_institutional_contacts(self):
        self.assertTrue(detect("Телефон канцелярії: +380501234567", ["+380501234567"], "НОМЕР")[0]["excluded"])
        self.assertFalse(detect("Телефон адвоката: +380501234567", ["+380501234567"], "НОМЕР")[0]["excluded"])

    def test_court_letterhead_phones_and_model_fragments(self):
        examples = (
            ("ГОСПОДАРСЬКИЙ СУД\nДНІПРОПЕТРОВСЬКОЇ ОБЛАСТІ\n"
             "вул. Володимира Винниченка 1, м. Дніпро, 49027\n"
             "E-mail: inbox@dp.arbitr.gov.ua, тел. (056) 377-18-49, \u00a0fax (056) 377-38-63\n\nУХВАЛА",
             ["(056) 377-18-49", "377-18", "(056) 377-38-63"]),
            ("ГОСПОДАРСЬКИЙ СУД ХМЕЛЬНИЦЬКОЇ ОБЛАСТІ\n\n"
             "29000, м. Хмельницький, майдан Незалежності, 1 тел. 71-81-84, факс 71-81-98",
             ["71-81-84", "71-81-98"]),
            ("УКРАЇНА\nВЕРХОВНИЙ СУД\nТелефон: +380501234567",
             ["+380501234567"]),
        )
        for text, values in examples:
            for kind in ("НОМЕР", "ІНФОРМАЦІЯ"):
                with self.subTest(text=text, kind=kind):
                    self.assertTrue(all(c["excluded"] for c in detect(text, values, kind)))

    def test_letterhead_contact_scope_and_private_contacts(self):
        heading = "ГОСПОДАРСЬКИЙ СУД ХМЕЛЬНИЦЬКОЇ ОБЛАСТІ\n\n"
        for body in (
            "УХВАЛА\nтел. 71-81-84",
            "Позивач ОСОБА_1\nтел. 71-81-84",
            "Телефон адвоката: 71-81-84",
            "м. Хмельницький, особистий телефон: 71-81-84",
            "Суд повідомив контакт: тел. 71-81-84",
            "Невідомий рядок\nтел. 71-81-84",
        ):
            with self.subTest(body=body):
                self.assertFalse(detect(heading + body, ["71-81-84"], "НОМЕР")[0]["excluded"])
        self.assertFalse(detect("тел. 71-81-84", ["71-81-84"], "НОМЕР")[0]["excluded"])
        text = heading + "тел. 71-81-84\nУХВАЛА\nтел. 71-81-84"
        spans = [{"start": m.start(), "end": m.end(), "value": m.group(), "type": "НОМЕР"}
                 for m in re.finditer("71-81-84", text)]
        self.assertEqual([c["excluded"] for c in mark_service_candidates(text, spans)], [True, False])

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
        result = raw_candidates(text, ner=[EntitySpan(s, s + 3, "456", "НОМЕР", 0.99)])
        self.assertTrue(next(c for c in result if c["sources"] == ["ner"])["excluded"])
        self.assertFalse(next(c for c in result if c["value"] == "1234567890")["excluded"])

    def test_report_contains_exclusions_and_safe_json(self):
        text = "Суддя Петренко І. П. </script>"
        result = raw_candidates(text)
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
