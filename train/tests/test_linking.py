"""Linking of pseudonyms: B-cubed on three hand-written gold sets, the regression case of every rule, odd input."""
from __future__ import annotations

import copy
import itertools
import json
import random
import time
from collections import Counter
from pathlib import Path

import pytest

from pseudo import linking
from pseudo.linking import assign
from pseudo.predict import spans_to_submission

DATA = Path(__file__).parent / "data"
P, A, N, I = "ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ"
# linking_dev.json was open while the module was written; linking_judge.json was written by the reviewer who chose the
# module; linking_holdout.json was held back (one failure on it, the «foreign» document, was fixed afterwards).
SETS = ("linking_dev.json", "linking_judge.json", "linking_holdout.json")
SHUFFLES = 20


def document(mentions: list) -> tuple[str, list[dict], list[str]]:
    """A text with the mentions [text, entity id, type = ОСОБА] one after another, its spans and the gold entities."""
    text, spans, gold = "Суд встановив таке.", [], []
    for mention, entity, *rest in mentions:
        kind = rest[0] if rest else P
        text += " У матеріалах справи зазначено: "
        spans.append({"start": len(text), "end": len(text) + len(mention), "type": kind})
        gold.append(f"{kind}_{entity}")
        text += mention + "."
    return text, spans, gold


def b_cubed(gold: list[str], pred: list[str]) -> tuple[float, float]:
    """Precision and recall as data/evaluate.py computes them for one document."""
    gc, pc, both = Counter(gold), Counter(pred), Counter(zip(gold, pred))
    n = len(gold)
    return sum(both[x] / pc[x[1]] for x in zip(gold, pred)) / n, sum(both[x] / gc[x[0]] for x in zip(gold, pred)) / n


def score(groups: list[dict]) -> tuple[dict[str, float], list[str]]:
    """B-cubed F1 per type (documents weighted by their mentions) and the names of the documents with an error."""
    acc = {t: [0.0, 0.0, 0] for t in (P, A, N, I)}
    wrong = []
    for group in groups:
        text, spans, gold = document(group["mentions"])
        pred = assign(text, spans)
        for kind, a in acc.items():
            idx = [i for i, s in enumerate(spans) if s["type"] == kind]
            if not idx:
                continue
            p, r = b_cubed([gold[i] for i in idx], [pred[i] for i in idx])
            a[0], a[1], a[2] = a[0] + p * len(idx), a[1] + r * len(idx), a[2] + len(idx)
            if p < 1 or r < 1:
                wrong.append(f"{group['name']} {kind}")
    return {t: 2 * (p / n) * (r / n) / (p / n + r / n) if p + r else 0.0 for t, (p, r, n) in acc.items() if n}, wrong


def clusters(kind: str, *mentions: str) -> list[int]:
    """The number of the pseudonym of every mention, the mentions standing in one document in this order."""
    text, spans, _ = document([[m, 0, kind] for m in mentions])
    return [int(p.rsplit("_", 1)[1]) for p in assign(text, spans)]


def wrong_orders(kind: str, *mentions: tuple[str, str]) -> int:
    """In how many orders of the (mention, entity) pairs the pseudonyms are not exactly the entities."""
    wrong = 0
    for order in itertools.permutations(mentions):
        text, spans, gold = document([[m, entity, kind] for m, entity in order])
        wrong += b_cubed(gold, assign(text, spans)) != (1.0, 1.0)
    return wrong


def load(name: str) -> list[dict]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------------------------------
# Gold sets
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SETS)
def test_gold_set_in_file_order(name):
    f1, wrong = score(load(name))
    assert wrong == []
    assert f1 == {P: 1.0, A: 1.0, N: 1.0, I: 1.0}


@pytest.mark.parametrize("name", SETS)
def test_gold_set_in_shuffled_orders(name):
    # The file order flatters a linker: every person there starts with a nominative full name. The original module
    # reached 0.9932 (judge set) and 0.9959 (holdout) on the mean of 200 such orders; this one reaches 1.0 on all 200.
    groups = load(name)
    for seed in range(SHUFFLES):
        rng = random.Random(seed)
        shuffled = [{"name": g["name"], "mentions": rng.sample(g["mentions"], len(g["mentions"]))} for g in groups]
        f1, wrong = score(shuffled)
        assert wrong == [], f"seed {seed}"
        assert f1 == {P: 1.0, A: 1.0, N: 1.0, I: 1.0}


def test_gold_sets_have_the_expected_size():
    sizes = {name: sum(len(g["mentions"]) for g in load(name)) for name in SETS}
    assert sizes == {"linking_dev.json": 120, "linking_judge.json": 365, "linking_holdout.json": 109}


# ----------------------------------------------------------------------------------------------------------------------
# Persons
# ----------------------------------------------------------------------------------------------------------------------


def test_possessive_surname_gender_from_case_endings():
    # The man is first named in the genitive, which is also her nominative: the feminine forms are still hers.
    assert clusters(P, "Кузнецової Гелени Аркадіївни", "Кузнецова Гліба Аркадійовича", "Кузнецовою Г. А.", "Кузнецовій Г. А.",
                    "Кузнецової", "Кузнецовим Г. А.") == [1, 2, 1, 1, 1, 2]
    # ... and they never join him when no woman is there to take them.
    assert clusters(P, "Нікітіна Дениса Юрійовича", "Нікітіною Д. Ю.", "Нікітіної") == [1, 2, 2]


def test_mans_surname_in_a_is_not_feminine_evidence():
    assert clusters(P, "Діброва Гліб Аркадійович", "Дібровою Г. А.", "Діброви Гліба Аркадійовича", "Діброву Гліба Аркадійовича",
                    "Діброві Г. А.") == [1, 1, 1, 1, 1]
    assert clusters(P, "Калина Гліб Аркадійович", "Калиною Г. А.", "Калини") == [1, 1, 1]
    # «Зіновій» ends like a feminine dative of a possessive surname.
    assert clusters(P, "Зіновій Марчук", "Марчук Зіновій Устимович", "Зіновієм Марчуком") == [1, 1, 1]


def test_surname_and_first_name_agree_in_case_and_gender():
    assert clusters(P, "Заєць Валентин Олександрович", "Заєць Валентина Олександрівна", "Заєць Валентина", "Зайця Валентина",
                    "Валентин Заєць") == [1, 2, 2, 1, 1]
    assert clusters(P, "Притула Роман Михайлович", "Притули Романи Михайлівни", "Романа Притули", "Романи Притули") == [1, 2, 1, 2]
    assert clusters(P, "Мазуренка Романа Ярославовича", "Мазуренко Роману Савівну", "Романа Мазуренко", "Романа Мазуренка") == [1, 2, 2, 1]
    assert clusters(P, "Гнатенка Олександра Сидоровича", "Гнатенко Олександрі Савівні", "Олександрою Гнатенко",
                    "Гнатенко Олександр") == [1, 2, 2, 1]
    assert clusters(P, "Мазурок Валерія Павлівна", "Мазурок Валерій Павлович", "Валерій Мазурок", "Валерії Мазурок",
                    "Валерія Мазурка") == [1, 2, 2, 1, 2]


def test_agreement_only_chooses_between_persons():
    # A man's surname left undeclined by the clerk: no woman fits, so the mention is still his.
    assert clusters(P, "Шевченко Олега", "Шевченко Олег Петрович") == [1, 1]
    assert clusters(P, "Чжан Вея", "Чжан Вей") == [1, 1]
    assert clusters(P, "Сороки Гліба Аркадійовича", "Сорока Гліба Аркадійовича") == [1, 1]
    assert clusters(P, "Ахмедова Рустама Ільгар огли", "Ахмедов Рустам Ільгар огли", "Ахмедову Р. І.") == [1, 1, 1]
    # Women's names in a consonant keep their oblique forms.
    assert clusters(P, "Любов Мазур", "Мазур Любові Петрівни", "Любові Мазур", "Мазур Любов Петрівна") == [1, 1, 1, 1]
    assert clusters(P, "Мазур Естер Петрівна", "Мазур Естері Петрівни", "Естері Мазур") == [1, 1, 1]


def test_whole_name_beats_surname_and_initial_of_a_namesake():
    # A second person with the same surname is known by initials only. The strict reading of the two-word mention
    # fits him (surname and one initial); the word-by-word reading fits the whole name of the right person.
    for mentions in (
        (("Остапчук Любов Свиридівна", "a"), ("Остапчука Л. П.", "b"), ("Любові Остапчук", "a")),
        (("Остапчук Любов'ю Свиридівною", "a"), ("Остапчука Л. П.", "b"), ("Остапчук Любов", "a")),
        (("Седих Нодар Зурабович", "a"), ("Седих Н. В.", "b"), ("Седих Нодара", "a")),
        (("Мунтяну Вероніки Харитонівни", "a"), ("Мунтяну В. І.", "b"), ("Вероніка Мунтяну", "a")),
        (("Гриценко Харитон Юхимович", "a"), ("Гриценко Х. П.", "b"), ("Гриценко Харитона", "a")),
        (("Вербицькому Назарові Свиридовичу", "a"), ("Вербицький Н. З.", "b"), ("Вербицький Назар", "a")),
        (("Садовому Назарію Юхимовичу", "a"), ("Садова Н. П.", "b"), ("Садовим Назарієм", "a")),
    ):
        assert wrong_orders(P, *mentions) == 0, mentions


def test_adjectival_surname_of_a_man_first_named_in_the_dative():
    # «Садовому» is the dative of «Садовий», not of «Садовом»: his other forms are his, and not a third person.
    # (Next to «Садова Н. П.», which may be the genitive of a man called «Садов», a bare «Садовим» follows recency.)
    assert wrong_orders(P, ("Садовому Назарію Юхимовичу", "m"), ("Садова Ніна Петрівна", "f"), ("Садовим", "m"), ("Садовою", "f")) == 0
    assert clusters(P, "Садовому Назарію Юхимовичу", "Садовим", "Садового Н. Ю.", "Садовий Назарій") == [1, 1, 1, 1]
    assert clusters(P, "Садова Н. П.", "Садовому Назарію Юхимовичу", "Садовим") == [1, 2, 2]
    assert clusters(P, "Вербицькому Назарові Свиридовичу", "Вербицький Назар", "Вербицьким Н. С.") == [1, 1, 1]


def test_agreement_with_surnames_in_a_ya_iy():
    man, woman = ("Притула Роман Михайлович", "m"), ("Притула Романа Михайлівна", "f")  # she is named in the nominative
    assert wrong_orders(P, man, woman, ("Романа Притули", "m")) == 0
    assert wrong_orders(P, man, woman, ("Романи Притули", "f")) == 0
    man, woman = ("Лобода Богдан Юхимович", "m"), ("Лобода Богдана Юхимівна", "f")
    for mention in (("Лобода Богдана", "f"), ("Лободу Богдану", "f"), ("Лободі Богдану", "m"), ("Лободи Богдана", "m"),
                    ("Лободою Богданом", "m"), ("Лободи Богдани", "f")):
        assert wrong_orders(P, man, woman, mention) == 0, mention
    man, woman = ("Стогній Ярослав Несторович", "m"), ("Стогній Ярослава Несторівна", "f")
    for mention in (("Стогній Ярослава", "f"), ("Ярославі Стогній", "f"), ("Стогнія Ярослава", "m"), ("Стогній Ярослав", "m")):
        assert wrong_orders(P, man, woman, mention) == 0, mention
    assert wrong_orders(P, ("Гмиря Валентин Юхимович", "m"), ("Гмиря Валентина Юхимівна", "f"), ("Гмирі Валентина", "m")) == 0
    # The same when the anchors stand in an oblique case.
    assert wrong_orders(P, ("Лободи Богдана Юхимовича", "m"), ("Лободи Богдани Юхимівни", "f"), ("Лобода Богдана", "f")) == 0


def test_mans_surname_in_a_left_undeclined_next_to_his_declined_name():
    assert wrong_orders(P, ("Щербини Устима Гордійовича", "a"), ("Щербина Устима Гордійовича", "a")) == 0
    assert wrong_orders(P, ("Діброви Гліба Аркадійовича", "a"), ("Діброва Гліба Аркадійовича", "a")) == 0
    assert wrong_orders(P, ("Тичини Павла Григоровича", "a"), ("Тичина Павло Григорович", "a"), ("Тичина Павла Григоровича", "a")) == 0
    assert wrong_orders(P, ("Сороки Устима Гордійовича", "a"), ("Сорока Устима Гордійовича", "a")) == 0
    # ... and the possessive reading still stands for anything shorter than the whole name.
    assert clusters(P, "Нікітіна Дениса Юрійовича", "Нікітіною Д. Ю.", "Нікітіної") == [1, 2, 2]


def test_patronymics_without_ov():
    assert wrong_orders(P, ("Гришиної Мирослави Кузьмівни", "f"), ("Гришина Мирона Кузьмича", "m"), ("Гришиною М. К.", "f"),
                        ("Гришиної", "f")) == 0
    assert wrong_orders(P, ("Романюка Станіслава Лукича", "m"), ("Романюк Станіслава Луківна", "f"), ("Романюк Станіслава", "f")) == 0


def test_hard_adjectival_surname_is_not_a_possessive_one():
    man, woman = ("Зеленим Трохимом Юхимовичем", "m"), ("Зеленої Таїсії Юхимівни", "f")
    assert wrong_orders(P, man, woman, ("Зелена Т. Ю.", "f")) == 0
    assert wrong_orders(P, man, woman, ("Зелена", "f")) == 0
    assert wrong_orders(P, man, woman, ("Зеленого", "m")) == 0
    # -им stays the instrumental of possessive surnames, -ів among them.
    assert clusters(P, "Семків Атанас Олексійович", "Семківим Атанасом Олексійовичем", "Кузнецов Гліб", "Кузнецовим Г.") == [1, 1, 2, 2]


def test_surname_that_looks_like_a_feminine_patronymic():
    assert wrong_orders(P, ("Головня Кирило Савович", "a"), ("Головні Кирила Савовича", "a"), ("Головні К. С.", "a")) == 0
    assert wrong_orders(P, ("Головні Олександра Юхимовича", "m"), ("Головні Олександри Юхимівни", "f"), ("Олександрою Головнею", "f")) == 0


def test_third_declension_instrumental_with_a_doubled_consonant():
    assert clusters(P, "Гайдамака Нінель Свиридівна", "Гайдамакою Нінеллю Свиридівною", "Нінеллю Гайдамакою") == [1, 1, 1]
    assert clusters(P, "Шварц Рахіль Давидівна", "Рахіллю Шварц", "Шварц Любов'ю Давидівною") == [1, 1, 2]


def test_double_first_name_with_a_surname_in_ovych():
    assert clusters(P, "Адамович Єва-Марія Тадеївна", "Адамович Єва-Марія", "Єва-Марія Адамович") == [1, 1, 1]
    # Three names, one of them a patronymic: still a man.
    assert clusters(P, "Нечуй-Левицький Іван Семенович", "Нечуя-Левицького Івана Семеновича", "Нечуй-Левицька Іванна Семенівна") == [1, 1, 2]


def test_role_words_inside_a_span():
    assert wrong_orders(P, ("Обвинувачений Довженко Харитон Панасович", "a"), ("Довженко Харитон Панасович", "a"),
                        ("Довженко Х. П.", "a"), ("Довженка Х. П.", "a")) == 0
    assert clusters(P, "Довженко Харитон Панасович", "Свідок Довженко Х. П.", "потерпілого Довженка Х. П.",
                    "Представник позивача Довженко Х. П.", "старшого лейтенанта Довженка") == [1, 1, 1, 1, 1]


def test_placeholder_is_linked_by_its_word_and_number():
    assert clusters(P, "ОСОБА_1", "ОСОБА_2", "ОСОБА_1", "ОСОБА_12", "особа_1") == [1, 2, 1, 3, 1]
    assert clusters(P, "ОСОБА_1", "ОСОБА_1,", "ОСОБА_2", "(ОСОБА_2)") == [1, 1, 2, 2]
    assert clusters(P, "ОСОБА 1", "ОСОБА 2", "Особа 3", "Особа2", "ОСОБА_3") == [1, 2, 3, 2, 3]
    assert clusters(P, "ОСОБА_1, ОСОБА_2", "ОСОБА_1, ОСОБА_3", "ОСОБА_1, ОСОБА_2") == [1, 2, 1]
    # A year swallowed by the span is not a placeholder, also when it is glued to «р.».
    assert clusters(P, "Коваль І. П., 1980", "Коваль Іван Петрович") == [1, 1]
    assert clusters(P, "Довженко Харитон Панасович", "Довженко Харитон Панасович, 1980р.н.", "Довженко 1980") == [1, 1, 1]


def test_behaviour_kept_from_the_original_module():
    assert clusters(P, "Гнатюк Олег Іванович", "Гнатюка", "Гнатюк Іван Олегович", "Гнатюка", "Гнатюком", "Гнатюк Олег",
                    "Гнатюк О.") == [1, 1, 2, 2, 2, 1, 1]
    assert clusters(P, "Панчишин Ростислав Едуардович", "Суддя Панчишин Р. Е.", "ФОП Панчишина Ростислава Едуардовича") == [1, 1, 1]
    assert clusters(P, "Гнатюк Олег Іванович", "Гнатюк Oлег Iванович", "Гнaтюка Олега") == [1, 1, 1]  # Latin o, I, a inside
    assert clusters(P, "Данильченко Яна Борисівна", "Данильченко Яків Борисович", "Данильченку Я. Б.", "Данильченка") == [1, 2, 2, 2]
    # The two rules the review ruled out: a name with patronymic does not join a person known by initials only.
    assert clusters(P, "Заславський Орест Орестович", "Сокіл О. О.", "Ореста Орестовича") == [1, 2, 1]


# ----------------------------------------------------------------------------------------------------------------------
# The other types
# ----------------------------------------------------------------------------------------------------------------------


def test_unit_code_written_apart_and_abbreviations():
    assert clusters(I, "Військова частина А1234", "в/ч А 1234", "військової частини А 1234", "ВЧ А1234", "військовій частині A1234",
                    "військова частина А1235") == [1, 1, 1, 1, 1, 2]
    assert clusters(I, "Міністерство оборони України", "МОУ", "Міністерства оборони України", "Міністерство юстиції України") == [1, 1, 1, 2]
    assert clusters(I, "Державне підприємство «Укрпошта»", "ДП «Укрпошта»", "ТОВ «Ромашка»",
                    "Товариство з обмеженою відповідальністю «Ромашка»", "КП «Ромашка»") == [1, 1, 2, 2, 3]
    assert clusters(I, "Шевченківський районний територіальний центр комплектування та соціальної підтримки",
                    "Шевченківського районного ТЦК та СП", "Шевченківський РТЦК та СП", "Київський міський ТЦК та СП") == [1, 1, 1, 2]
    assert clusters(I, "Збройні Сили України", "ЗСУ", "військова частина НОМЕР_1", "військової частини НОМЕР_1",
                    "військова частина НОМЕР_2") == [1, 1, 2, 2, 3]
    assert clusters(I, "05.09.1987", "наказ № 45 від 05.09.1987", "1987-09-05", "05 вересня 1987") == [1, 2, 1, 1]


def test_unit_code_letters_ordinals_and_short_words():
    # After «частина» a letter of either case is the letter of the code; a short word before a number is not.
    assert clusters(I, "в/ч А 7031", "в/ч а 7031", "в.ч. А7031", "військова частина а7031", "А 7031", "в/ч А 7032") == [1, 1, 1, 1, 2, 3]
    assert clusters(I, "з 2019 року", "З 2019 року", "у 2019 році", "У 2019 році", "ДО 2019 року") == [1, 1, 2, 2, 3]
    assert clusters(I, "110 окрема механізована бригада", "110-та окрема механізована бригада", "110-ї окремої механізованої бригади",
                    "111-та окрема механізована бригада") == [1, 1, 1, 2]
    assert clusters(I, "111 окремий полк", "111-й окремий полк", "111-го окремого полку") == [1, 1, 1]
    assert clusters(I, "МОУ", "Міноборони", "Міністерство оборони України") == [1, 1, 1]
    assert clusters(I, "17.04.1992", "«17» квітня 1992 року", "\"17\" квітня 1992 р.", "18.04.1992") == [1, 1, 1, 2]
    assert clusters(I, "0671112233", "тел. 067 111 22 33", "телефон: +380671112233", "тел. 067 111 22 34") == [1, 1, 1, 2]


def test_accounts_and_pages_by_their_exact_spelling():
    assert clusters(I, "https://instagram.com/svitlana_3240", "https://instagram.com/svitlana.3240", "instagram.com/svitlana_3240",
                    "https://www.instagram.com/svitlana_3240/") == [1, 2, 1, 1]
    assert clusters(I, "@svitlana_3240", "@svitlana3240", "svitlana_3240", "ІНФОРМАЦІЯ_1", "ІНФОРМАЦІЯ_2", "ІНФОРМАЦІЯ_1") == [1, 2, 1, 3, 4, 3]


def test_case_numbers_keep_their_slashes():
    assert clusters(N, "755/1234/21", "755/123/421", "№ 755/1234/21", "755 / 1234 / 21", "12/5", "125") == [1, 2, 1, 1, 3, 4]
    assert clusters(N, "АВ 123456", "АВ-123456", "ав123456", "4149 4991 1234 5678", "4149-4991-1234-5678") == [1, 1, 1, 2, 2]


def test_declined_and_hyphenated_settlements():
    assert clusters(A, "м. Київ, вул. Вербова, 7", "м. Києва, вул. Вербова, 7", "у місті Києві по вулиці Вербовій, 7",
                    "м. Канів, вул. Вербова, 7") == [1, 1, 1, 2]
    assert clusters(A, "м. Миколаїв, вул. Вербова, 7", "м. Миколаєва, вул. Вербова, 7", "м. Кам'янець-Подільський, вул. Вербова, 7",
                    "м. Кам'янця-Подільського, вул. Вербова, 7", "м. Кам'янець-Бузький, вул. Вербова, 7") == [1, 1, 2, 2, 3]


def test_house_numbers_and_street_names():
    assert clusters(A, "м. Буча, вул. Лісова, 1-2", "м. Буча, вул. Лісова, 12", "м. Буча, вул. Лісова, 1-А",
                    "м. Буча, вул. Лісова, 1а") == [1, 2, 3, 3]
    assert clusters(A, "м. Буча, вул. Героїв Майдану, 5", "м. Буча, вул. Героїв України, 5", "Україна, м. Буча, вулиця Героїв Майдану, 5",
                    "м. Буча, вул. Героїв України, буд. 5, Україна") == [1, 2, 1, 2]
    # A street named after a person: with the first name, with its initial, by the surname alone.
    assert wrong_orders(A, ("м. Буча, вул. Івана Мазепи, 5", "a"), ("м. Буча, вул. Мазепи, 5", "a"), ("м. Буча, вул. І. Мазепи, 5", "a"),
                        ("м. Буча, вул. Івана Франка, 5", "b")) == 0
    assert clusters(A, "м. Буча, вул. М. Грушевського, 5", "м. Буча, вул. Грушевського, 5", "м. Буча, вул. Михайла Грушевського, 5",
                    "м. Буча, вул. 8 Березня, 5", "м. Буча, вул. Березня, 5") == [1, 1, 1, 2, 3]


def test_address_without_settlement_joins_the_most_recent_one():
    assert clusters(A, "м. Харків, вул. Миру, 5", "м. Суми, вул. Миру, 5", "вул. Миру, 5", "м. Харків, вулиця Миру, 5",
                    "вул. Миру, 5") == [1, 2, 2, 1, 1]
    # Before both: the one named soonest after it, and never a third address.
    assert clusters(A, "вул. Миру, 5", "м. Харків, вул. Миру, 5", "м. Суми, вул. Миру, 5", "вул. Миру, 5") == [1, 1, 2, 2]
    assert clusters(A, "вул. Миру, 5", "вул. Миру, 5", "вул. Миру, 6") == [1, 1, 2]


def test_phone_prefixes():
    assert clusters(N, "+380501234567", "0501234567", "(050) 123-45-67", "8 050 123 45 67", "8(050)1234567", "380501234567",
                    "8 050 123 45 68", "+80501234567") == [1, 1, 1, 1, 1, 1, 2, 3]


# ----------------------------------------------------------------------------------------------------------------------
# Odd input, determinism
# ----------------------------------------------------------------------------------------------------------------------

T = ("Суд встановив, що Коваленко Петро Іванович, 12.03.1980 року народження, мешкає: м. Суми, вул. Миру, 4, паспорт АВ 123456. "
     "Позивач Коваленка П. І. та «Гнатюк Олена Ігорівна», ... !!! --- (в/ч А1234).")


def sp(sub: str, kind: str, text: str = T) -> dict:
    start = text.index(sub)
    return {"start": start, "end": start + len(sub), "type": kind}


UNSORTED = [sp("Гнатюк Олена Ігорівна", P), sp("АВ 123456", N), sp("Коваленка П. І.", P), sp("м. Суми, вул. Миру, 4", A),
            sp("Коваленко Петро Іванович", P), sp("12.03.1980", I)]
MALFORMED = {
    "empty list": (T, []),
    "empty text and list": ("", []),
    "punctuation-only spans": (T, [sp("...", P), sp("!!!", P), sp("---", A), sp("!!!", N), sp("...", I), sp(", ", P)]),
    "start == end": (T, [{"start": 5, "end": 5, "type": t} for t in (P, P, A, N, I)]),
    "whitespace-only span": (T, [{"start": 3, "end": 4, "type": P}, {"start": 13, "end": 14, "type": P}]),
    "overlapping spans": (T, [sp("Коваленко Петро Іванович", P), sp("Петро Іванович, 12.03.1980", P), sp("Коваленко", P),
                              sp("12.03.1980", I), sp("12.03.1980 року народження", I), sp("м. Суми, вул. Миру, 4", A),
                              sp("вул. Миру, 4, паспорт", A)]),
    "spans not sorted": (T, UNSORTED),
    "type outside the four": (T, [sp("Коваленко Петро Іванович", "ОРГАНІЗАЦІЯ"), sp("Коваленка П. І.", "ОРГАНІЗАЦІЯ"),
                                  sp("АВ 123456", "PHONE"), sp("12.03.1980", ""), sp("Гнатюк Олена Ігорівна", "ОРГАНІЗАЦІЯ")]),
    "type None / missing / int": (T, [{"start": 0, "end": 3, "type": None}, {"start": 4, "end": 9}, {"start": 4, "end": 9, "type": 7}]),
    "missing start / end": (T, [{"type": P}, {"start": 3, "type": P}, {"end": 9, "type": N}, {"start": None, "end": None, "type": A}]),
    "negative / reversed / out of range": (T, [{"start": -5, "end": 4, "type": P}, {"start": 30, "end": 10, "type": P},
                                               {"start": 10**6, "end": 10**6 + 5, "type": A},
                                               {"start": len(T) - 3, "end": len(T) + 50, "type": I}, {"start": -10, "end": -2, "type": N}]),
    "non-integer offsets": (T, [{"start": "18", "end": "27", "type": P}, {"start": 18.0, "end": 27.9, "type": P},
                                {"start": "x", "end": [], "type": N}, {"start": float("nan"), "end": 5, "type": I},
                                {"start": float("inf"), "end": 5, "type": A}, {"start": 0, "end": float("-inf"), "type": P},
                                {"start": 10**400, "end": 5, "type": P}]),
    "spans that are not dicts": (T, [None, "ОСОБА", 5, ("a", "b"), sp("Коваленко", P)]),
    "text is None": (None, [{"start": 0, "end": 5, "type": P}]),
    "text is bytes": (T.encode(), [{"start": 0, "end": 5, "type": P}, {"start": 0, "end": 5, "type": N}]),
    "extra keys in spans": (T, [{**sp("Коваленко Петро Іванович", P), "score": 0.9, "text": "x"}]),
    "odd unicode": ("A\u0000​﻿ 𝔘𝔫𝔦 \U0001F600 Ко́валь І. П. ı İ ß",
                    [{"start": 0, "end": 4, "type": P}, {"start": 5, "end": 9, "type": P}, {"start": 10, "end": 12, "type": N},
                     {"start": 12, "end": 27, "type": P}, {"start": 27, "end": 33, "type": P}, {"start": 27, "end": 33, "type": A},
                     {"start": 27, "end": 33, "type": I}]),
    "very long single span": ("Коваль " * 5000, [{"start": 0, "end": 35000, "type": t} for t in (P, A, N, I)]),
    "placeholders as persons": ("ОСОБА_1 та ОСОБА_2, ОСОБА_1; 12345", [{"start": 0, "end": 7, "type": P}, {"start": 11, "end": 18, "type": P},
                                                                         {"start": 20, "end": 27, "type": P}, {"start": 29, "end": 34, "type": P}]),
}


@pytest.mark.parametrize("name", MALFORMED)
def test_malformed_input_never_raises_and_keeps_the_contract(name):
    text, spans = MALFORMED[name]
    before = copy.deepcopy(spans)
    out = assign(text, spans)
    assert spans == before or name == "non-integer offsets"  # NaN != NaN; that case is compared below by repr
    assert repr(spans) == repr(before)
    assert isinstance(out, list) and len(out) == len(spans)
    seen: dict[str, int] = {}
    for span, pseudonym in zip(spans, out):
        head, _, number = pseudonym.rpartition("_")
        if isinstance(span, dict) and span.get("type") is not None:
            assert head == str(span["type"])
        # Numbers start at 1 and grow by one in order of first appearance, per type.
        assert number.isdigit() and 1 <= int(number) <= seen.get(head, 0) + 1
        seen[head] = max(seen.get(head, 0), int(number))
    assert assign(text, copy.deepcopy(before)) == out


def test_odd_offsets_give_an_empty_mention():
    for value in (float("inf"), float("-inf"), float("nan"), "x", None, []):
        assert assign("Коваль І. П.", [{"start": value, "end": 5, "type": P}, {"start": 0, "end": value, "type": P}]) == ["ОСОБА_1", "ОСОБА_1"]
        assert linking._mention_text("Коваль І. П.", {"start": value, "end": 5}) == ""


def test_same_clusters_whatever_the_order_of_the_spans():
    by_start = sorted(UNSORTED, key=lambda s: s["start"])
    unsorted, ordered = assign(T, UNSORTED), assign(T, by_start)
    assert ordered == ["ОСОБА_1", "ІНФОРМАЦІЯ_1", "АДРЕСА_1", "НОМЕР_1", "ОСОБА_1", "ОСОБА_2"]
    a = {s["start"]: p for s, p in zip(UNSORTED, unsorted)}
    b = {s["start"]: p for s, p in zip(by_start, ordered)}
    assert all((a[x] == a[y]) == (b[x] == b[y]) for x in a for y in a)
    assert assign(T, UNSORTED) == unsorted


def test_recency_follows_the_text_not_the_list_of_spans():
    # The bare surnames are ambiguous: each goes to the namesake named last before it in the text. The answer must
    # not change when the same spans come in another order.
    text = ("Гнатюк Олег Іванович та Гнатюк Іван Олегович. Позов Гнатюка задоволено, а Гнатюком Олегом подано скаргу. "
            "Надалі Гнатюка звільнено. Адреси: м. Харків, вул. Миру, 5; м. Суми, вул. Миру, 5; вул. Миру, 5.")
    spans = [sp("Гнатюк Олег Іванович", P, text), sp("Гнатюк Іван Олегович", P, text), sp("Гнатюка задоволено", P, text),
             sp("Гнатюком Олегом", P, text), sp("Гнатюка звільнено", P, text), sp("м. Харків, вул. Миру, 5", A, text),
             sp("м. Суми, вул. Миру, 5", A, text), sp("вул. Миру, 5.", A, text)]
    for span, cut in ((spans[2], " задоволено"), (spans[4], " звільнено"), (spans[7], ".")):
        span["end"] -= len(cut)
    assert assign(text, spans) == ["ОСОБА_1", "ОСОБА_2", "ОСОБА_2", "ОСОБА_1", "ОСОБА_1", "АДРЕСА_1", "АДРЕСА_2", "АДРЕСА_2"]
    expected = {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 5, 6: 6, 7: 6}  # span -> the first span of its entity
    rng = random.Random(5)
    for order in ([4, 3, 2, 1, 0, 7, 6, 5], [2, 4, 0, 1, 3, 7, 5, 6], *(rng.sample(range(8), 8) for _ in range(20))):
        pseudonyms = dict(zip(order, assign(text, [spans[i] for i in order])))
        assert all((pseudonyms[i] == pseudonyms[j]) == (expected[i] == expected[j]) for i in order for j in order), order
        # Numbers follow the list that was given: the first span of every type gets 1.
        firsts = {s["type"]: i for i, s in reversed([(i, spans[i]) for i in order])}
        assert all(pseudonyms[i].endswith("_1") for i in firsts.values())


def test_same_output_for_the_same_input_after_other_documents():
    groups = load("linking_judge.json")
    first = [assign(*document(g["mentions"])[:2]) for g in groups]
    again = [assign(*document(g["mentions"])[:2]) for g in reversed(groups)]
    assert again[::-1] == first


def test_many_mentions_of_few_persons_are_fast():
    rng = random.Random(7)
    people = [f"{s} {n} {p}" for s in ("Гнатюк", "Савчук", "Мороз", "Бойко", "Левченко", "Поліщук", "Мазур", "Литвин")
              for n, p in (("Олег", "Іванович"), ("Назар", "Петрович"), ("Тарас", "Маркович"))]
    text, spans, _ = document([[rng.choice(people), 0] for _ in range(3000)])
    started = time.perf_counter()
    assert len(set(assign(text, spans))) == len(people)
    assert time.perf_counter() - started < 5  # about 0.1 s


def test_many_different_persons_are_fast_too():
    # Without the index of name stems every mention is compared with every person: 2000 different persons then take
    # more than ten seconds, with it a fraction of one. The clusters are the same either way.
    surnames = ("Гнатюк Савчук Мороз Бойко Левченко Поліщук Мазур Литвин Коваль Шевченко Ткаченко Кравець Олійник Шевчук Бондар "
                "Мельник Кучер Лисенко Руденко Марченко Заєць Притула Сорока Ковальський Кузнецов Діброва Седих Стогній Гмиря "
                "Петренко Іваненко Савченко Романюк Остапчук Довженко Вербицький Садовий Головня Щербина Тичина").split()
    names = "Олег Назар Тарас Іван Петро Андрій Роман Богдан Степан Юрій".split()
    fathers = "Іванович Петрович Маркович Олегович Андрійович".split()
    people = [f"{s} {n} {f}" for s in surnames for n in names for f in fathers]
    random.Random(3).shuffle(people)
    text, spans, _ = document([[p, 0] for p in people])
    started = time.perf_counter()
    assert len(set(assign(text, spans))) == len(people) == 2000
    assert time.perf_counter() - started < 5  # about 0.3 s
    # The index only leaves out persons that cannot fit: the same partition as when every person is a candidate.
    few = people[:150] + [p.split()[0] for p in people[:60]] + [f"{p.split()[1]} {p.split()[0]}а" for p in people[:60]]
    text, spans, _ = document([[p, 0] for p in few])
    indexed = assign(text, spans)
    original = linking._HeadIndex.candidates
    try:
        linking._HeadIndex.candidates = lambda self, m: sorted(self.words)
        assert assign(text, spans) == indexed
    finally:
        linking._HeadIndex.candidates = original


# ----------------------------------------------------------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------------------------------------------------------


def test_spans_to_submission_links_sorts_and_numbers():
    text = ("Позивач Гнатюк Олена Ігорівна (паспорт АВ 123456) та відповідач Коваленко Петро Іванович, м. Суми, вул. Миру, 4. "
            "Гнатюк О. І. просила стягнути з Коваленка П. І. борг; паспорт АВ123456.")
    spans = [sp(m, t, text) for m, t in (("Гнатюк Олена Ігорівна", P), ("АВ 123456", N), ("Коваленко Петро Іванович", P),
                                         ("м. Суми, вул. Миру, 4", A), ("Гнатюк О. І.", P), ("Коваленка П. І.", P), ("АВ123456", N))]
    expected = ["ОСОБА_1", "НОМЕР_1", "ОСОБА_2", "АДРЕСА_1", "ОСОБА_1", "ОСОБА_2", "НОМЕР_1"]
    for given in (spans, spans[::-1], spans[3:] + spans[:3]):
        before = copy.deepcopy(given)
        row = spans_to_submission("doc-7", text, given)
        assert row == {"id": "doc-7", "annotations": [{"start": s["start"], "end": s["end"], "pseudonym": p}
                                                      for s, p in zip(spans, expected)]}
        assert given == before
    assert spans_to_submission(3, text, []) == {"id": 3, "annotations": []}


def test_pseudonyms_are_numbered_in_order_of_first_appearance():
    # Persons are clustered the richest mention first, so the entity found first is not the one that appears first:
    # the bare surname and the initials stand before both full names.
    text = ("Коваленка викликано до суду разом із Гнатюк О. І.; м. Суми, вул. Миру, 4 та м. Харків, вул. Миру, 4. "
            "Гнатюк Олена Ігорівна не з'явилась, Коваленко Петро Іванович з'явився (вул. Миру, 4, м. Харків).")
    spans = [sp(m, t, text) for m, t in (("Коваленка", P), ("Гнатюк О. І.", P), ("м. Суми, вул. Миру, 4", A), ("м. Харків, вул. Миру, 4", A),
                                         ("Гнатюк Олена Ігорівна", P), ("Коваленко Петро Іванович", P), ("вул. Миру, 4, м. Харків", A))]
    expected = ["ОСОБА_1", "ОСОБА_2", "АДРЕСА_1", "АДРЕСА_2", "ОСОБА_2", "ОСОБА_1", "АДРЕСА_2"]
    assert assign(text, spans) == expected
    for given in (spans, spans[::-1], spans[4:] + spans[:4]):
        assert [a["pseudonym"] for a in spans_to_submission("d", text, given)["annotations"]] == expected


def test_a_linker_that_fails_still_gives_every_span_a_pseudonym(monkeypatch):
    def broken(mentions: list[str]) -> list[int]:
        raise RuntimeError("boom")

    text, spans, _ = document([["Коваль Іван Петрович", 0], ["Коваля І. П.", 0], ["м. Суми, вул. Миру, 4", 0, A], ["Коваль", 0]])
    assert assign(text, spans) == ["ОСОБА_1", "ОСОБА_1", "АДРЕСА_1", "ОСОБА_1"]
    monkeypatch.setitem(linking._LINKERS, P, broken)
    assert assign(text, spans) == ["ОСОБА_1", "ОСОБА_2", "АДРЕСА_1", "ОСОБА_3"]  # one pseudonym per span, the other type untouched
    monkeypatch.setitem(linking._LINKERS, P, lambda mentions: [0])  # a wrong number of clusters is a failure too
    assert assign(text, spans) == ["ОСОБА_1", "ОСОБА_2", "АДРЕСА_1", "ОСОБА_3"]
