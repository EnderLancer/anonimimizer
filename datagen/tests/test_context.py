"""The context rules of the fill step (silver.fill.context): sub-type of an ІНФОРМАЦІЯ placeholder, its hints and masks,
the vote over the occurrences of one index, and the case of an institution. For every rule there is one context it
must take and one, as close as possible, that it must leave to another rule."""
from __future__ import annotations

import re

import pytest

from silver.fill.context import (INFO_LEFT, INFO_RIGHT, bank_in_quotes, choose_subtype, guess_case, guess_gender, guess_org_case,
                                 info_subtype, info_subtypes, inside_quotes, number_subtype)

MASK = {"mask": True}

# (left, right, sub-type, hints that must be present); the id names the rule of the v6 specification, section 3
MATCH = [
    # --- A. inside the quotes of the text
    ("A1 birth date in quotes", "зазначивши вірно дату народження боржника - ОСОБА_2, «", "», замість невірної", "dob", {}),
    ("A2 e-mail in quotes", "на електронну адресу суду «", "» надійшла заява", "email", {}),
    ("A2 e-mail in quotes, general cue", "повідомлення надіслано на e-mail позивача та його представника: «", "»", "email", {}),
    ("A3 date in quotes, birth", "змінивши рік народження з «", "» на", "dob", {}),
    ("A3 date in quotes, bare", "виправити дату «", "» на", "date_event", MASK),
    ("A4 surname in quotes", "змінити прізвище дитини з «", "» на", "quoted", {"quote_kind": "person", **MASK}),
    ("A5 formation", "оперативного командування «", "» Сухопутних військ", "quoted", {"quote_kind": "formation"}),
    ("A5 border-service section", "відділу прикордонної служби «", "» ІІ категорії", "quoted", {"quote_kind": "formation"}),
    ("A6 institution", "лікування в КНП «", "» за адресою", "quoted", {"quote_kind": "institution"}),
    ("A6 institution, full form", "комунального закладу Львівської обласної ради «", "»", "quoted", {"quote_kind": "institution"}),
    ("A7 company", "ТОВ «", "» (код ЄДРПОУ", "quoted", {"quote_kind": "company"}),
    ("A7 company, full legal form", "Товариство з обмеженою відповідальністю «", "» звернулося", "quoted", {"quote_kind": "company"}),
    ("A7 bank by abbreviation", "рахунок в АТ КБ «", "» відкрито", "quoted", {"quote_kind": "bank"}),
    ("A7 bank by the card before a joint-stock form", "перерахувала на банківську картку АТ «", "» № НОМЕР_3", "quoted", {"quote_kind": "bank"}),
    ("A7 bank by the account", "по рахунку НОМЕР_2, відкритому в АТ «", "»,", "quoted", {"quote_kind": "bank"}),
    ("A curly quotes", "на орендованій ТОВ “ ", " ” території майданчика", "quoted", {"quote_kind": "company"}),
    ("A7 bank by noun", "у відділенні банку «", "» отримав", "quoted", {"quote_kind": "bank"}),
    ("A8 quoted, no cue", "мобільного додатку «", "» встановлено", "quoted", {"quote_kind": "unknown", **MASK}),
    # --- B. right-context checks
    ("B1 р.н.", "ОСОБА_1, ", " р.н., уродженець", "dob", {"style": "numeric"}),
    ("B2 unit code on the right", "за позовом ОСОБА_1 до ", " (військова частина НОМЕР_1) про визнання", "org", {"org_kind": "unit"}),
    ("B2 unit code after a dash", "третя особа: ", " - в/ч НОМЕР_2", "org", {"org_kind": "unit"}),
    ("B2 parent body on the right", "Зобов`язав ", " Державної прикордонної служби України нарахувати", "org", {"org_kind": "unit"}),
    # --- C. before the general rules
    ("C1 place", "уродженця ", ", зареєстрованого", "place", MASK),
    ("C2 requisites in a bracket", "після смерті ОСОБА_1 (", ", РНОКПП НОМЕР_1) відкрилась", "dob", {}),
    ("C2 requisites, wider right side", "стягнути з ОСОБА_1 (", ", остання відома адреса: АДРЕСА_1)", "dob", {}),
    ("C2 requisites in a list", "померлого ОСОБА_3, АДРЕСА_1, ", ", РНОКПП НОМЕР_2", "dob", {}),
    ("C2 requisites in a list, wider right side", "з ОСОБА_3 (", ", реєстраційний номер облікової картки", "dob", {}),
    ("C3 derived, majority", "до досягнення дитиною повноліття, тобто до ", ".", "derived", {"until": "majority"}),
    ("C3 derived, majority after a dash", "до його повноліття - ", ".", "derived", {"until": "majority"}),
    ("C3 derived, 23 in words", "до досягнення нею двадцяти трьох років, тобто до ", ".", "derived", {"until": "age23"}),
    ("C3 derived, 23 in digits", "до досягнення ОСОБА_3 23-річного віку, тобто до ", ",", "derived", {"until": "age23"}),
    ("C3 derived, end of studies", "до закінчення навчання ОСОБА_3, тобто по ", " включно", "derived", {"until": "age23"}),
    ("C3 derived, age of three", "до досягнення дитиною трирічного віку, тобто до ", ".", "derived", {"until": "age3"}),
    ("C4 death cue adjacent", "ОСОБА_2, яка померла ", ",", "death", {}),
    ("C4 death cue beats a birth cue further left", "який народився у 1950 році та помер ", " у місті", "death", {}),
    ("C5 death verb on the right", "ОСОБА_1 зазначав, що ", " ОСОБА_3 помер, однак", "death", {}),
    ("C5 death verb after a place", "Встановлено, що ", " у місті Києві померла ОСОБА_2", "death", {}),
    ("C6 death with a linking word", "Днем смерті ОСОБА_2 вважати ", ".", "death", {}),
    ("C7 birth verb on the right", "Сторони перебувають у шлюбі. ", " у сторін народилася донька", "dob", {"child": True}),
    ("C8 short name in a bracket", "до ІНФОРМАЦІЯ_1 (далі - відповідач, ", "), у якому просить", "org", {"org_kind": "short"}),
    ("C9 video-conference account", "(ВКЗ) за обліковим записом: ", ".", "email", {}),
    ("C9 court-system account", "з використанням платформи «Електронний суд» - ", ".", "email", {}),
    ("C10 року народження", "неповнолітньої ОСОБА_3, ", " року народження", "dob", {"no_roku": True, "child": True}),
    ("C10 plain року after a person and a comma", "щодо ОСОБА_3, ", " року, уродженця", "dob", {"no_roku": True}),
    ("C10 plain року", "спростувати оприлюднену ", " року інформацію", "date_event", {"no_roku": True, **MASK}),
    ("C10 plain року beats an org cue", "розповсюджену відповідачем ", " року у мережі", "date_event", {"no_roku": True, **MASK}),
    # --- D. the general rules
    ("D dob after a person", "сина ОСОБА_3, ", ", у розмірі", "dob", {"child": True}),
    ("D dob after a role word", "позивач, ", ", 11.01.2024 звернувся", "dob", {}),
    ("D death, loose cue", "у зв`язку із загибеллю (смертю) ", " її чоловіка", "death", {"weak": True, **MASK}),
    ("D email", "електронну адресу ", ".", "email", {}),
    ("D email after a list of placeholders", "на електронні адреси ІНФОРМАЦІЯ_4, ІНФОРМАЦІЯ_5, ", ".", "email", {}),
    ("D email, cue on the right", "повідомлено також ", " (електронна пошта ОСОБА_9)", "email", {}),
    ("D url, site", "опублікована на веб-сайті ", " стаття", "url", {"url_kind": "site"}),
    ("D url, page", "відеозапис за посиланням ", " розміщено", "url", {"url_kind": "page"}),
    ("D url, profile", "у мережі Facebook за посиланням ", " поширено", "url", {"url_kind": "profile"}),
    ("D url, site by the words after the cue", "зазначивши інтернет-ресурс ", ";", "url", {"url_kind": "site"}),
    ("D url, page of a TV channel", "на офіційній інтернет сторінці телеканалу «Прямий» за посиланням ", ".", "url", {"url_kind": "page"}),
    ("D url, a video of a channel", "відео розміщене на youtube каналі за посиланням: ", ".", "url", {"url_kind": "page"}),
    ("D url, a list of links", "на інтернет ресурсах ІНФОРМАЦІЯ_5 та ", ", ІНФОРМАЦІЯ_8", "url", {"url_kind": "site"}),
    ("D url, video conference", "Відеоконференція буде проведена за допомогою додатку «EasyCon», за веб-посиланням ", "\nКопію",
     "url", MASK),
    ("D url, the link already given", "були два посилання, а саме для ІНФОРМАЦІЯ_3 та для ", "\nЗайшовши", "url", MASK),
    ("D url, a title after the link", "за посиланням: ІНФОРМАЦІЯ_3 «", " ?», наступного змісту", "url", MASK),
    # nick is masked since the review of v6: 11 of 18 hand-labelled accounts were right
    ("D nick, handle", "зареєстрований під ніком ", " у месенджері", "nick", {"nick_kind": "handle", **MASK}),
    ("D nick, page", "на власній сторінці у Facebook ", " поширив", "nick", {"nick_kind": "page", **MASK}),
    ("D e-mail in a mailing list", "1 - позивачу (office@firm.ua); 2 - відповідачу (", ");", "email", {}),
    ("D military_name", "у військовій частині ", " у АДРЕСА_1", "military_name", {}),
    ("D unit code on the left", "до військової частини НОМЕР_1 (", ") про визнання", "org", {"org_kind": "unit"}),
    ("D respondent list", "Відповідачі: ІНФОРМАЦІЯ_1, ІНФОРМАЦІЯ_2 та ", ".", "org", {"org_kind": "party"}),
    ("D short name", "до Міністерства оборони України (далі - ", ") про", "org", {"org_kind": "short"}),
    ("D позов … до", "з позовом в інтересах неповнолітньої дитини та її матері до ", " про", "org", {"org_kind": "party"}),
    ("D strict org", "- зобов'язати ", " підготувати", "org", {"org_kind": "party"}),
    ("D strict org, interested person", "заінтересована особа: ", ", про встановлення", "org", {"org_kind": "party"}),
    ("D loose org", "начальника відділу кадрів ", " ОСОБА_5", "org", {"org_kind": "party"}),
    # --- E. fallbacks
    ("E email_cue_far", "електронна адреса для листування та надсилання процесуальних документів учаснику ", ",", "email", {}),
    ("E email_neighbour_at", "1 - позивачу (a.b@ukr.net); 2 - ОСОБА_5 - ", " для відома", "email", {}),
    ("E email_neighbour_at_right", "2 - заявнику ОСОБА_1 (", "); 3 - Товариству (lawyer@firm.ua)", "email", {}),
    ("E email_neighbour_at_right, line break", "2 - заявнику ОСОБА_1 (", ")\n3 - Товариству (lawyer@firm.ua)", "email", {}),
    ("E email_right_cue", "адвоката Залізняка І.І. (", ") у режимі відеоконференції", "email", {}),
    ("E ident_cue", "підсистеми ЄСІТС (ідентифікатор ", ")", "ident", MASK),
    ("E death_verb_right", "Судом встановлено, що ", " її чоловік помер", "death", {}),
    ("E birth_verb_right", "Судом встановлено, що ", " у позивачки народився син", "dob", {}),
    ("E dob_person_bracket", "2 - заявнику ОСОБА_1 (", ") та", "dob", {}),
    ("E dob_person_alias", "ОСОБА_10 (ОСОБА_11), ", ", громадянина Німеччини", "dob", {}),
    ("E dob_person_dash", "ОСОБА_4 – ", ", уродженця смт", "dob", {}),
    ("E org_right_edrpou", "кошти перераховано ", " (код ЄДРПОУ 12345678)", "org", {"org_kind": "party"}),
    ("E org_holder", "документи знаходяться у володінні ", ", а саме:", "org", {"org_kind": "holder"}),
    ("E org_party_right", "Не погоджуючись, ", " звернулося до суду", "org", {"org_kind": "party"}),
    ("E org_party_left", "позовом ОСОБА_1 в інтересах дітей до ІНФОРМАЦІЯ_1, ", ".", "org", {"org_kind": "party"}),
    ("E org_doc_author", "на підставі довідки ", " від 05.07.2023 № 15", "org", {"org_kind": "party"}),
    ("E org_addressee", "ОСОБА_1 звернувся до ", " із запитом", "org", {"org_kind": "party"}),
    ("E org_official", "ухвалою слідчого судді ", " ОСОБА_3", "org", {"org_kind": "official", **MASK}),
    ("E nothing", "Суд вважає, що ", " не довів", "other", MASK),
    # --- F. the tail of a unit's name
    ("F unit head on the left", "до НОМЕР_2 прикордонного загону ", " (військової частини НОМЕР_1) про", "org", {"org_kind": "unit", **MASK}),
    ("F named after", "НОМЕР_2 прикордонний загін імені ", " (військова частина НОМЕР_1)", "org", {"org_kind": "unit", **MASK}),
]

# (left, right, the sub-type the rule in the id would give, what is expected instead)
NO_MATCH = [
    ("A not in quotes: no closing quote", "ТОВ «", " звернулося до суду", "quoted", None),
    ("A1 legal form before the quote", "дату народження засновника та директора підприємства ТОВ «", "»", "dob",
     ("quoted", {"quote_kind": "company"})),
    ("A2 legal form before the quote", "з офіційної електронної пошти ГО «", "»", "email", ("quoted", {"quote_kind": "company"})),
    ("A3 place of birth", "місце народження «", "»", "dob", ("quoted", {"quote_kind": "unknown"})),
    ("A3 дат inside a word", "має надати «", "»", "date_event", ("quoted", {"quote_kind": "unknown"})),
    ("A4 another sentence", "прізвище. Додаток «", "»", "quoted:person", ("quoted", {"quote_kind": "unknown"})),
    ("A5 lower-case abbreviation", "все ок «", "»", "quoted:formation", ("quoted", {"quote_kind": "unknown"})),
    ("A6 not an institution form", "некомерційної організації «", "»", "quoted:institution", ("quoted", {"quote_kind": "unknown"})),
    ("A7 company, not a bank", "АТ «", "»", "quoted:bank", ("quoted", {"quote_kind": "company"})),
    ("B1 no р.н.", "ОСОБА_1, ", ", уродженець", "dob:numeric", ("dob", {})),
    ("B2 another bracket", "за позовом ОСОБА_1 до ", " (далі - відповідач) про", "org:unit", ("org", {"org_kind": "party"})),
    ("C1 cue not adjacent", "уродженця м. Київ, зареєстрованого ", " за адресою", "place", None),
    ("C2 bracket without a requisite", "після смерті ОСОБА_1 (", ") відкрилась спадщина", "dob", ("death", {"weak": True})),
    ("C3 bare age", "до досягнення нею трирічного віку ", ", оскільки", "derived", None),
    ("C3 minority", "до неповноліття - ", ".", "derived", None),
    ("C4 cue not adjacent", "свідоцтво про смерть видане ", ",", "death:strong", ("death", {"weak": True})),
    ("C5 after a person and a comma", "ОСОБА_7, ", " помер ІНФОРМАЦІЯ_2", "death", ("dob", {})),
    ("C5 a date after the verb", "спадкодавець ОСОБА_3 ", " помер 16.01.2021", "death", ("dob", {})),
    ("C6 birth verb near", "ОСОБА_2 народився в Києві, а смерть настала ", ".", "death", ("dob", {})),
    ("C7 comma before the verb", "щодо ОСОБА_3 ", ", народився в м. Тернопіль", "dob:child", ("dob", {})),
    ("C8 no bracket", "позивач, ", ", 11.01.2024 звернувся", "org", ("dob", {})),
    ("C9 not an account", "через підсистему Електронний суд ", " подано заяву", "email", None),
    ("C10 no року", "розповсюджену відповідачем ", " у соціальній мережі", "date_event", ("org", {"org_kind": "party"})),
    ("D dob: role word inside a word", "про встановлення факту родинних відносин, ", " (АДРЕСА_1)", "dob", None),
    ("D email: post office", "засобами поштового зв`язку ", ".", "email", None),
    ("D email: Electronic court", "через систему Електронний суд ", " подав", "email", None),
    ("D email: the next placeholder is the address", "повідомлено також ", " (електронна пошта: ІНФОРМАЦІЯ_3)", "email", None),
    ("D url: a reference, not a link", "з посиланням на те, що ", " є відокремленим", "url", None),
    ("D nick: нік inside a word", "бойова техніка ", " була", "nick", None),
    ("D mailing list: the item is not closed", "1 - позивачу (office@firm.ua); 2 - відповідачу (", " про визнання", "email",
     ("org", {"org_kind": "party"})),
    ("D military_name: another частина", "інспектора чергової частини ", " ОСОБА_2", "military_name", None),
    ("D unit left: no code", "до військової частини ", " про визнання", "org", ("military_name", {})),
    ("D respondent: another word", "несе відповідальність ", " згідно", "org", None),
    ("D short name: no dash", "далі за текстом ", " зазначено", "org", None),
    ("D позов … до: another sentence", "позов задоволено; відповідно до ", " набрав чинності", "org", None),
    ("D loose org: dropped cue", "медичний центр ", " надав", "org", None),
    ("E email_cue_far: another sentence", "електронна адреса відома. Для листування та надсилання документів учаснику ", ",", "email", None),
    ("E email_neighbour_at: X names the body", "1 - позивачу (a.b@ukr.net); 2 - ОСОБА_5 - ", " (izmail@dpsu.gov.ua", "email", None),
    ("E email_neighbour_at_right: no address", "2 - заявнику ОСОБА_1 (", "); 3 - Товариству", "email", ("dob", {})),
    ("E email_right_cue: another phrase", "адвоката Залізняка І.І. (", ") у судовому засіданні", "email", None),
    ("E ident_cue: another word", "за ідентифікацією ", " встановлено", "ident", None),
    ("E death_verb_right: comma in the gap", "Судом встановлено, що ", " її чоловік, який помер", "death", None),
    ("E birth_verb_right: comma in the gap", "Судом встановлено, що ", " у лікарні, де народився син", "dob", None),
    ("E dob_person_dash: no requisite", "ОСОБА_4 – ", " звернувся до суду", "dob", ("org", {"org_kind": "party"})),
    ("E org_right_edrpou: named after", "загін імені ", " (код ЄДРПОУ 12345678)", "org", None),
    ("E org_holder: no noun of holding", "документи знаходяться ", ", а саме:", "org", None),
    ("E org_party_right: no verb of a party", "Не погоджуючись, ", " було встановлено", "org", None),
    ("E org_party_left: open name of no party", "Кабінету Міністрів України, ", " від", "org", None),
    ("E org_doc_author: no date or number", "на підставі довідки ", " про доходи", "org", None),
    ("E org_addressee: a date after «інформація від»", "недостовірною є інформація від ", " на веб-сайті", "org", None),
    ("E org_official: no name follows", "ухвалою слідчого судді ", " розглянуто", "org", None),
    ("F whole unit name", "за позовом ОСОБА_1 до ", " (військова частина НОМЕР_1) про", "org:masked", ("org", {"org_kind": "unit"})),
]


@pytest.mark.parametrize("left, right, subtype, hints", [m[1:] for m in MATCH], ids=[m[0] for m in MATCH])
def test_rule_matches(left, right, subtype, hints):
    st, got = info_subtype(left, right)
    assert st == subtype
    assert {k: got.get(k) for k in hints} == hints
    if "mask" not in hints:
        assert "mask" not in got  # a rule that is good enough for a label must not be masked


def _same(st: str, hints: dict, label: str) -> bool:
    """Does (st, hints) carry what the label «sub-type[:kind]» names?"""
    name, _, kind = label.partition(":")
    if st != name:
        return False
    if not kind:
        return True
    if kind == "masked":
        return bool(hints.get("mask"))
    if kind == "strong":
        return not hints.get("weak")
    return kind in (hints.get("quote_kind"), hints.get("org_kind"), hints.get("style")) or bool(hints.get(kind))


@pytest.mark.parametrize("left, kind", [
    ("формування податкового кредиту ТОВ «", "company"),          # no bank is a limited-liability company
    ("картки рахунків бухгалтерського обліку ДП «", "company"),
    ("оператора мобільного зв`язку ПрАТ «", "company"),           # the value side knows the operators
    ("які перебувають у володінні банку Акціонерного товариства «", "bank"),
    ("банківська картка № НОМЕР_1 належать банку АТ «", "bank"),
])
def test_bank_cue_before_a_legal_form(left, kind):
    assert info_subtype(left, "» за період") == ("quoted", {"quote_kind": kind})
    assert bank_in_quotes(left) is (kind == "bank")


def test_inside_quotes_knows_every_pair():
    assert all(inside_quotes(f"ТОВ {a}", f"{b} звернулося") for a, b in ("«»", '""', "„“", "“”", ("“ ", " ”")))
    assert not inside_quotes("ТОВ ", " звернулося") and not inside_quotes("ТОВ «", " звернулося")


@pytest.mark.parametrize("left, right, not_this, instead", [m[1:] for m in NO_MATCH], ids=[m[0] for m in NO_MATCH])
def test_rule_does_not_match(left, right, not_this, instead):
    st, got = info_subtype(left, right)
    assert not _same(st, got, not_this)
    if instead:
        assert st == instead[0]
        assert {k: got.get(k) for k in instead[1]} == instead[1]


@pytest.mark.parametrize("left, expected", [
    ("малолітньої ОСОБА_3, ", True), ("народилась донька ОСОБА_3, ", True), ("з малолітнім сином ОСОБА_4, ", True),
    # a cue further left than 45 characters counts when only names and dates stand in between
    ("малолітніх дітей: ОСОБА_3, ІНФОРМАЦІЯ_1, ОСОБА_5, ІНФОРМАЦІЯ_7, та ОСОБА_4, ", True),
    ("на утриманні малолітніх дітей, проживає разом із дружиною та працює водієм, ОСОБА_4, ", False),
    ("повнолітнього сина ОСОБА_2, ", False), ("є донькою ОСОБА_2, ", False), ("біля матері ОСОБА_1, ", False),
    ("до досягнення дитиною повноліття.\nСтягнення з ОСОБА_1, ", False), ("позивача ОСОБА_1, ", False),
])
def test_child_hint(left, expected):
    st, hints = info_subtype(left, ", уродженця")
    assert st == "dob"
    assert bool(hints.get("child")) is expected


@pytest.mark.parametrize("right, expected", [(" року народження", True), (" року, уродженця", True), (" р. народження", True),
                                             (", уродженця", False), (" рокував", False)])
def test_no_roku_hint(right, expected):
    assert bool(info_subtype("щодо ОСОБА_3, ", right)[1].get("no_roku")) is expected


def test_no_roku_for_the_other_dates():
    assert info_subtype("ОСОБА_2, яка померла ", " року у місті")[1] == {"no_roku": True}
    assert info_subtype("до досягнення дитиною повноліття, тобто до ", " року.")[1] == {"until": "majority", "no_roku": True}


# --- the vote over the occurrences of one index ----------------------------------------------
def test_choose_subtype_majority():
    assert choose_subtype([("dob", {}), ("death", {}), ("dob", {})]) == ("dob", {})
    assert choose_subtype([("org", {"org_kind": "party"}), ("dob", {}), ("other", MASK), ("org", {"org_kind": "unit"})])[0] == "org"


def test_choose_subtype_ties_go_to_the_last():
    assert choose_subtype([("dob", {}), ("death", {})])[0] == "death"
    assert choose_subtype([("death", {}), ("dob", {})])[0] == "dob"
    assert choose_subtype([("dob", {}), ("org", {}), ("death", {}), ("org", {}), ("dob", {})])[0] == "dob"


def test_choose_subtype_masked_votes_never_beat_unmasked_ones():
    event = ("date_event", {"no_roku": True, **MASK})
    assert choose_subtype([event, event, ("org", {"org_kind": "party"}), ("other", MASK)]) == ("org", {"org_kind": "party"})
    assert choose_subtype([("death", {"weak": True, **MASK}), ("death", {"weak": True, **MASK}), ("dob", {})]) == ("dob", {})
    # nothing unmasked: the same vote over the masked ones, «other» only when nothing else is left
    assert choose_subtype([("other", MASK), event, ("death", {"weak": True, **MASK}), event]) == event
    assert choose_subtype([("other", MASK), ("death", {"weak": True, **MASK})]) == ("death", {"weak": True, **MASK})
    assert choose_subtype([("other", MASK), ("other", MASK)]) == ("other", MASK)


def test_choose_subtype_takes_the_hints_of_the_last_winning_occurrence():
    votes = [("dob", {"child": True}), ("death", {}), ("dob", {"no_roku": True})]
    assert choose_subtype(votes) == ("dob", {"no_roku": True})
    # a masked occurrence of the winning sub-type does not lend its hints to a labelled one
    assert choose_subtype([("org", {"org_kind": "party"}), ("org", {"org_kind": "official", **MASK})]) == ("org", {"org_kind": "party"})
    assert choose_subtype([("quoted", {"quote_kind": "unknown", **MASK}), ("quoted", {"quote_kind": "bank"}),
                           ("quoted", {"quote_kind": "unknown", **MASK})]) == ("quoted", {"quote_kind": "bank"})


def test_choose_subtype_returns_a_copy_of_the_hints():
    votes = [("dob", {"child": True})]
    choose_subtype(votes)[1]["mask"] = True
    assert votes == [("dob", {"child": True})]


def _doc(text: str) -> dict:
    return {"text": text, "placeholders": [{"type": m.group(1), "idx": int(m.group(2)), "start": m.start(), "end": m.end()}
                                           for m in re.finditer(r"(ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_(\d+)", text)]}


def test_info_subtypes_votes_per_index():
    doc = _doc("Справа за позовом ОСОБА_1 до ІНФОРМАЦІЯ_1 про визнання дій протиправними. Суд встановив, що ІНФОРМАЦІЯ_2 помер "
               "ОСОБА_2, батько позивача. Позивач просив зобов'язати ІНФОРМАЦІЯ_1 нарахувати та виплатити допомогу. "
               "Як зазначено в рішенні, ІНФОРМАЦІЯ_1 заперечень не подав. Спадщина відкрилась після смерті ОСОБА_2, який помер "
               "ІНФОРМАЦІЯ_2 року у місті Києві, а його син ОСОБА_3, ІНФОРМАЦІЯ_3 року народження, проживає з матір'ю.")
    assert info_subtypes(doc) == {1: ("org", {"org_kind": "party"}),  # two org votes; the third occurrence has no cue
                                  2: ("death", {"no_roku": True}),  # two death votes, the hints of the last one
                                  3: ("dob", {"no_roku": True, "child": True})}


def test_info_subtypes_a_masked_occurrence_does_not_outvote_a_labelled_one():
    doc = _doc("Оприлюднену ІНФОРМАЦІЯ_1 року інформацію та поширену ІНФОРМАЦІЯ_1 року у мережі визнати недостовірною. "
               "За позовом ОСОБА_1 до ІНФОРМАЦІЯ_1 про захист честі.")
    assert info_subtypes(doc) == {1: ("org", {"org_kind": "party"})}
    only_dates = _doc("Оприлюднену ІНФОРМАЦІЯ_1 року інформацію визнати недостовірною. Суд вважає, що ІНФОРМАЦІЯ_2 не довів.")
    assert info_subtypes(only_dates) == {1: ("date_event", {"no_roku": True, **MASK}), 2: ("other", MASK)}


def test_info_subtypes_ignores_the_other_placeholder_types():
    assert info_subtypes(_doc("ОСОБА_1 проживає за адресою АДРЕСА_1, телефон НОМЕР_1.")) == {}


# --- the windows ------------------------------------------------------------------------------
def test_context_window_sizes():
    assert (INFO_LEFT, INFO_RIGHT) == (110, 70)


# cues of many rules, placed outside the window: they must not reach any rule
NOISE_LEFT = "ОСОБА_9 померла, електронна адреса: a@b.ua, на сайті ТОВ «Альфа» військова частина НОМЕР_9 (далі - відповідач, "
NOISE_RIGHT = " року народження р.н. (військова частина НОМЕР_9) помер ОСОБА_9 (код ЄДРПОУ 12345678) lawyer@firm.ua"


@pytest.mark.parametrize("left, right", [m[1:3] for m in MATCH + NO_MATCH], ids=[m[0] for m in MATCH + NO_MATCH])
def test_more_context_than_the_window_changes_nothing(left, right):
    left = ("У судовому засіданні сторони підтримали свої вимоги та просили їх задовольнити повністю; " * 2 + left)[-INFO_LEFT:]
    right = (right + "; у судовому засіданні сторони підтримали свої вимоги та просили їх задовольнити")[:INFO_RIGHT]
    assert info_subtype(NOISE_LEFT + left, right + NOISE_RIGHT) == info_subtype(left, right)


@pytest.mark.parametrize("left, right, subtype, hints", [m[1:] for m in MATCH], ids=[m[0] for m in MATCH])
def test_rules_cut_their_own_windows(left, right, subtype, hints):
    """A rule must not look further than the window it was measured with: neutral text around a short context,
    as in a real decision, leaves the answer as it is."""
    padded = info_subtype("Розглянувши матеріали, суд установив таке.\n" + left, right + " та інше, як зазначено вище")
    assert padded[0] == subtype
    assert {k: padded[1].get(k) for k in hints} == hints


# --- the cases of tests/test_orgs.py (test_info_subtype_cues, test_guess_org_case) ----------------
@pytest.mark.parametrize("left, right, expected", [
    ("за позовом ОСОБА_1 до ", " про визнання", "org"), ("надійшла касаційна скарга ", " на ухвалу", "org"),
    ("- зобов'язати ", " підготувати", "org"), ("визнати протиправними дії ", " щодо", "org"), ("перевіривши клопотання ", " про", "org"),
    ("ТОВ «", "» (код ЄДРПОУ", "quoted"), ("сина ОСОБА_3, ", ", у розмірі", "dob"), ("позивач, ", ", 11.01.2024 звернувся", "dob"),
    # v6: the «date» sub-type is gone; what its cue caught is a date computed from a child's birth date
    ("повноліття, тобто до ", ";", "derived"),
    ("уродженця ", ", зареєстрованого", "place"), ("у військовій частині ", " у АДРЕСА_1", "military_name"),
    ("ОСОБА_2, яка померла ", ",", "death"), ("електронну адресу ", ".", "email"),
])
def test_info_subtype_cues(left, right, expected):
    assert info_subtype(left, right)[0] == expected


@pytest.mark.parametrize("left, expected", [
    ("за позовом ОСОБА_1 до ІНФОРМАЦІЯ_1, ", "gent"), ("надійшла касаційна скарга ", "gent"), ("- зобов'язати ", "accs"),
    ("УХВАЛИВ: 1. Повернути ", "datv"), ("забезпечення, виданої ", "ablt"), ("(далі - ", "nomn"), ("третя особа - ", "nomn"),
    ("Не погодившись із рішенням, ", "nomn"), ("в особі ", "gent"),
    ("перевіривши касаційну скаргу ", "gent"), ("апеляційну скаргу ", "gent"), ("Згідно відповіді ", "gent"), ("продовжено ", "datv"),
    ("звернулась до суду ", "nomn"),
])
def test_guess_org_case(left, expected):
    assert guess_org_case(left) == expected


@pytest.mark.parametrize("left, expected", [
    # a name right after a noun is its attribute
    ("поновити на посаді стрільця відділення охорони взводу охорони ", "gent"), ("головою військово-лікарської комісії ", "gent"),
    ("з відділу прикордонної служби «ІНФОРМАЦІЯ_1» ", "gent"), ("перевіривши касаційну ", "gent"),
    # … but not after a word that starts a new phrase, a verb, or a noun the name stands next to in the nominative
    ("установив, що ", "nomn"), ("06 лютого 2024 року ", "nomn"), ("суд апеляційної інстанції відмовив ", "nomn"),
    ("за апеляційною скаргою представника компанії ", "nomn"), ("залучити як заінтересовану особу ", "nomn"),
    ("самостійних вимог щодо предмета спору ", "nomn"),
    # the cue stands before an open name or before another party of the list
    ("за позовом ОСОБА_1 до Державної прикордонної служби України, ", "gent"),
    ("до Міністерства оборони України, ІНФОРМАЦІЯ_4, Військової частини НОМЕР_1, ", "gent"),
    ("до Головного управління Пенсійного фонду України в Житомирській області, ", "gent"),
    ("зобов`язати Міністерство оборони України, ", "accs"),
    ("за касаційною скаргою ІНФОРМАЦІЯ_1 (військова частина НОМЕР_1) та ", "gent"),
    ("з позовом до ІНФОРМАЦІЯ_1 (далі - відповідач-1), ", "gent"),
    ("на стороні позивача: Житлова комісія ІНФОРМАЦІЯ_1, ", "nomn"), ("заінтересовані особи Міністерство оборони України, ", "nomn"),
    ("інформації, яка наявна у Діловодстві Верховного Суду, ", "nomn"),
    # a place after a noun
    ("під час проходження військово-лікарської комісії в ", "loct"), ("проходив військову службу у ", "loct"),
    ("військово-лікарською комісією при ", "loct"), ("позивач звернувся в ", "accs"),
    # the unit's own name agrees with «частина», whose code stands in the open
    ("за позовом ОСОБА_2 до військової частини НОМЕР_1 (НОМЕР_3) ", "gent"), ("зобов`язати військову частину НОМЕР_1 ", "accs"),
    ("військовою частиною НОМЕР_1 ", "ablt"), ("проходив службу у Військовій частині НОМЕР_1 ", "datv"),
    ("Військова частина НОМЕР_1 ", "nomn"),
    # «Військової частини НОМЕР_1 (X)»: the bracket keeps the nominative
    ("до Військової частини НОМЕР_1 (", "nomn"),
])
def test_guess_org_case_without_a_cue(left, expected):
    assert guess_org_case(left) == expected


def test_guess_org_case_reads_85_characters():
    left = "за позовом ОСОБА_1 до Державної прикордонної служби України, "
    assert guess_org_case("Не погодившись із рішенням, відповідач подав скаргу. Повернути " + "х" * 90 + " " + left) == guess_org_case(left)


def test_the_other_helpers_keep_their_signatures():
    assert number_subtype("РНОКПП ", "") == "rnokpp"
    assert number_subtype("номер кузова НОМЕР_1, реєстраційний номер ", "") == "vehicle"
    assert guess_case("стягнути з ") == "gent"
    assert guess_gender("позивачка ") == "femn"
