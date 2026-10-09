"""Institution, military-unit and company names for ІНФОРМАЦІЯ, declined by grammatical case.

ЄДРСР anonymises a military body that is a party to a serviceman's case (the ministry, a ТЦК, a unit, a medical
commission, a support command) and, in commercial cases, company names - usually inside quotes. An Org keeps the
words that decline (its head noun phrase) apart from the invariant tail, so that every mention can be rendered in
the case its context asks for: «Міністерства оборони України» after «позовом до», «Міністерство …» as a subject.

Names come from two places. The word lists below compose them with no outside data. When scripts/fetch_orgs.py has
built data/resources/org_names.parquet, a share of the civil bodies and company names are real ones instead
(silver.orgnames): names from court decisions for every kind of case, the Treasury's social-protection bodies,
prosecutor's offices and courts for servicemen's and social cases, the companies of the register of legal entities
for commercial cases. Military bodies always come from the lists: no public register names them.

Which kind of body is drawn depends on the slot (`org_kind` of the context rules) and on what the decision is about:
KIND_MIX holds the shares the audit of data v5 measured on hand-labelled slots. A name that already stands in the
decision is drawn again (Seen): the open mention is unlabelled, the same name as a labelled value would contradict it.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, TypeVar

import pymorphy3

from silver.fill.context import bank_in_quotes
from silver.orgnames import load_names
from silver.patterns import FULL_ANY, INI_ANY

_morph = pymorphy3.MorphAnalyzer(lang="uk")
REGIONS_LOC = ["Вінницькій", "Волинській", "Дніпропетровській", "Донецькій", "Житомирській", "Закарпатській", "Запорізькій",
               "Івано-Франківській", "Київській", "Кіровоградській", "Луганській", "Львівській", "Миколаївській", "Одеській",
               "Полтавській", "Рівненській", "Сумській", "Тернопільській", "Харківській", "Херсонській", "Хмельницькій",
               "Черкаській", "Чернівецькій", "Чернігівській"]


def _restore_case(src: str, out: str) -> str:
    if src.isupper():
        return out.upper()
    if src[:1].isupper():
        return out[:1].upper() + out[1:]
    return out


# court texts write the dative / genitive of these masculine nouns with -у, where the dictionary prefers -а / -ові
_NOUN_FORMS = {"центр": {"gent": "центру", "datv": "центру", "loct": "центрі"}, "відділ": {"gent": "відділу", "datv": "відділу"},
               "госпіталь": {"gent": "госпіталю", "datv": "госпіталю"}, "загін": {"gent": "загону", "datv": "загону"},
               "полк": {"gent": "полку", "datv": "полку"}, "батальйон": {"gent": "батальйону", "datv": "батальйону"}}


def _parses(word: str, part: str):
    word = word.rsplit("-", 1)[-1]  # «Військово-лікарська», «Квартирно-експлуатаційний»: the last part carries the form
    # the dictionary tags some nominatives without a case («відділ»: NOUN,inan masc)
    return [p for p in _morph.parse(word) if part in p.tag and ("nomn" in p.tag or p.tag.case is None)]


def _noun_form(p, case: str) -> str | None:
    """The singular form of the noun in `case`; court texts prefer «Департаменту» to the dictionary's «Департаментові»."""
    forms = [f.word for f in p.lexeme if case in f.tag and "plur" not in f.tag]
    if not forms:
        f = p.inflect({case})
        return f.word if f else None
    return next((w for w in forms if not w.endswith(("ові", "еві"))), forms[0])


def _inflect_word(word: str, part: str, case: str, agree: set[str]) -> str:
    """One word of a head phrase. `agree`: gender / number (and animacy for the accusative) of the head noun."""
    if "-" in word:
        first, last = word.rsplit("-", 1)
        return f"{first}-{_inflect_word(last, part, case, agree)}"
    if part == "NOUN" and word.lower() in _NOUN_FORMS and case in _NOUN_FORMS[word.lower()]:
        return _restore_case(word, _NOUN_FORMS[word.lower()][case])
    parses = _parses(word, part)
    if part == "NOUN":
        for p in parses:
            out = _noun_form(p, case)
            if out:
                return _restore_case(word, out)
        return word
    if agree:
        parses = [p for p in parses if agree <= set(p.tag.grammemes)] or parses
    want = ({"datv"} if case == "loct" else {case}) | agree  # «у Головному управлінні», not the dictionary's «Головнім»
    for p in parses:
        f = p.inflect(want)
        if f:
            return _restore_case(word, f.word)
    return word


def _noun_at(words: list[str]) -> int | None:
    """The head noun: the first word that is a noun and not also an adjective («Головний», «Ізмаїльський» are both)."""
    nouns = [i for i, w in enumerate(words) if _parses(w, "NOUN")]
    # «рада» is also a short adjective («рада бачити»): without the exception «Печерська міська рада» has no plain noun
    # and the surname-like «Печерська» is taken for the head
    return next((i for i in nouns if words[i].lower() == "рада" or not _parses(words[i], "ADJF")), nouns[0] if nouns else None)


@lru_cache(maxsize=50_000)
def inflect_phrase(head: str, case: str) -> str:
    """Decline a noun phrase («Головне управління», «Деснянський районний територіальний центр») as a whole:
    the noun by the case, the adjectives before it agreeing with it. Unknown words stay as they are."""
    if case == "nomn":
        return head
    words = head.split()
    noun_at = _noun_at(words)
    if noun_at is None:
        return head
    tag = _parses(words[noun_at], "NOUN")[0].tag
    agree = {g for g in (tag.number, tag.gender if tag.number == "sing" else None) if g}
    if case == "accs" and tag.animacy and tag.gender == "masc":
        agree.add(tag.animacy)  # «Головний центр» stays nominative-like in the accusative (inanimate)
    out = []
    for i, w in enumerate(words):
        if i == noun_at:
            out.append(_inflect_word(w, "NOUN", case, set()))
        elif i < noun_at and (_parses(w, "ADJF") or _parses(w, "PRTF")):
            out.append(_inflect_word(w, "ADJF" if _parses(w, "ADJF") else "PRTF", case, agree))
        else:
            out.append(w)
    return " ".join(out)


def inflect_adjectives(head: str, case: str, gender: str) -> str:
    """Adjectives that stand before an abbreviation of `gender`: «Печерський районний» + «ТЦК та СП» ->
    «Печерського районного» in the genitive. The abbreviation is inanimate: the accusative equals the nominative."""
    if case == "nomn":
        return head
    agree = {gender} | ({"inan"} if case == "accs" and gender == "masc" else set())
    return " ".join(_inflect_word(w, "ADJF", case, agree) if _parses(w, "ADJF") else w for w in head.split())


@dataclass(frozen=True)
class Org:
    head: str        # the words that decline: «Головне управління», «Військова частина»
    tail: str = ""   # invariant: «Національної поліції України», «А1234»
    fixed: bool = False  # abbreviations never decline: «ТЦК та СП», «в/ч А1234»
    agree: str = ""  # gender of the abbreviation in the tail when the head is adjectives only: «Печерський» + «РТЦК та СП»

    def render(self, case: str = "nomn") -> str:
        if self.fixed:
            head = self.head
        else:
            head = inflect_adjectives(self.head, case, self.agree) if self.agree else inflect_phrase(self.head, case)
        return f"{head} {self.tail}".strip()

    def __str__(self) -> str:
        return self.render()


_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'", "‘": "'"})
# the name inside the quotes of a full name: «Комтех Логістік» in «ТОВ «Комтех Логістік»», "Дельта" in «ПП "Дельта"»
_IN_QUOTES = re.compile(r"«([^«»]{2,})»|\"([^\"]{2,})\"")
MAX_REDRAWS = 5  # how many times a name that already stands in the decision is drawn again
# the forms of a body that are looked for in the decision: the nominative and the genitive first (nine open mentions in
# ten), the other cases because «Міністерству оборони України» names the same body
SEEN_CASES = ("nomn", "gent", "datv", "accs", "ablt", "loct")
T = TypeVar("T")


def inner_name(name: str) -> str:
    """The bare name of a body that carries one in quotes: «АТ «Укртелеком»» -> «Укртелеком». A body without one stays."""
    m = _IN_QUOTES.search(name)
    return (m.group(1) or m.group(2)).strip() if m else name


class Seen:
    """What a decision already names: its text (open names stay unlabelled there, so the same name as a labelled
    value would contradict them) and the names drawn for its other placeholders."""

    def __init__(self, text: str = ""):
        self.text = text.translate(_APOSTROPHES).lower()
        self.taken: set[str] = set()

    @staticmethod
    def forms(value: Org | str) -> list[str]:
        """The spellings to look for: every case form of a body, and the bare name in its quotes."""
        if isinstance(value, Org):
            names = list(dict.fromkeys(value.render(case) for case in SEEN_CASES))
            names += [a or b for a, b in _IN_QUOTES.findall(names[0])]
        else:
            names = [value]
        return [n.translate(_APOSTROPHES).lower() for n in names]

    def has(self, value: Org | str) -> bool:
        for name in self.forms(value):
            # (?<!\w)захід(?!\w): the whole name, not a part of a longer word («заходів» does not hold «Захід»)
            if name in self.taken or (name in self.text and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", self.text)):
                return True
        return False

    def add(self, value: Org | str) -> None:
        self.taken.update(self.forms(value))


def real_org(name: str) -> Org:
    """A real name in the nominative as an Org: the words up to its head noun decline, the rest is the tail.
    «Вінницька обласна прокуратура» declines whole; «Управління соціального захисту населення …» only in its
    first word. A name with no noun the dictionary knows never declines."""
    words = name.split()
    noun_at = _noun_at(words)
    if noun_at is None:
        return Org(name, fixed=True)
    return Org(" ".join(words[: noun_at + 1]), " ".join(words[noun_at + 1:]))


# how often a real name stands in for a composed one, when the table has names of that kind
P_REAL_COMPANY = 0.6
P_REAL_BODY = 0.5
P_REAL_SOCIAL = 0.6
P_EDR = 0.5          # in a commercial case: the share of real company names taken from the register of legal entities
P_UPPER = 0.13       # … written in capitals, as the register has them and one name in eight in the court decisions
_NAMES: dict | None = None


def usable_name(name: str) -> bool:
    """A real name the generator may draw: not a person's name («Балаклейский водоканал В.П. Трубчанинов», «Баліцького
    Миколи Васильовича» - the label layer would annotate it as ОСОБА inside the value) and not shorter than three
    letters («ТЦ»)."""
    return sum(c.isalpha() for c in name) >= 3 and not FULL_ANY.search(name) and not INI_ANY.search(name)


def _table() -> dict:
    global _NAMES
    if _NAMES is None:
        _NAMES = {key: [n for n in names if usable_name(n)] for key, names in load_names().items()}
    return _NAMES


# ---- inventories ----------------------------------------------------------------------------------------------
CITIES_LOC = ["місті Києві", "м. Києві", "м. Львові", "м. Одесі", "м. Харкові", "м. Дніпрі", "м. Запоріжжі", "м. Вінниці",
              "м. Полтаві", "м. Чернігові", "м. Черкасах", "м. Сумах", "м. Житомирі", "м. Рівному", "м. Луцьку", "м. Тернополі",
              "м. Кропивницькому", "м. Миколаєві", "м. Хмельницькому", "м. Ужгороді", "м. Чернівцях", "м. Івано-Франківську"]
CITIES_GEN = ["Києва", "Львова", "Одеси", "Харкова", "Дніпра", "Запоріжжя", "Вінниці", "Полтави", "Чернігова", "Черкас", "Сум",
              "Житомира", "Рівного", "Луцька", "Тернополя", "Кропивницького", "Миколаєва", "Хмельницького", "Ужгорода"]
DISTRICT_ADJ = ["Шевченківський", "Печерський", "Голосіївський", "Дарницький", "Деснянський", "Дніпровський", "Оболонський",
                "Подільський", "Святошинський", "Солом'янський", "Галицький", "Франківський", "Сихівський", "Київський",
                "Малиновський", "Приморський", "Суворовський", "Основ'янський", "Новобаварський", "Миколаївський",
                "Вінницький", "Полтавський", "Чернігівський", "Бориспільський", "Броварський", "Бучанський", "Обухівський",
                "Білоцерківський", "Уманський", "Ніжинський", "Ковельський", "Стрийський", "Дрогобицький", "Кременчуцький",
                "Криворізький", "Павлоградський", "Мелітопольський", "Ізмаїльський", "Хмільницький", "Коломийський"]
# civil bodies: head + tail that takes the place
BODIES = [("Головне управління", "Державної податкової служби"), ("Головне управління", "Пенсійного фонду України"),
          ("Головне управління", "Національної поліції"), ("Головне управління", "Державної міграційної служби України"),
          ("Головне управління", "Держпраці"), ("Головне управління", "Держгеокадастру"), ("Управління", "Державної казначейської служби України"),
          ("Головне управління", "Державної служби України з надзвичайних ситуацій"), ("Управління", "Служби безпеки України"),
          ("Регіональний сервісний центр", "Головного сервісного центру МВС"), ("Управління", "Державної виконавчої служби"),
          ("Міжрегіональне управління", "Національного агентства України з питань державної служби")]
NATIONAL = [("Міністерство", "юстиції України"), ("Міністерство", "соціальної політики України"), ("Пенсійний фонд", "України"),
            ("Державна міграційна служба", "України"), ("Державна податкова служба", "України"), ("Національна поліція", "України"),
            ("Фонд", "соціального страхування України"), ("Державна служба", "України з питань праці"),
            ("Національне агентство", "України з питань державної служби"), ("Державна казначейська служба", "України")]
MILITARY_CENTRAL = [("Міністерство", "оборони України"), ("Міністерство", "оборони України"), ("Генеральний штаб", "Збройних Сил України"),
                    ("Командування", "Сухопутних військ Збройних Сил України"), ("Командування", "Сил логістики Збройних Сил України"),
                    ("Командування", "Медичних сил Збройних Сил України"), ("Командування", "Сил підтримки Збройних Сил України"),
                    ("Командування", "Повітряних Сил Збройних Сил України"), ("Національна гвардія", "України"),
                    ("Державна прикордонна служба", "України"), ("Служба безпеки", "України"), ("Головне управління", "розвідки Міністерства оборони України"),
                    ("Центральне управління", "матеріального забезпечення Сухопутних військ Збройних Сил України"),
                    ("Головний центр", "підготовки особового складу Збройних Сил України"),
                    ("Головне управління", "персоналу Генерального штабу Збройних Сил України")]
MOU_DEPTS = [("Департамент", "соціального захисту військовослужбовців та пенсіонерів Міністерства оборони України"),
             ("Відділ", "організаційного та аналітичного забезпечення Міністерства оборони України"),
             ("Департамент", "кадрової політики Міністерства оборони України"),
             ("Департамент", "фінансів Міністерства оборони України"),
             ("Управління", "правового забезпечення Міністерства оборони України"),
             ("Головне управління", "військової юстиції Збройних Сил України"),
             ("Департамент", "військової освіти та науки Міністерства оборони України")]
MEDICAL = [("Військово-лікарська комісія", "Центрального госпіталю Міністерства оборони України"),
           ("Військово-лікарська комісія", "Національного військово-медичного клінічного центру «Головний військовий клінічний госпіталь»"),
           ("Центральна військово-лікарська комісія", "Міністерства оборони України"),
           ("Військово-лікарська комісія", "Військово-медичного клінічного центру {region_gen} регіону"),
           ("Національний військово-медичний клінічний центр", "«Головний військовий клінічний госпіталь»"),
           ("Військово-медичний клінічний центр", "{region_gen} регіону"), ("Військовий госпіталь", "(військова частина {unit})"),
           ("Гарнізонна військово-лікарська комісія", "{city_gen}")]
UNIT_NAMES = ["Захисник", "Січ", "Варта", "Десна", "Воля", "Холодний Яр", "Азов", "Омега", "Чоп", "Мостиська", "Оберіг", "Світязь",
              "Волошине", "Карпати", "Сокіл", "Буревій", "Гвардія", "Патріот", "Либідь", "Дніпро", "Степ", "Полісся", "Тиса",
              "Рубіж", "Кордон", "Козак", "Запоріжжя", "Донбас", "Говерла", "Хортиця"]
UNIT_TYPES = [("Військовий навчальний центр", ""), ("Окремий батальйон охорони", ""), ("Навчальний центр", ""), ("Центр підготовки", ""),
              ("Окрема механізована бригада", ""), ("Окрема бригада", "територіальної оборони"), ("Окремий полк", ""),
              ("Загін", "спеціального призначення"), ("Прикордонний загін", ""), ("Відділення", "прикордонної служби"),
              ("Окремий стрілецький батальйон", ""), ("Зенітний ракетний полк", ""), ("Окрема артилерійська бригада", ""),
              ("Окремий батальйон", "територіальної оборони"), ("Окрема штурмова бригада", "")]
REGIONS_GEN = ["Західного", "Північного", "Південного", "Східного", "Центрального"]
# company names
NAME_WORDS = ["Комтех", "Гровінг", "Дельта", "Будсоюз", "Промінь", "Світанок", "Либідь", "Зоря", "Меридіан", "Вектор", "Оріон",
              "Атлант", "Фаворит", "Гарант", "Прогрес", "Карпати", "Оберіг", "Корвет", "Галичина", "Поділля", "Славутич", "Терра",
              "Нова Хвиля", "Альфа", "Омега", "Фенікс", "Титан", "Кристал", "Моноліт", "Еталон", "Стандарт", "Імперіал", "Авангард",
              "Континент", "Світ", "Либідь", "Дніпро", "Таврія", "Полісся", "Волинь", "Буковина", "Берегиня", "Злагода"]
NAME_TAILS = ["Логістік", "Стейт", "Груп", "Трейд", "Інвест", "Сервіс", "Технології", "Системи", "Партнерс", "Солюшнс", "Капітал",
              "Девелопмент", "Енерджі", "Агро", "Фуд", "Текстиль", "Буд", "Пром", "Транс", "Техніка", "Плюс", "Ресурс", "Консалт",
              "Експорт", "Маркет", "Холдинг", "Центр", "Україна", "ЛТД"]
COMPOUND_PREFIX = ["Укр", "Київ", "Дніпро", "Захід", "Схід", "Агро", "Пром", "Буд", "Енерго", "Транс", "Тех", "Еко", "Мега",
                   "Інтер", "Євро", "Полі", "Авто", "Метал", "Хім", "Електро", "Гео", "Фарм", "Зерно", "Нафто", "Газ", "Водо"]
COMPOUND_ROOT = ["буд", "торг", "сервіс", "інвест", "трейд", "альянс", "стандарт", "плюс", "ресурс", "капітал", "консалт", "агро",
                 "метал", "пром", "постач", "маркет", "імпекс", "експорт", "комплект", "лізинг", "холдинг", "центр", "систем",
                 "техніка", "трейдинг", "груп", "енерго", "монтаж", "проект", "дорбуд", "зв'язок", "сервіс"]
LEGAL_FORMS = [("Товариство з обмеженою відповідальністю", "ТОВ", 55), ("Приватне акціонерне товариство", "ПрАТ", 10),
               ("Публічне акціонерне товариство", "ПАТ", 6), ("Акціонерне товариство", "АТ", 6), ("Приватне підприємство", "ПП", 8),
               ("Державне підприємство", "ДП", 5), ("Комунальне підприємство", "КП", 5), ("Фермерське господарство", "ФГ", 3),
               ("Товариство з додатковою відповідальністю", "ТДВ", 2)]


# --- the unit's own name: the slot «【X】 (військова частина НОМЕР_1)», where the code stays in the text ---------------
BRIGADES = [("окрема механізована бригада", ""), ("окрема механізована бригада", ""), ("окрема мотопіхотна бригада", ""),
            ("окрема танкова бригада", ""), ("окрема артилерійська бригада", ""), ("окрема десантно-штурмова бригада", ""),
            ("окрема штурмова бригада", ""), ("окрема гірсько-штурмова бригада", ""), ("окрема аеромобільна бригада", ""),
            ("окрема бригада", "морської піхоти"), ("зенітна ракетна бригада", ""), ("окрема бригада", "армійської авіації"),
            ("бригада", "оперативного призначення Національної гвардії України"), ("окрема бригада", "охорони громадського порядку")]
REGIMENTS = [("окремий полк", "зв'язку"), ("окремий полк", "спеціального призначення"), ("окремий полк", "оперативного забезпечення"),
             ("зенітний ракетний полк", ""), ("окремий стрілецький полк", ""), ("полк", "Національної гвардії України"),
             ("окремий полк", "радіоелектронної боротьби"), ("окремий інженерний полк", "")]
PATRONS = ["імені короля Данила", "імені князя Костянтина Острозького", "імені кошового отамана Івана Сірка",
           "імені гетьмана Івана Виговського", "імені князя Романа Великого", "імені Чорних Запорожців", "імені Січових Стрільців"]
ARMS = ["Сухопутних військ Збройних Сил України", "Десантно-штурмових військ Збройних Сил України",
        "Сил територіальної оборони Збройних Сил України", "Повітряних Сил Збройних Сил України", "Збройних Сил України"]
BORDER_ADJ = ["Мукачівський", "Чопський", "Львівський", "Луцький", "Чернівецький", "Одеський", "Харківський", "Сумський",
              "Чернігівський", "Житомирський", "Подільський", "Ізмаїльський", "Білгород-Дністровський", "Краматорський",
              "Донецький", "Луганський", "Херсонський", "Бердянський"]
DIRECTIONS_NEUT = ["Західне", "Південне", "Східне", "Північне", "Центральне"]
DIRECTORATES = [("{dir} регіональне управління", "Державної прикордонної служби України"),
                ("{dir} регіональне управління", "Державної прикордонної служби України"),
                ("{dir} територіальне управління", "Національної гвардії України"),
                ("{dir} оперативно-територіальне об'єднання", "Національної гвардії України"),
                ("Регіональне управління", "Сил територіальної оборони «{name}»"), ("Оперативне командування", "«{name}»"),
                ("Повітряне командування", "«{name}»")]
DIRECTION_NAMES = ["Захід", "Південь", "Північ", "Схід", "Центр"]
TRAINING = [("{n} навчальний центр", "{arm}"), ("{n} навчальний центр", ""), ("{n} об'єднаний навчально-тренувальний центр", "{arm}"),
            ("Навчальний центр", "підготовки молодших спеціалістів Державної прикордонної служби України"),
            ("Навчальний центр", "Національної гвардії України"), ("{n} центр", "підготовки підрозділів {arm}"),
            ("Міжнародний центр", "миротворчості та безпеки"), ("{n} окремий навчальний батальйон", "")]
# --- names that stand inside the quotes of the text ---------------------------------------------------------------
# a formation after «оперативного командування», «ОТУ», «угруповання військ»: a direction or a region …
FORMATION_DIRECTIONS = ["Захід", "Південь", "Північ", "Схід", "Центр", "Захід", "Південь", "Північ", "Схід", "Донецьк", "Луганськ", "Таврія",
                        "Хортиця", "Одеса", "Харків", "Суми", "Слобожанщина", "Волинь", "Запоріжжя", "Київ", "Херсон", "Придніпров'я"]
# … after «відділ прикордонної служби», «ВПС», «відділення інспекторів прикордонної служби»: the place of the section
FORMATION_PLACES = ["Чоп", "Мостиська", "Шегині", "Краковець", "Рава-Руська", "Ужгород", "Тиса", "Солотвино", "Порубне", "Мамалига",
                    "Могилів-Подільський", "Кучурган", "Рені", "Ізмаїл", "Гоптівка", "Бачівськ", "Нові Яриловичі", "Доманове", "Ягодин",
                    "Устилуг", "Грушів", "Смільниця", "Вилок", "Лужанка", "Старокозаче", "Сокиряни", "Великі Мости", "Амбуків"]
FORMATION_NAMES = sorted(set(FORMATION_DIRECTIONS + FORMATION_PLACES))
# «відділу прикордонної служби «», «ВПС «», «впс «»: a border-service section, named by its place
_BORDER_CUE = re.compile(r"прикордонн|\bВПС\b|\bВІПС\b|пункт\w*\s+пропуск", re.I)
# «оперативного командування «», «ОТУ «», «угруповання військ «», «ОК «»: a formation, named by a direction or a region
_COMMAND_CUE = re.compile(r"командуванн|угруп[оу]ванн|\bОТУ\b|\bОСУВ\b|\bОК\b|\bОУВ\b|управлінн", re.I)
# bank names as the decisions write them inside quotes, with the legal form that usually precedes
BANKS = [("АТ КБ", "ПриватБанк"), ("АТ КБ", "ПриватБанк"), ("АТ", "Ощадбанк"), ("АТ", "Державний ощадний банк України"),
         ("АТ", "Універсал Банк"), ("АТ", "Райффайзен Банк"), ("АТ", "ПУМБ"), ("АТ", "Перший Український Міжнародний Банк"),
         ("АТ", "Сенс Банк"), ("АТ", "Альфа-Банк"), ("АТ", "ОТП Банк"), ("АТ", "Укрсиббанк"), ("АБ", "Укргазбанк"), ("АТ", "Укрексімбанк"),
         ("АТ", "Кредобанк"), ("АТ", "ПроКредит Банк"), ("АТ", "Акцент-Банк"), ("АТ", "А-Банк"), ("АТ", "Банк Кредит Дніпро"),
         ("АТ", "Ідея Банк"), ("АТ", "Таскомбанк"), ("АТ", "МТБ Банк"), ("АБ", "Південний"), ("АТ", "Мегабанк"), ("АТ", "Банк Форвард"),
         ("ПАТ", "Дельта Банк"), ("АТ", "Банк Восток"), ("АТ", "Правекс Банк"), ("АТ", "Креді Агріколь Банк"), ("АТ", "Банк Альянс"),
         ("АТ", "Полтава-банк"), ("АТ", "Радабанк"), ("АТ", "Піреус Банк МКБ"), ("АКБ", "Індустріалбанк"), ("АТ КБ", "Глобус"),
         ("ПАТ", "Банк Фінанси та Кредит"), ("ПАТ КБ", "Надра"), ("АТ", "КредитВест Банк"), ("АТ", "Банк інвестицій та заощаджень")]
MOBILE_OPERATORS = [("ПрАТ", "Київстар"), ("ПрАТ", "Київстар"), ("ПрАТ", "ВФ Україна"), ("ТОВ", "лайфселл"), ("ТОВ", "Інтертелеком"),
                    ("АТ", "Укртелеком"), ("ПрАТ", "Датагруп")]
# «оператора мобільного зв'язку ПрАТ «», «операторів телекомунікацій: ПрАТ «…», ТОВ «», «сім-карту мобільного оператора ПрАТ "»:
# the company in the quotes is a mobile operator (35 of 36 read)
_OPERATOR_CUE = re.compile(r"оператор\w*|мобільн\w+\s+зв.язк|телекомунікац|рухом\w+\s+зв|електронн\w+\s+комунікац|(?:сім|sim)-?\s?карт", re.I)
# «ПрАТ «», «ТОВ "»: the legal form right before the opening quote, which the operator's name must fit
_FORM_BEFORE_QUOTE = re.compile(r"(?<!\w)(ПрАТ|ТОВ|АТ)\s*[«\"„“]\s*$")
# «оперативного командування «», «ОК «»: one of the four operational commands …
_OPERATIONAL_COMMAND = re.compile(r"оперативн\w+\s+командуванн\w+|\bОК\b", re.I)
OPERATIONAL_COMMANDS = ["Захід", "Південь", "Північ", "Схід"]
# … «повітряного командування «», «ПвК «»: one of the four air commands
_AIR_COMMAND = re.compile(r"повітрян\w+\s+командуванн\w+|\bПвК\b", re.I)
AIR_COMMANDS = ["Захід", "Південь", "Центр", "Схід"]
MEDICAL_FORMS = [("Комунальне некомерційне підприємство", "КНП"), ("Комунальний заклад", "КЗ"), ("Комунальна установа", "КУ")]
# short names that stand after «(далі - …)»; none of them declines
SHORT_MILITARY = ["ТЦК та СП", "ТЦК та СП", "РТЦК та СП", "ТЦК", "в/ч {unit}", "в/ч {unit}", "Міноборони", "МОУ", "ВЛК", "ЦВЛК",
                  "Генштаб ЗСУ", "ГШ ЗСУ", "НГУ", "ДПСУ", "КЕВ м. {city}", "{district} РТЦК та СП", "{district} ТЦК та СП",
                  "{region} ОТЦК та СП", "Адміністрація ДПСУ", "ГУ НГУ", "ЗСУ"]
SHORT_CIVIL = ["ГУ ПФУ в {region_loc} області", "ГУ ПФУ в {region_loc} області", "ГУ ДПС у {region_loc} області",
               "ГУНП в {region_loc} області", "ГУ ДМС у {region_loc} області", "УДКСУ у {region_loc} області", "ПФУ", "ДПС", "ДМС",
               "Мін'юст", "Мінсоцполітики", "{district} РДА", "УСЗН", "ГУ ДСНС у {region_loc} області", "ДКСУ", "ГУ Держпраці"]

# shares of military / civil / company bodies by the kind of slot and by what the decision is about (audit of data v5:
# 281 / 1 / 1 hidden bodies of known kind in servicemen's decisions, 12 / 3 / 0 in the others)
KINDS = ("military", "civil", "company")
KIND_MIX = {"party": {"military": (95, 4, 1), "commercial": (5, 15, 80), "other": (70, 25, 5)},
            # a holder of documents, a place of work or custody: banks, hospitals, police units, mobile operators
            "holder": {"military": (60, 25, 15), "commercial": (10, 45, 45), "other": (10, 45, 45)}}
OFFICIAL_MIX = (("court", 40), ("prosecutor", 35), ("bank", 25))
# what the words before a slot say about the body in it; the first cue that matches decides
# «на банківську картку», «з рахунку», «суддя», «слідчого судді», «прокурор», «прокурора відділу»
_OFFICIAL_CUES = (("bank", re.compile(r"картк|рахун|банк", re.I)), ("court", re.compile(r"судд", re.I)), ("prosecutor", re.compile(r"прокур", re.I)))
# «КНП «», «некомерційного підприємства «», «лікарні «»: a hospital; «ДУ «», «державній установі «»: mostly a penal
# institution; «КЗ «», «закладу освіти «»: mostly a school
_INSTITUTION_CUES = ((re.compile(r"відбува\w+\s+покаранн|виконання покарань|засуджен", re.I), (("penal", 1),)),
                     (re.compile(r"\bКНП\b|некомерційн|медичн|лікарн|охорони здоров", re.I), (("hospital", 1),)),
                     (re.compile(r"\bДУ\b|державн\w+\s+установ|виконання покарань", re.I), (("penal", 75), ("hospital", 25))),
                     (re.compile(r"освіт|\bЗЗСО\b|\bЗДО\b|навчальн", re.I), (("school", 1),)),
                     (re.compile(r"\bКЗ\b|\bКУ\b|комунальн\w+\s+(?:заклад|установ)", re.I), (("hospital", 55), ("school", 45))))
INSTITUTION_MIX = (("hospital", 60), ("school", 20), ("penal", 20))
CUE_WINDOW = 45  # characters before the slot in which a cue counts
ORG_KINDS = ("party", "unit", "short", "holder", "official")
# the service a unit belongs to, when the text names it right after the slot or after the bracket with the unit's code:
# «【X】 Державної прикордонної служби України», «【X】 (військова частина НОМЕР_1) Національної гвардії України»
_PARENT_RIGHT = re.compile(r"^\s*\)?\s*(?:\([^()]{0,50}\)\s*)?(?:(?:прикордонн\w+\s+)?заг[оі]н\w*\s+)?"
                           r"(Державної прикордонної служби|Національної гвардії|Збройних Сил|Міністерства оборони)", re.I)
PARENTS = {"державної прикордонної служби": "border", "національної гвардії": "guard", "збройних сил": "army", "міністерства оборони": "army"}
# what a holder slot is, by the words before it: «в умовах гауптвахти 【X】», «на гауптвахті 【X】» …
_GUARDHOUSE_CUE = re.compile(r"гауп\w*вахт", re.I)
# … «у провадженні 【X】 перебувають матеріали кримінального провадження»: the body that investigates …
_INVESTIGATION_CUE = re.compile(r"(?:у|в)\s+провадженні(?:\s+\w+){0,2}\s*$", re.I)
# … «у приміщенні 【X】 повістку», «до цього РТЦК та СП»: a draft office
_DRAFT_OFFICE_CUE = re.compile(r"ТЦК|повістк|комплектування", re.I)
# … «у володінні 【X】», «зберігаються в 【X】», «які належать 【X】»: a holder of records - only here a bank or a mobile operator fits
_RECORDS_CUE = re.compile(r"володінн|розпорядженн|користуванн|зберіга|належ", re.I)

# ---- the short name after «(далі - …)», built from the body the bracket follows ----
# «військової частини А5885 (далі - 【в/ч А5885】)»
_ANTECEDENT_UNIT = re.compile(r"(?:військов\w+\s+частин\w+|в/ч)\s+([А-ЯІA-Z]{1,2}\s?\d{4})(?!\d)", re.I)
# «Шевченківського районного територіального центру комплектування та соціальної підтримки (далі - 【ТЦК та СП】)»
_ANTECEDENT_TCK = re.compile(r"(обласн\w+\s+|районн\w+\s+(?:у\s+місті\s+\w+\s+)?)?територіальн\w+\s+центр\w*\s+комплектування|"
                             r"(?<!\w)[РО]?ТЦК(?!\w)", re.I)
# «Військово-лікарської комісії … (далі - 【ВЛК】)», «Центральної військово-лікарської комісії (далі - 【ЦВЛК】)»
_ANTECEDENT_VLK = re.compile(r"(центральн\w+\s+)?військово\s?-\s?лікарськ\w+\s+комісі", re.I)
# «Квартирно-експлуатаційного відділу м. Ужгорода (далі - 【КЕВ м. Ужгорода】)»
_ANTECEDENT_KEV = re.compile(r"квартирно-експлуатаційн\w+\s+відділ\w*\s+(м\.\s?[А-ЯІЇЄҐ][\w'’-]+)", re.I)
# «Головного управління Пенсійного фонду України в Одеській області (далі - 【ГУ ПФУ в Одеській області】)»
_ANTECEDENT_REGIONAL = re.compile(r"Головн\w+\s+управлінн\w+\s+(Пенсійного фонду України|Державної податкової служби|Національної поліції|"
                                  r"Державної міграційної служби України)\s+"
                                  r"((?:в|у)\s+(?:[А-ЯІЇЄҐ][\w'’-]+\s+області|м(?:\.|істі)\s?[А-ЯІЇЄҐ][\w'’-]+))")
_REGIONAL_SHORT = {"Пенсійного фонду України": ("ГУ ПФУ", "ГУ ПФУ", "ГУПФУ"), "Державної податкової служби": ("ГУ ДПС",),
                   "Національної поліції": ("ГУНП", "ГУ НП"), "Державної міграційної служби України": ("ГУ ДМС",)}
# «Товариства з обмеженою відповідальністю «Комтех» (далі - 【ТОВ «Комтех»】)»
_ANTECEDENT_COMPANY = re.compile(r"(Товариств\w+ з обмеженою відповідальністю|Товариств\w+ з додатковою відповідальністю|"
                                 r"Приватн\w+ акціонерн\w+ товариств\w+|Публічн\w+ акціонерн\w+ товариств\w+|Акціонерн\w+ товариств\w+|"
                                 r"Приватн\w+ підприємств\w+|Державн\w+ підприємств\w+|Комунальн\w+ підприємств\w+|"
                                 r"Фермерськ\w+ господарств\w+)\s*([«\"][^«»\"]{2,60}[»\"])\s*$", re.I)
# the abbreviation of a legal form, by the words that tell it apart (the first that matches)
_FORM_ABBR = (("додатковою", "ТДВ"), ("обмеженою", "ТОВ"), ("приватн\\w+ акціонерн", "ПрАТ"), ("публічн", "ПАТ"), ("акціонерн", "АТ"),
              ("приватн", "ПП"), ("державн", "ДП"), ("комунальн", "КП"), ("фермерськ", "ФГ"))
# a central body that is itself the party: its usual abbreviations. Only when the name starts the phrase before the bracket:
# «Департаменту кадрової політики Міністерства оборони України (далі - …)» is not the ministry
_ANTECEDENT_CENTRAL = ((re.compile(r"Міністерств\w+\s+оборони\s+України", re.I), ("Міноборони", "МОУ", "Міноборони України", "МО України")),
                       (re.compile(r"Генеральн\w+\s+штаб\w*\s+Збройних\s+Сил\s+України", re.I), ("Генштаб ЗСУ", "ГШ ЗСУ", "Генеральний штаб")),
                       (re.compile(r"Адміністраці\w+\s+Державної\s+прикордонної\s+служби\s+України", re.I), ("Адміністрація ДПСУ", "АДПСУ")),
                       (re.compile(r"Державн\w+\s+прикордонн\w+\s+служб\w+\s+України", re.I), ("ДПСУ", "Держприкордонслужба")),
                       (re.compile(r"Національн\w+\s+гварді\w+\s+України", re.I), ("НГУ", "Нацгвардія")),
                       (re.compile(r"Служб\w+\s+безпеки\s+України", re.I), ("СБУ", "СБ України")),
                       (re.compile(r"Пенсійн\w+\s+фонд\w*\s+України", re.I), ("ПФУ", "ПФ України")),
                       (re.compile(r"Національн\w+\s+поліці\w+\s+України", re.I), ("НПУ", "Нацполіція")))
# where the phrase before «(далі - …)» starts: after the last comma, colon, line break or «до» («позовом до …»)
_PHRASE_START = re.compile(r"[,:;\n]\s*|(?<!\w)(?:до|від|та|і|з|із)\s+(?=[А-ЯІЇЄҐ«]|військов)")


def _region_adj(loc: str, ending: str) -> str:
    """«Одеській» + «а» -> «Одеська»: the region adjectives are listed in the locative."""
    return loc[:-2] + ending


def _femn(adj: str) -> str:
    """«Броварський» -> «Броварська» (the district adjectives are listed in the masculine)."""
    return adj[:-2] + "а" if adj.endswith("ий") else adj


def _neut(adj: str) -> str:
    """«Броварський» -> «Броварське»: «Броварське районне управління поліції»."""
    return adj[:-2] + "е" if adj.endswith("ий") else adj


class OrgGenerator:
    def __init__(self, rng: random.Random, names: dict | None = None):
        """`names`: (source, kind) -> real names (silver.orgnames.load_names); by default the table on disk,
        {} for composed names only."""
        self.rng = rng
        self.names = _table() if names is None else names
        self.used: Counter = Counter()  # where the names came from: «company_court», «body_composed», …
        self.drawn: Counter = Counter()  # kinds of the bodies drawn («party_military», «holder_civil») and redraws

    def fresh(self, draw: Callable[[], T], seen: Seen | None) -> T:
        """A value of `draw` that the decision does not name yet; after MAX_REDRAWS tries the last one is kept.
        Every try goes through the generator's rng, so the result depends on the seed and the decision only."""
        value = draw()
        if seen is None:
            return value
        for _ in range(MAX_REDRAWS):
            if not seen.has(value):
                break
            self.drawn["redraw"] += 1
            value = draw()
        else:
            self.drawn["collision_kept"] += seen.has(value)
        seen.add(value)
        return value

    def _real(self, source: str, kind: str, p: float) -> str | None:
        pool = self.names.get((source, kind))
        return self.rng.choice(pool) if pool and self.rng.random() < p else None

    # --- civil ----------------------------------------------------------------------------------------------
    def civil_body(self, registers: bool = False) -> Org:
        """`registers`: the decision is a serviceman's or a social case, where the Treasury's social-protection
        bodies and prosecutor's offices may stand. Courts are never drawn: decisions name courts in the open
        everywhere (435,854 open mentions in data v5), a labelled court contradicts every one of them."""
        r = self.rng
        kinds, weights = ["gu", "national", "social", "council"], [55, 15, 20, 10]
        if registers and self.names.get(("treasury", "prosecutor")):
            kinds, weights = kinds + ["prosecutor"], weights + [5]
        kind = r.choices(kinds, weights)[0]
        if kind == "prosecutor":
            self.used["body_treasury_prosecutor"] += 1
            return real_org(r.choice(self.names[("treasury", "prosecutor")]))
        if kind in ("gu", "national") and (name := self._real("court", "body", P_REAL_BODY)):
            self.used["body_court"] += 1
            return real_org(name)
        if kind == "social" and registers and (name := self._real("treasury", "social", P_REAL_SOCIAL)):
            self.used["body_treasury_social"] += 1
            return real_org(name)
        self.used["body_composed"] += 1
        if kind == "gu":
            head, body = r.choice(BODIES)
            place = r.choice([f"в {r.choice(REGIONS_LOC)} області", f"у {r.choice(CITIES_LOC)}", f"у {r.choice(REGIONS_LOC)} області"])
            return Org(head, f"{body} {place}")
        if kind == "national":
            return Org(*r.choice(NATIONAL))
        if kind == "social":
            d = _femn(r.choice(DISTRICT_ADJ))
            return r.choice([Org("Департамент", f"соціального захисту населення {r.choice(REGIONS_LOC).replace('ій', 'ої')} обласної державної адміністрації"),
                             Org("Управління", f"праці та соціального захисту населення {inflect_phrase(d + ' районна державна адміністрація', 'gent')}"),
                             Org("Управління", f"соціального захисту населення {inflect_phrase(d + ' районна в місті Києві державна адміністрація', 'gent')}"),
                             Org("Медико-соціальна експертна комісія", f"№ {r.randint(1, 4)} {r.choice(CITIES_GEN)}")])
        return r.choice([Org("Виконавчий комітет", inflect_phrase(_femn(r.choice(DISTRICT_ADJ)) + " міська рада", "gent")),
                         Org(f"{_femn(r.choice(DISTRICT_ADJ))} районна державна адміністрація", ""),
                         Org(f"{_femn(r.choice(DISTRICT_ADJ))} міська рада", "")])

    # --- military -------------------------------------------------------------------------------------------
    def unit(self) -> Org:
        r = self.rng
        code = f"{r.choice(['А', 'А', 'А', 'В', 'Т'])}{r.randint(1000, 9999)}"
        x = r.random()
        if x < 0.45:
            return Org(r.choice(["Військова частина", "Військова частина", "військова частина"]), code)
        if x < 0.55:
            return Org("в/ч", code, fixed=True)
        head, tail = r.choice(UNIT_TYPES)
        if r.random() < 0.3:  # numbered unit without a name
            return Org(f"{r.randint(1, 160)} {head[0].lower()}{head[1:]}", tail)
        return Org(head, f"{tail} «{r.choice(UNIT_NAMES)}»".strip())

    def tck(self) -> Org:
        r = self.rng
        d = r.choice(DISTRICT_ADJ)
        region = r.choice(REGIONS_LOC).replace("ій", "а")
        x = r.random()
        if x < 0.3:
            return Org(f"{d} районний територіальний центр", "комплектування та соціальної підтримки")
        if x < 0.45:
            return Org(f"{d} районний у місті Києві територіальний центр", "комплектування та соціальної підтримки")
        if x < 0.6:
            return Org(f"{region[:-1]}ий обласний територіальний центр", "комплектування та соціальної підтримки")
        if x < 0.75:
            return Org(f"{d} об'єднаний районний територіальний центр", "комплектування та соціальної підтримки")
        # a bare «ТЦК та СП» names no particular office and stands open in the decisions: it is a short name (SHORT_MILITARY) only
        # the adjectives decline, the abbreviation does not: «скарга Печерського РТЦК та СП», «зобов'язати Печерський РТЦК та СП»
        return Org(d, "РТЦК та СП", agree="masc") if r.random() < 0.5 else Org(f"{d} районний", "ТЦК та СП", agree="masc")

    def medical(self) -> Org:
        r = self.rng
        head, tail = r.choice(MEDICAL)
        tail = tail.format(region_gen=r.choice(REGIONS_GEN), city_gen=f"м. {r.choice(CITIES_GEN)}",
                           unit=f"{r.choice(['А', 'В'])}{r.randint(1000, 9999)}")
        return Org(head, tail)

    def military_body(self) -> Org:
        r = self.rng
        kind = r.choices(["central", "dept", "tck", "unit", "medical", "kev", "civil_mil"], [22, 12, 22, 24, 10, 4, 6])[0]
        if kind == "central":
            return Org(*r.choice(MILITARY_CENTRAL))
        if kind == "dept":
            return Org(*r.choice(MOU_DEPTS))
        if kind == "tck":
            return self.tck()
        if kind == "unit":
            return self.unit()
        if kind == "medical":
            return self.medical()
        if kind == "kev":
            return r.choice([Org("Квартирно-експлуатаційний відділ", f"м. {r.choice(CITIES_GEN)}"),
                             Org("Головне квартирно-експлуатаційне управління", "Збройних Сил України"),
                             Org(f"{r.choice(['Київське', 'Львівське', 'Одеське', 'Харківське', 'Дніпровське'])} квартирно-експлуатаційне управління", "")])
        # a civil body that handles servicemen's affairs
        if name := self._real("treasury", "social", P_REAL_SOCIAL / 2):
            self.used["body_treasury_social"] += 1
            return real_org(name)
        return r.choice([Org("Головне управління", f"Пенсійного фонду України в {r.choice(REGIONS_LOC)} області"),
                         Org("Головне управління", f"Національної поліції в {r.choice(REGIONS_LOC)} області"),
                         Org("Головне управління", "Національної поліції України"),
                         Org("Департамент", f"соціального захисту населення {r.choice(REGIONS_LOC).replace('ій', 'ої')} обласної державної адміністрації")])

    # --- companies ------------------------------------------------------------------------------------------
    def company_name(self, commercial: bool = False) -> str:
        """A bare company name, as it stands inside quotes: «Комтех Логістік», «Дельта-Пром», «Укрбудінвест».
        Real names: from court decisions in any case, from the register of legal entities in a commercial one."""
        r = self.rng
        if commercial and r.random() < P_EDR and (name := self._real("edr", "company", P_REAL_COMPANY)):
            self.used["company_edr"] += 1
            return name.upper() if r.random() < P_UPPER else name
        if name := self._real("court", "company", P_REAL_COMPANY):
            self.used["company_court"] += 1
            return name
        self.used["company_composed"] += 1
        x = r.random()
        if x < 0.35:
            return f"{r.choice(NAME_WORDS)} {r.choice(NAME_TAILS)}"
        if x < 0.5:
            return f"{r.choice(NAME_WORDS)}-{r.choice(NAME_TAILS)}"
        if x < 0.8:
            name = r.choice(COMPOUND_PREFIX).lower() + r.choice(COMPOUND_ROOT)
            if r.random() < 0.4:
                name += r.choice(COMPOUND_ROOT)
            return name[0].upper() + name[1:]
        if x < 0.9:
            return r.choice(NAME_WORDS)
        return f"{r.choice(NAME_WORDS)} {r.choice(NAME_TAILS)} {r.choice(['Україна', 'Груп', 'ЛТД'])}"

    def company(self, commercial: bool = False) -> Org:
        r = self.rng
        full, abbr, _ = r.choices(LEGAL_FORMS, [w for _, _, w in LEGAL_FORMS])[0]
        quote = r.choice(['«{}»', '«{}»', '"{}"'])
        name = quote.format(self.company_name(commercial))
        if r.random() < 0.45:
            return Org(abbr, name, fixed=True)
        words = full.split()
        # «Товариство з обмеженою відповідальністю»: only the first noun phrase declines
        head = " ".join(words[: words.index("з")]) if "з" in words else full
        tail = (" ".join(words[words.index("з"):]) + " " + name) if "з" in words else name
        return Org(head, tail)

    # --- the unit's own name ---------------------------------------------------------------------------------
    def own_unit_name(self, parent: str = "") -> Org:
        """The name of a unit whose code stands open next to the placeholder («【X】 (військова частина НОМЕР_1)»):
        a numbered brigade or regiment, a border detachment, a regional directorate, a territorial-defence or a
        training unit - never «Військова частина <code>», a ТЦК or the ministry.
        `parent`: the service the text names right after the slot («border», «guard», «army»): the unit is one of that
        service and its name ends before the parent body, which the text already has."""
        r = self.rng
        n = r.randint(1, 160)
        if parent == "border":
            x = r.random()
            if x < 0.4:
                return Org(f"{r.randint(1, 31)} прикордонний загін", r.choice(["", "", r.choice(PATRONS)]))
            if x < 0.8:
                return Org(f"{r.choice(BORDER_ADJ)} прикордонний загін", "")
            return Org(f"{r.choice(DIRECTIONS_NEUT)} регіональне управління", "")
        if parent == "guard":
            head, tail = r.choice([("бригада", "оперативного призначення"), ("полк", ""), ("окрема бригада", "охорони громадського порядку"),
                                   ("окремий батальйон", ""), ("полк", "охорони громадського порядку"),
                                   ("окремий загін", "спеціального призначення")])
            return Org(f"{n} {head}", tail if r.random() < 0.8 else f"{tail} {r.choice(PATRONS)}".strip())
        kinds, weights = ["brigade", "regiment", "border", "directorate", "tdf", "training"], [30, 12, 20, 15, 13, 10]
        if parent == "army":
            kinds, weights = ["brigade", "regiment", "tdf"], [55, 25, 20]
        kind = r.choices(kinds, weights)[0]
        if kind in ("brigade", "regiment"):
            pool = BRIGADES if kind == "brigade" else REGIMENTS
            head, tail = r.choice([u for u in pool if "України" not in u[1]] if parent else pool)
            x = r.random()
            if "України" not in tail and x < 0.2:
                tail = f"{tail} {r.choice(PATRONS)}".strip()
            elif "України" not in tail and x < 0.4 and not parent:
                tail = f"{tail} {r.choice(ARMS[:1] + ARMS[3:])}".strip()
            return Org(f"{n} {head}", tail)
        if kind == "border":
            x = r.random()
            if x < 0.45:
                return Org(f"{r.randint(1, 31)} прикордонний загін", r.choice(["", "", "Державної прикордонної служби України", r.choice(PATRONS)]))
            if x < 0.9:
                return Org(f"{r.choice(BORDER_ADJ)} прикордонний загін", r.choice(["", "", "Державної прикордонної служби України"]))
            return Org("Окремий контрольно-пропускний пункт", f"«{r.choice(['Київ', 'Одеса', 'Львів', 'Харків'])}»")
        if kind == "directorate":
            head, tail = r.choice(DIRECTORATES)
            return Org(head.format(dir=r.choice(DIRECTIONS_NEUT)), tail.format(name=r.choice(DIRECTION_NAMES)))
        if kind == "tdf":
            head = r.choice(["окрема бригада", "окрема бригада", "окремий батальйон"])
            tails = ["територіальної оборони", "територіальної оборони", "Сил територіальної оборони Збройних Сил України"]
            return Org(f"{n} {head}", r.choice(tails[:2] if parent else tails))
        head, tail = r.choice(TRAINING)
        return Org(head.format(n=r.randint(1, 250)), tail.format(arm=r.choice(ARMS)))

    # --- short names ------------------------------------------------------------------------------------------
    def short_name(self, kind: str, commercial: bool = False) -> Org:
        """What stands after «(далі - …)»: an abbreviation that never declines («ТЦК та СП», «в/ч А1234», «ГУ ПФУ в
        Одеській області», «ТОВ «Комтех»»)."""
        r = self.rng
        if kind == "company":
            _, abbr, _ = r.choices(LEGAL_FORMS, [w for _, _, w in LEGAL_FORMS])[0]
            return Org(abbr, f"«{self.company_name(commercial)}»", fixed=True)
        name = r.choice(SHORT_MILITARY if kind == "military" else SHORT_CIVIL)
        region = r.choice(REGIONS_LOC)
        return Org(name.format(unit=f"{r.choice(['А', 'А', 'А', 'В', 'Т'])}{r.randint(1000, 9999)}", city=r.choice(CITIES_GEN),
                               district=r.choice(DISTRICT_ADJ), region=_region_adj(region, "ий"), region_loc=region), fixed=True)

    def short_for(self, antecedent: str) -> Org | None:
        """The short name of the body that stands right before «(далі - …)»: «військової частини А5885» -> «в/ч А5885»,
        «… територіального центру комплектування та соціальної підтримки» -> «ТЦК та СП», «Головного управління Пенсійного
        фонду України в Одеській області» -> «ГУ ПФУ в Одеській області». None when the body has no usual short name:
        a random one would contradict the full name next to it."""
        r = self.rng
        starts = [m.end() for m in _PHRASE_START.finditer(antecedent)]
        phrase = antecedent[starts[-1]:].strip() if starts and len(antecedent) - starts[-1] >= 8 else antecedent.strip()
        if m := _ANTECEDENT_UNIT.search(phrase):
            return Org(r.choice(["в/ч", "в/ч", "ВЧ", "військова частина"]), m.group(1), fixed=True)
        if m := _ANTECEDENT_TCK.search(phrase):
            level = (m.group(1) or "").lower()
            names = ["ОТЦК та СП", "ТЦК та СП", "ТЦК"] if level.startswith("обл") else ["РТЦК та СП", "ТЦК та СП", "ТЦК та СП", "ТЦК"]
            return Org(r.choice(names if level else ["ТЦК та СП", "ТЦК та СП", "ТЦК"]), fixed=True)
        if m := _ANTECEDENT_VLK.search(phrase):
            return Org("ЦВЛК" if m.group(1) else "ВЛК", fixed=True)
        if m := _ANTECEDENT_KEV.search(phrase):
            return Org("КЕВ", m.group(1), fixed=True)
        if m := _ANTECEDENT_REGIONAL.search(phrase):
            return Org(r.choice(_REGIONAL_SHORT[m.group(1)]), " ".join(m.group(2).split()), fixed=True)
        if m := _ANTECEDENT_COMPANY.search(phrase):
            abbr = next(a for key, a in _FORM_ABBR if re.search(key, m.group(1), re.I))
            return Org(abbr, m.group(2), fixed=True)
        for rx, names in _ANTECEDENT_CENTRAL:
            if rx.match(phrase):
                return Org(r.choice(names), fixed=True)
        return None

    # --- holders of documents, places of work or custody -------------------------------------------------------
    def bank(self) -> Org:
        form, name = self.rng.choice(BANKS)
        return Org(form, f"«{name}»", fixed=True)

    def police_unit(self) -> Org:
        r = self.rng
        d, region = r.choice(DISTRICT_ADJ), r.choice(REGIONS_LOC)
        parent = r.choice([f"Головного управління Національної поліції в {region} області", f"ГУНП в {region} області"])
        return r.choice([Org(f"{_neut(d)} районне управління", f"поліції {parent}"),
                         Org("Відділ", f"поліції № {r.randint(1, 5)} {inflect_phrase(_neut(d) + ' районне управління', 'gent')} поліції {parent}"),
                         Org("Слідче управління", parent), Org("Головне управління", f"Національної поліції в {region} області"),
                         Org(f"{_neut(d)} відділення", f"поліції {parent}")])

    def hospital(self) -> Org:
        r = self.rng
        full, abbr = r.choice(MEDICAL_FORMS)
        name = f"«{self.institution_name('hospital')}»"
        if r.random() < 0.5:
            return Org(abbr, name, fixed=True)
        return Org(full, f"{name} {inflect_phrase(_femn(r.choice(DISTRICT_ADJ)) + ' міська рада', 'gent')}" if r.random() < 0.5 else name)

    def prosecutors_office(self) -> Org:
        r = self.rng
        if name := self._real("treasury", "prosecutor", P_REAL_BODY):
            return real_org(name)
        d, region = r.choice(DISTRICT_ADJ), r.choice(REGIONS_LOC)
        return r.choice([Org(f"{_region_adj(region, 'а')} обласна прокуратура", ""), Org(f"{_femn(d)} окружна прокуратура", ""),
                         Org(f"{_femn(d)} окружна прокуратура", f"{_region_adj(region, 'ої')} області"), Org("Офіс", "Генерального прокурора")])

    def guardhouse_body(self) -> Org:
        """Whose guardhouse a serviceman is held in: a unit or a body of the Military Law Enforcement Service."""
        r = self.rng
        if r.random() < 0.4:  # the unit by its code: «гауптвахти військової частини А1234»
            return Org(r.choice(["Військова частина", "військова частина"]), f"{r.choice(['А', 'А', 'А', 'В', 'Т'])}{r.randint(1000, 9999)}")
        region = r.choice(REGIONS_LOC)
        return r.choice([Org(f"{r.choice(DIRECTIONS_NEUT)} територіальне управління", "Військової служби правопорядку"),
                         Org("Зональний відділ", f"Військової служби правопорядку в {region} області"),
                         Org("Відділ", f"Військової служби правопорядку у {r.choice(CITIES_LOC)}"),
                         Org("Центральне управління", "Військової служби правопорядку (по м. Києву і Київській області)")])

    def holder_slot(self, left: str) -> str | None:
        """What a holder slot is, when the words before it tell: «guardhouse», «investigation», «draft_office»."""
        if _GUARDHOUSE_CUE.search(left[-CUE_WINDOW:]):
            return "guardhouse"
        if _INVESTIGATION_CUE.search(left[-CUE_WINDOW:]):
            return "investigation"
        return "draft_office" if _DRAFT_OFFICE_CUE.search(left) else None

    def holder(self, kind: str, registers: bool = False, commercial: bool = False, left: str = "") -> Org:
        """`left`: only where the text speaks of records («у володінні», «зберігаються в», «які належать») a bank or a
        mobile operator fits; elsewhere a company slot is a company."""
        r = self.rng
        if kind == "military":
            return self.military_body()
        if kind == "civil":
            what = r.choices(["body", "police", "hospital"], [35, 35, 30])[0]
            return self.civil_body(registers) if what == "body" else self.police_unit() if what == "police" else self.hospital()
        what = r.choices(["bank", "operator", "company"], [45, 25, 30])[0] if not left or _RECORDS_CUE.search(left[-CUE_WINDOW:]) else "company"
        if what == "bank":
            return self.bank()
        if what == "operator":
            form, name = r.choice(MOBILE_OPERATORS)
            return Org(form, f"«{name}»", fixed=True)
        return self.company(commercial)

    def official(self, left: str = "") -> Org:
        """The body of a judge, a prosecutor or a bank card («суддя 【X】», «прокурор 【X】», «картки 【X】»): the one the
        words before the slot (`left`) name, else any of the three. The slot is masked, so a court may stand here:
        nothing is taught from it."""
        r = self.rng
        kind = r.choices([k for k, _ in OFFICIAL_MIX], [w for _, w in OFFICIAL_MIX])[0]
        kind = next((k for k, rx in _OFFICIAL_CUES if rx.search(left[-CUE_WINDOW:])), kind)
        if kind == "bank":
            return self.bank()
        d, region = r.choice(DISTRICT_ADJ), r.choice(REGIONS_LOC)
        if kind == "court":
            return r.choice([Org(f"{d} районний суд", f"м. {r.choice(CITIES_GEN)}"), Org(f"{d} районний суд", f"{_region_adj(region, 'ої')} області"),
                             Org(f"{_region_adj(region, 'ий')} окружний адміністративний суд", ""),
                             Org(f"{_region_adj(region, 'ий')} апеляційний суд", ""), Org(f"{d} міськрайонний суд", f"{_region_adj(region, 'ої')} області")])
        return self.prosecutors_office()

    # --- names inside the quotes of the text -------------------------------------------------------------------
    def institution_name(self, what: str | None = None, left: str = "") -> str:
        """A bare name of a hospital, a school or a penal institution, as it stands in quotes after «КНП», «КЗ», «ДУ».
        `left`: the words before the opening quote, which say which of the three it is."""
        r = self.rng
        mix = next((m for rx, m in _INSTITUTION_CUES if rx.search(left[-CUE_WINDOW:])), INSTITUTION_MIX)
        what = what or r.choices([k for k, _ in mix], [w for _, w in mix])[0]
        d, region, city = _femn(r.choice(DISTRICT_ADJ)), r.choice(REGIONS_LOC), r.choice(CITIES_GEN)
        if what == "hospital":
            return r.choice(["Центральна районна лікарня", f"{d} центральна районна лікарня", f"Міська клінічна лікарня № {r.randint(1, 18)}",
                             f"Міська лікарня № {r.randint(1, 12)}", f"{_region_adj(region, 'а')} обласна клінічна лікарня",
                             f"{d} багатопрофільна лікарня інтенсивного лікування", f"Центр первинної медико-санітарної допомоги № {r.randint(1, 9)}",
                             f"{_region_adj(region, 'а')} обласна психіатрична лікарня", f"Міська поліклініка № {r.randint(1, 20)}",
                             f"Клінічна лікарня швидкої медичної допомоги м. {city}", f"{d} міська лікарня",
                             f"{_region_adj(region, 'ий')} обласний наркологічний диспансер"])
        if what == "school":
            return r.choice([f"Ліцей № {r.randint(1, 120)}", f"Гімназія № {r.randint(1, 90)}", f"{d[:-1]}ий ліцей", f"Ліцей № {r.randint(1, 60)} м. {city}",
                             f"Заклад дошкільної освіти № {r.randint(1, 300)}", f"Середня загальноосвітня школа № {r.randint(1, 150)}",
                             f"{d[:-1]}ий професійний ліцей", f"Спеціалізована школа № {r.randint(1, 200)}"])
        return r.choice([f"{d} виправна колонія (№ {r.randint(1, 140)})", f"{d} виправна колонія (№ {r.randint(1, 140)})",
                         f"{d} установа виконання покарань (№ {r.randint(1, 40)})", f"{d[:-1]}ий слідчий ізолятор",
                         f"{d[:-1]}ий виправний центр (№ {r.randint(1, 140)})"])

    def quoted_name(self, quote_kind: str = "company", commercial: bool = False, seen: Seen | None = None, left: str = "") -> str:
        """The bare name for a placeholder inside the quotes of the text, by what stands before the quote: a company
        after a legal form, a bank after a bank cue, a direction or a place after a military formation, a hospital /
        school / colony after «КНП» / «КЗ» / «ДУ» (`left`, the text before the quote, tells which). An unknown kind
        gets a company name (the caller masks it)."""
        r = self.rng
        # the rules give the kind of the placeholder's last mention; the first one, which the value is drawn at, may be
        # the one that names the bank («на банківську картку АТ «»)
        if quote_kind == "bank" or (quote_kind == "company" and bank_in_quotes(left)):
            quote_kind = "bank"

            def draw() -> str:
                name = r.choice(BANKS)[1]
                return name.upper() if r.random() < P_UPPER else name
        elif quote_kind == "formation":
            # the cue closest to the quote decides; without one, either list
            cues = [(m.end(), names) for rx, names in ((_BORDER_CUE, FORMATION_PLACES), (_COMMAND_CUE, FORMATION_DIRECTIONS),
                                                       (_OPERATIONAL_COMMAND, OPERATIONAL_COMMANDS), (_AIR_COMMAND, AIR_COMMANDS))
                    for m in rx.finditer(left[-CUE_WINDOW:])]
            # «оперативного командування «»: both the general cue and the exact one end there, the shorter list is the exact one
            names = min((c for c in cues if c[0] == max(cues)[0]), key=lambda c: len(c[1]))[1] if cues else FORMATION_DIRECTIONS + FORMATION_PLACES

            def draw() -> str:
                # a short list (the four operational commands): take a name the decision does not have while there is one
                free = [n for n in names if not seen.has(n)] if seen is not None and len(names) <= 2 * MAX_REDRAWS else names
                return r.choice(free or names)
        elif quote_kind == "institution":
            def draw() -> str:
                return self.institution_name(left=left)
        elif quote_kind == "company" and _OPERATOR_CUE.search(left[-90:]):
            # a mobile operator: one whose legal form is the one before the quote, when the text has it
            form = _FORM_BEFORE_QUOTE.search(left)
            names = [name for f, name in MOBILE_OPERATORS if not form or f == form.group(1)] or [name for _, name in MOBILE_OPERATORS]

            def draw() -> str:
                return r.choice(names)
        else:
            def draw() -> str:
                return self.company_name(commercial)
        self.drawn[f"quoted_{quote_kind if quote_kind in ('bank', 'formation', 'institution') else 'company'}"] += 1
        return self.fresh(draw, seen)

    # --- entry point ----------------------------------------------------------------------------------------
    def _draw(self, org_kind: str, military: bool, commercial: bool, social: bool, left: str = "", right: str = "") -> tuple[Org, str]:
        """A body and the kind it is counted under («military», «civil», «company», «official»)."""
        r = self.rng
        if org_kind == "unit":  # military whatever the decision is about: the code of the unit stands next to the slot
            parent = _PARENT_RIGHT.search(right)
            return self.own_unit_name(PARENTS[parent.group(1).lower()] if parent else ""), "military"
        if org_kind == "official":
            return self.official(left), "official"
        slot = self.holder_slot(left) if org_kind == "holder" else None
        if slot == "guardhouse":
            return self.guardhouse_body(), "military"
        if slot == "investigation":
            return (self.police_unit() if r.random() < 0.7 else self.prosecutors_office()), "civil"
        if slot == "draft_office":
            return self.tck(), "military"
        about = "military" if military else "commercial" if commercial else "other"
        kind = r.choices(KINDS, KIND_MIX["holder" if org_kind == "holder" else "party"][about])[0]
        if org_kind == "short":
            return self.short_name(kind, commercial), kind
        if org_kind == "holder":
            return self.holder(kind, registers=military or social, commercial=commercial, left=left), kind
        if kind == "military":
            return self.military_body(), kind
        return (self.civil_body(registers=military or social) if kind == "civil" else self.company(commercial)), kind

    def institution(self, military: bool = False, commercial: bool = False, social: bool = False, org_kind: str = "party",
                    seen: Seen | None = None, left: str = "", right: str = "", count_as: str | None = None) -> Org:
        """A body that a decision anonymised as ІНФОРМАЦІЯ, chosen by the kind of slot (`org_kind` of the context
        rules; an unknown one counts as a party) and by what the decision is about. `seen`: the names the decision
        already has - a body that stands there is drawn again. `left`, `right`: the text around the slot, which tells
        an «official» slot whether a court, a prosecutor's office or a bank stands in it, a «holder» slot whose
        guardhouse or investigation it is, and a «unit» slot which service the unit belongs to. `count_as`: the name the
        draw is counted under in `drawn` when it is not the kind of slot («other» for a placeholder no rule typed)."""
        if org_kind not in ORG_KINDS:
            org_kind = "party"
        kinds: list[str] = []

        def draw() -> Org:
            org, kind = self._draw(org_kind, military, commercial, social, left, right)
            kinds.append(kind)
            return org
        org = self.fresh(draw, seen)
        # only the body that is kept counts: a redraw is counted once, under «redraw»
        self.drawn["official" if org_kind == "official" else f"{count_as or org_kind}_{kinds[-1]}"] += 1
        return org

    def unit_name(self, seen: Seen | None = None) -> Org:
        """For «у військовій частині ІНФОРМАЦІЯ_1»: what follows the words the text already has - the unit's code,
        its name in quotes, or the formation it belongs to."""
        r = self.rng

        def draw() -> Org:
            x = r.random()
            if x < 0.5:
                return Org(f"{r.choice(['А', 'А', 'А', 'В', 'Т'])}{r.randint(1000, 9999)}", fixed=True)
            if x < 0.8:
                return Org(f"«{r.choice(UNIT_NAMES)}»", fixed=True)
            return Org(*r.choice([("Національної гвардії України", ""), ("Державної прикордонної служби України", ""),
                                  ("Сухопутних військ Збройних Сил України", ""), ("Сил територіальної оборони Збройних Сил України", "")]), fixed=True)
        return self.fresh(draw, seen)
