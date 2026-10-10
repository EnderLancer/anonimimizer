"""Entity linking for pseudonyms: every mention of one entity inside a court decision gets the same pseudonym.

    assign(text, spans) -> ["ОСОБА_1", "ОСОБА_1", "АДРЕСА_1", "ОСОБА_2", ...]

The module uses the standard library only and imports nothing from the package, so it can be loaded by its path.

Persons (the type the official Consistency metric, B-cubed F1, is computed on) are linked by matching mentions word by
word, without ever deciding which word is the surname:

1. A mention is parsed into capitalised words and one-letter initials («Коваля І. П.» -> word «коваля», initials І, П).
   Role and rank words that a span may swallow («Суддя», «Обвинувачений», «Свідок») are dropped. A placeholder left in
   the text («ОСОБА_1», «Особа2», «ОСОБА 1») is not parsed: it is linked by its word and number.
2. Every word gets a set of readings (stem, paradigm, kind of form): the word minus one ending of a closed set. The
   paradigms are the consonant / -о declension (Коваль - Коваля - Ковалеві - Ковалем), the -а declension (Сорока -
   Сороці), and the masculine and feminine adjectival declensions (Ковальський - Ковальського; Ковальська -
   Ковальської). Stems are made comparable across the usual alternations: fleeting vowels (Кравець - Кравця), і / о /
   е and ї / є in a closed syllable (Федір - Федора, Київ - Києва), г / к / х before -і (Ольга - Ользі), soft signs,
   the doubled consonant of «Нінеллю». Two words are the same name when they share a reading; because the adjectival
   paradigms are split by gender, «Ковальського» and «Ковальської» do not match, and «Зелена» is no form of the man
   «Зеленим» (the ending -им belongs to the consonant declension only after -ов / -ів / -ин).
3. The gender of a mention comes from the patronymic (also «Кузьмич», «Лукич»; «Головні» is a surname) and from the
   case endings of surnames in -ський / -цький / -зький. Mentions of different gender are different persons.
4. The words of one mention agree in case and gender, and this removes readings. Every reading knows the cases and
   the genders it may stand for; a name in a consonant or -о that is not declined is a nominative or a woman.
   - With a patronymic the case and the gender are known: «Олександра» in «Гнатенка Олександра Сидоровича» is not a
     woman's name, «Лобода» in «Лобода Богдан Юхимович» is not the genitive of «Лобод», «Ярославі» in «Мазур
     Ярославі Степанівні» is «Ярослава».
   - A surname and a first name without a patronymic are tried in every case and gender, and each way that agrees is
     a reading of the mention, with its gender when only one fits: «Заєць Валентина» is the woman (his genitive is
     «Зайця Валентина»), «Романа Притули» and «Гмирі Валентина» are men in the genitive, «Лобода Богдана» is the
     woman or the genitive of «Лобод Богдан» but never «Лобода Богдан», «Лободі Богдану» is his dative.
   - Read word by word, one rule is left: next to a man's patronymic in the genitive or the dative, a possessive
     surname with the ending of that case is in that case. «Кузнецова Гліба Аркадійовича» is a man called
     «Кузнецов», so the feminine forms «Кузнецовою» and «Кузнецовій» are never his.
5. Two mentions are compared by pairing their words. What is left over must not contradict: left-over words and
   initials, read in text order as first letters, must be a prefix of one another («Коваль І.» fits «Коваль Іван
   Петрович», «Коваль П. І.» does not), and two mentions that each keep a word the other lacks are different persons.
   A surname in a consonant or -о that is declined belongs to a man, so «Бондаря М. С.» is not «Бондар Марія
   Степанівна».
6. Mentions are clustered greedily, the most informative first and then in text order. A mention joins an entity when
   it fits the richest mention of that entity and contradicts none of its other mentions. If several entities fit,
   the one with the most specific match wins, then the one mentioned most recently before the mention (or, failing
   that, soonest after it). So a bare surname shared by a father and a son goes to whoever was named last, and never
   starts a third person. Three readings are tried:
   - with agreement (step 4) and word by word, and the better match of the two wins; agreement decides only between
     persons that fit equally well. So «Шевченко Олега» still finds «Шевченко Олег Петрович» (clerks do leave a
     man's surname undeclined), and «Седих Нодара» goes to «Седих Нодар Зурабович», not to «Седих Н. В.»;
   - with no rule at all, before the mention starts a new person, and only to join a person whose whole name it
     repeats: «Щербина Устима Гордійовича» after «Щербини Устима Гордійовича».

The other types are linked by normalised keys:
- НОМЕР: without punctuation except «/» (the fields of a case number) and with Cyrillic / Latin look-alike letters
  unified; phone numbers by their last nine digits («+380 50 123 45 67» = «8 050 123 45 67» = «тел. 050 123-45-67»).
- ІНФОРМАЦІЯ: dates as (year, month, day); e-mails in lower case; accounts and pages by their exact spelling;
  institution names word by word with the stemming above, after the abbreviations courts use are spelled out («в/ч»,
  «в.ч.», «ВЧ», «ТЦК та СП», «МОУ», «ДП», «ТОВ» ...), a code written apart is glued («в/ч А 1234» = «Військова
  частина А1234») and an ordinal loses its ending («110-та бригада» = «110 бригада»).
- АДРЕСА: by the parsed parts (settlement, street, house, building, apartment), where the postal index, the country,
  the region and the district are optional and a different house or apartment is a different address. A street named
  after a person is known by the surname («вул. Мазепи» = «вул. Івана Мазепи» = «вул. І. Мазепи»). An address
  without a settlement joins the matching address mentioned most recently before it, as a bare surname does.

`assign` never raises (an odd span gives an empty mention), keeps no state between calls except caches of pure
functions, and depends only on its arguments; the entities do not depend on the order of the spans, because mentions
are linked in the order of their offsets. Persons are found through an index of the stems of their names, so 3000
different full names in one call take half a second; ІНФОРМАЦІЯ is compared pair by pair, and 3000 different
institutions in one call take several seconds (a decision has a few).

Limits
- There is no dictionary of names. Two spellings of one first name are two names: after «Мазур Наталія Петрівна» and
  «Наталя Мазур», the regular genitive «Мазур Наталії Петрівни» joins the first and «Мазур Наталі Петрівни» the
  second. A man's surname in -а cannot be told from the feminine of a possessive one, so «Щербиною У. Г.» does not
  join «Щербина Устима Гордійовича» when that slip is the only full mention of him.
- A bare name with patronymic is linked to a full name only: «Гребенюк Л. Т.», «Лілія Гребенюк», «Лілії Тимурівни»
  gives two persons, and it never joins a person known by initials alone («Сокіл О. О.»).
- Forms that are ambiguous in the language stay ambiguous and go to the most recent candidate: «Кузнецова Г. А.» (her
  nominative, his genitive), «Ярославі Яремі» (her dative, his locative), «Валерій Мазурок» next to «Валерія
  Мазурок» (his nominative, the dative of an adjectival «Валерія»), a woman named only by two words next to a man
  named only by two words.
- A mention the source itself gets wrong is a new person: «Войцехівською Радимом Борисовичем» (a man with a feminine
  surname form; the one split on the silver dev set). A span cut inside a word («Тимченка Зінов») is not repaired.
- A role word outside the closed list that a span swallows («Понятий Довженко Х. П.») is read as a name word, and so
  is a rank that is also a surname («Майор», «Капітан», «Голова»).
- Two-digit years use a fixed pivot (31-99 -> 19xx, 00-30 -> 20xx): «05.09.87» is «05.09.1987».
- An abbreviation has one expansion; «ДП» of a «дочірнє підприємство» stays apart from its full name.
- Identifiers that differ only in hyphens, spaces or dots are one («2-1234» = «21-234», «125» = «1,25»).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

PERSON, ADDRESS, NUMBER, INFO = "ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ"

# --------------------------------------------------------------------------------------------------------------------
# Letters
# --------------------------------------------------------------------------------------------------------------------

_LATIN_TO_CYRILLIC = str.maketrans("aceiopxyABCEHIKMOPTXY", "асеіорхуАВСЕНІКМОРТХУ")
_CYRILLIC_TO_LATIN = str.maketrans("АВЕІКМНОРСТХУ", "ABEIKMHOPCTXY")
_APOSTROPHES = "'’ʼ`‘ʹ′"
_APOSTROPHE_RE = re.compile(f"[{_APOSTROPHES}]")
_CYRILLIC_RE = re.compile("[а-яґєіїА-ЯҐЄІЇ]")
_VOWELS = frozenset("аеиіоуяюєї")
_WORD_RE = re.compile(rf"[^\W\d_]+(?:[{_APOSTROPHES}][^\W\d_]+)*")


def _fix_script(word: str) -> str:
    """Replace Latin look-alikes typed inside a Cyrillic word (OCR and keyboard slips: «Iван», «Kоваль»)."""
    if len(word) == 1 or _CYRILLIC_RE.search(word):
        return word.translate(_LATIN_TO_CYRILLIC)
    return word


# --------------------------------------------------------------------------------------------------------------------
# Word morphology: analyses (stem, paradigm, kind)
# --------------------------------------------------------------------------------------------------------------------

_N2, _N1, _ADJ_M_HARD, _ADJ_M_SOFT, _ADJ_F = "n2", "n1", "adj-mh", "adj-ms", "adj-f"
_NOMINATIVE, _MASCULINE_OBLIQUE, _OBLIQUE = 0, 1, 2

# Consonant and -о declension (Коваль, Андрій, Петро, Шевченко). -им is the instrumental of surnames in -ов / -ів / -ин and
# of nothing else: «Зеленим» is «Зелений», so that «Зелена» is not taken for the genitive of a man called «Зелен».
_N2_NOMINATIVE = ("", "ь", "й", "о")
# -ї is the locative of names in -й (Валерій - Валерії); no woman's name of this declension ends in -й.
_N2_MASCULINE_OBLIQUE = ("а", "я", "у", "ю", "ові", "еві", "єві", "ом", "ем", "єм", "им", "ї")
# -і is kept apart: it is the locative of a man's name (Олександр - Олександрі) but also the oblique form of women's
# names in a consonant (Любов - Любові, Нінель - Нінелі).
_N2_OBLIQUE = ("і",)
_N1_NOMINATIVE = ("а", "я")
_N1_OBLIQUE = ("и", "і", "ї", "у", "ю", "ою", "ею", "єю")
_ADJ_M_HARD_NOMINATIVE = ("ий", "ой")
_ADJ_M_HARD_OBLIQUE = ("ого", "ому", "им", "ім")
_ADJ_M_SOFT_NOMINATIVE = ("ій",)
_ADJ_M_SOFT_OBLIQUE = ("ього", "ьому", "ім", "їм")
_ADJ_F_NOMINATIVE = ("а", "я")
_ADJ_F_OBLIQUE = ("ої", "ій", "у", "ю", "ою", "ьої", "ьою")
_MIN_STEM = 2
_POSSESSIVE_STEM_RE = re.compile(r"(?:[оеєії]в|[иії]н)$")
# Before the ending -і of the -а declension the stem consonant softens: Ольга - Ользі, Сорока - Сороці, Муха - Мусі.
_VELAR_SOURCE = {"з": "г", "ц": "к", "с": "х"}


def _closed_syllable_variants(stem: str) -> list[str]:
    """Stems an endingless form may have before a vowel ending: Кравець -> кравц, Заєць -> зайц, Федір -> федор,
    Київ -> києв."""
    out: list[str] = []
    if len(stem) < 3 or stem[-1] in _VOWELS:
        return out
    vowel, before = stem[-2], stem[-3]
    if vowel in "ео" and before not in _VOWELS:
        out.append(stem[:-2] + stem[-1])
    elif vowel == "є" and before in _VOWELS:
        out.append(stem[:-2] + "й" + stem[-1])
    # The last vowel may be followed by more than one consonant (Чорновіл, but also Поділ(ь)ськ-like stems).
    for i in range(len(stem) - 1, -1, -1):
        if stem[i] in _VOWELS:
            if stem[i] == "і" and i > 0:
                out.extend(stem[:i] + v + stem[i + 1:] for v in "ое")
            elif stem[i] == "ї" and i > 0:
                out.append(stem[:i] + "є" + stem[i + 1:])
            break
    return out


# What a reading may stand for: codes of a case (Nominative, Genitive, Dative, Accusative, Instrumental, Locative) and
# a gender, «Gm» = a man in the genitive. Only the agreement between the words of one mention asks for it.
_ENDING_CASES: dict[tuple[str, str], str] = {
    **{(_N2, e): "GA" for e in ("а", "я")},
    **{(_N2, e): "DL" for e in ("у", "ю", "ові", "еві", "єві")},
    **{(_N2, e): "I" for e in ("ом", "ем", "єм", "им")},
    **{(_N2, e): "L" for e in ("ї", "і")},
    **{(_N1, e): "N" for e in _N1_NOMINATIVE},
    (_N1, "и"): "G", (_N1, "і"): "GDL", (_N1, "ї"): "GDL", (_N1, "у"): "A", (_N1, "ю"): "A",
    **{(_N1, e): "I" for e in ("ою", "ею", "єю")},
    **{(_ADJ_M_HARD, e): "N" for e in _ADJ_M_HARD_NOMINATIVE},
    (_ADJ_M_HARD, "ого"): "GA", (_ADJ_M_HARD, "ому"): "DL", (_ADJ_M_HARD, "им"): "I", (_ADJ_M_HARD, "ім"): "L",
    (_ADJ_M_SOFT, "ій"): "N", (_ADJ_M_SOFT, "ього"): "GA", (_ADJ_M_SOFT, "ьому"): "DL",
    (_ADJ_M_SOFT, "ім"): "IL", (_ADJ_M_SOFT, "їм"): "IL",
    **{(_ADJ_F, e): "N" for e in _ADJ_F_NOMINATIVE},
    (_ADJ_F, "ої"): "G", (_ADJ_F, "ьої"): "G", (_ADJ_F, "ій"): "DL", (_ADJ_F, "у"): "A", (_ADJ_F, "ю"): "A",
    (_ADJ_F, "ою"): "I", (_ADJ_F, "ьою"): "I",
}
# Names of the -а declension belong to women and to men (Сорока, Микола).
_PARADIGM_GENDERS = {_N2: "m", _N1: "fm", _ADJ_M_HARD: "m", _ADJ_M_SOFT: "m", _ADJ_F: "f"}
# A name in a consonant, -й, -ь or -о that is not declined: a nominative, or a woman in any case («Заєць Валентини»).
_UNDECLINED = frozenset({"Nm", "Nf", "Gf", "Df", "Af", "If", "Lf"})


def _stands_for(word: str, paradigm: str, kind: int, ending: str) -> frozenset[str]:
    if paradigm == _N2 and kind == _NOMINATIVE:
        # A word in another vowel read as it stands («Мунтяну», «Церетелі») is not declined at all: no agreement.
        return _UNDECLINED if ending or word[-1] not in _VOWELS or word[-1] == "о" else frozenset()
    codes = {case + gender for case in _ENDING_CASES[paradigm, ending] for gender in _PARADIGM_GENDERS[paradigm]}
    if paradigm == _N2 and kind == _OBLIQUE and ending == "ю":
        codes.add("If")  # Любов'ю, Нінеллю
    return frozenset(codes)


@lru_cache(maxsize=65536)
def _analysed(word: str) -> dict[tuple[str, str, int], frozenset[str]]:
    """All (stem, paradigm, kind) readings of a lower-case word, each with the cases and genders it may stand for.
    Soft signs are dropped from stems (Стрілець - Стрільця)."""
    out: dict[tuple[str, str, int], frozenset[str]] = {}

    def add(raw: str, paradigm: str, kind: int, ending: str = "") -> None:
        stem = raw.replace("ь", "")
        if len(stem) < _MIN_STEM:
            return
        stems = [stem]
        if kind == _MASCULINE_OBLIQUE and ending in ("ю", "я") and (stem[-1] in "бпвмф" or stem[-2:-1] == stem[-1]):
            # Not a masculine form: after a labial or a doubled consonant -ю is the feminine instrumental
            # (Любов'ю, Нінеллю), and the bearer may be a woman.
            kind = _OBLIQUE
            if paradigm == _N2 and ending == "ю" and stem[-2:-1] == stem[-1] and len(stem) > _MIN_STEM:
                # The consonant is doubled only in this form: Нінель - Нінеллю, Рахіль - Рахіллю.
                stems.append(stem[:-1])
        if paradigm == _N2 and ending in ("", "ь"):
            stems.extend(v for v in _closed_syllable_variants(stem) if len(v) >= _MIN_STEM)
        if paradigm == _N1 and ending == "і" and stem[-1] in _VELAR_SOURCE:
            stems.append(stem[:-1] + _VELAR_SOURCE[stem[-1]])
        stands_for = _stands_for(word, paradigm, kind, ending)
        for one in stems:
            out[one, paradigm, kind] = out.get((one, paradigm, kind), frozenset()) | stands_for

    def strip(endings: tuple[str, ...], paradigm: str, kind: int) -> None:
        for ending in endings:
            if not ending:
                add(word, paradigm, kind, ending)
            elif word.endswith(ending) and len(word) - len(ending) >= _MIN_STEM:
                raw = word[: len(word) - len(ending)]
                # «Крайнього» is soft; reading it as hard «крайнь» + «ого» would let it match a hard «Крайний».
                if paradigm == _ADJ_M_HARD and raw.endswith("ь"):
                    continue
                if paradigm == _N2 and ending == "им" and not _POSSESSIVE_STEM_RE.search(raw):
                    continue
                add(raw, paradigm, kind, ending)

    strip(_N2_NOMINATIVE, _N2, _NOMINATIVE)
    strip(_N2_MASCULINE_OBLIQUE, _N2, _MASCULINE_OBLIQUE)
    strip(_N2_OBLIQUE, _N2, _OBLIQUE)
    strip(_N1_NOMINATIVE, _N1, _NOMINATIVE)
    strip(_N1_OBLIQUE, _N1, _OBLIQUE)
    strip(_ADJ_M_HARD_NOMINATIVE, _ADJ_M_HARD, _NOMINATIVE)
    strip(_ADJ_M_HARD_OBLIQUE, _ADJ_M_HARD, _OBLIQUE)
    strip(_ADJ_M_SOFT_NOMINATIVE, _ADJ_M_SOFT, _NOMINATIVE)
    strip(_ADJ_M_SOFT_OBLIQUE, _ADJ_M_SOFT, _OBLIQUE)
    strip(_ADJ_F_NOMINATIVE, _ADJ_F, _NOMINATIVE)
    strip(_ADJ_F_OBLIQUE, _ADJ_F, _OBLIQUE)
    return out


@lru_cache(maxsize=65536)
def _analyses(word: str) -> frozenset[tuple[str, str, int]]:
    return frozenset(_analysed(word))


_NO_MATCH, _SAME, _SAME_MASCULINE = 0, 1, 2
_ADJECTIVAL_FORM_RE = re.compile(r"[сцз]ьк(?:ий|ого|ому|им|ім|ої|ій|ою)$")


_Readings = frozenset  # of (stem, paradigm, kind): the analyses a word may have inside its mention


@lru_cache(maxsize=262144)
def _match_words(a: str, b: str, xa: _Readings, xb: _Readings) -> int:
    """Are two lower-case words forms of one name? _SAME_MASCULINE: yes, and the declension shows the bearer is a man.

    `xa` and `xb` are the readings of the words in their mentions: `_analyses` of the word, or the part of them that
    the rest of the mention allows."""
    if a == b:
        return _SAME if xa & xb else _NO_MATCH
    kinds_a: dict[tuple[str, str], set[int]] = {}
    for stem, paradigm, kind in xa:
        kinds_a.setdefault((stem, paradigm), set()).add(kind)
    # «Яворським» and «Яворської» can only be adjectival forms, so a noun reading of the other word («Яворська» as
    # the genitive of a surname «Яворськ») must not pair with them.
    adjectival_only = bool(_ADJECTIVAL_FORM_RE.search(a) or _ADJECTIVAL_FORM_RE.search(b))
    shared = False
    feminine_reading = False
    masculine_reading = False
    for stem, paradigm, kind in xb:
        ka = kinds_a.get((stem, paradigm))
        if ka is None or (adjectival_only and paradigm in (_N1, _N2)):
            continue
        shared = True
        if paradigm in (_N1, _ADJ_F):
            feminine_reading = True
        elif paradigm == _N2:
            # Two different forms, one with a masculine case ending: women's surnames and names of this declension
            # do not change (Шевченко Олени, but Шевченка Олега; Смирновим is never Смирнова Глафіра).
            if kind == _MASCULINE_OBLIQUE or _MASCULINE_OBLIQUE in ka:
                masculine_reading = True
        else:
            masculine_reading = True
    if not shared:
        return _NO_MATCH
    return _SAME_MASCULINE if masculine_reading and not feminine_reading else _SAME


# --------------------------------------------------------------------------------------------------------------------
# Person mentions
# --------------------------------------------------------------------------------------------------------------------

# Words a model span may swallow together with the name; they are not part of the name. Role and rank words that are
# also usual surnames (Майор, Капітан, Солдат, Голова) are left out: there the surname is the likelier reading.
_TITLE_NOUNS = (
    "секретар прокурор адвокат пан громадянин позивач відповідач свідок представник захисник заявник скаржник "
    "боржник стягувач нотаріус експерт інспектор начальник командир директор лейтенант сержант полковник "
    "підполковник"
).split()
_TITLE_ADJECTIVES = "головуюч обвинувачен потерпіл підсудн засуджен слідч старш молодш рядов".split()
_TITLES = frozenset(
    "суддя судді суддю суддею пані громадянка громадянки громадянку громадянці громадянкою позивачка позивачки "
    "позивачку позивачці позивачкою відповідачка відповідачки відповідачку відповідачці відповідачкою свідка свідку "
    "свідком фоп гр".split()
    + [noun + ending for noun in _TITLE_NOUNS for ending in ("", "а", "у", "ові", "еві", "ом", "ем")]
    + [stem + ending for stem in _TITLE_ADJECTIVES for ending in ("ий", "ого", "ому", "им", "а", "ої", "ій", "у", "ою")]
)
# A span that covers two neighbouring names is read as the first of them.
_NAME_SEPARATOR_RE = re.compile(r"\s*(?:[,;/\n]|\s(?:та|і|й|або)\s)\s*(?=[^\W\d_])")
# A patronymic has a name before its suffix (Ян-івна, Льв-івна); «Головні» and «Жовні» are forms of the surnames
# «Головня» and «Жовна».
_FEMININE_PATRONYMIC_RE = re.compile(r"^(?!головн)..+[іїо]вн(а|и|і|у|ою|о)$")
# -ович / -евич / -йович, and the few patronymics without that suffix (Ілліч, Кузьмич, Лукич ...).
_MASCULINE_PATRONYMIC_RE = re.compile(r"(?:[оеє]вич|ьич|ліч|кузьмич|лукич|хомич|савич|фомич|микитич)(а|у|ем|еві|і|ю|)$")
# Surnames in -ський / -цький / -зький show the gender in most case forms. -ська and -ську are left out: they are
# also the genitive and the dative of men's surnames in -сько (Місько - Міська).
_FEMININE_SURNAME_RE = re.compile(r"[сцз]ьк(?:ої|ій|ою)$")
_MASCULINE_SURNAME_RE = re.compile(r"[сцз]ьк(?:ий|ого|ому|им|ім)$")
# Possessive surnames (Кузнецов - Кузнецова, Нікітін - Нікітіна) decline as adjectives in the feminine, and their
# feminine nominative and accusative look like his genitive and dative. No gender is read from their endings:
# «Дібровою» is also a man called Діброва and «Зіновій» is a first name. The feminine forms are kept away from a man
# by `_loose_readings` and by the split of the paradigms instead.
_POSSESSIVE_AMBIGUOUS_RE = re.compile(r"(?:[оеє]в|[иії]н)[ау]$")


@dataclass(frozen=True)
class _Mention:
    """A parsed person mention. `items` keeps text order: (word, True) for a name word, (letter, False) for an initial."""

    items: tuple[tuple[str, bool], ...]
    gender: str  # "f", "m" or "" when the mention does not tell
    readings: tuple[_Readings, ...] = ()  # one per word, in the order of `words`

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(v for v, is_word in self.items if is_word)

    @property
    def richness(self) -> tuple[int, int]:
        n_words = sum(1 for _, is_word in self.items if is_word)
        return len(self.items), n_words


def _mention_gender(words: tuple[str, ...], names: int) -> str:
    """`names` counts the two halves of a hyphenated name as one."""
    # The patronymic is the firmest sign, so it is asked first.
    if any(_FEMININE_PATRONYMIC_RE.search(w) for w in words):
        return "f"
    # A word in -ович alone may be a surname (Калинович Оксана, Адамович Єва-Марія); with three names one of them is
    # the patronymic.
    if names >= 3 and any(_MASCULINE_PATRONYMIC_RE.search(w) for w in words):
        return "m"
    if any(_FEMININE_SURNAME_RE.search(w) for w in words):
        return "f"
    if any(_MASCULINE_SURNAME_RE.search(w) for w in words):
        return "m"
    return ""


def _is_masculine_oblique(reading: tuple[str, str, int]) -> bool:
    _, paradigm, kind = reading
    return kind == _MASCULINE_OBLIQUE or (paradigm in (_ADJ_M_HARD, _ADJ_M_SOFT) and kind == _OBLIQUE)


def _loose_readings(words: tuple[str, ...], gender: str) -> tuple[_Readings, ...]:
    """The analyses of every word, read word by word. One rule holds even here: «Кузнецова Гліба Аркадійовича», next
    to a man's patronymic in the genitive or the dative a possessive surname with the ending of that case stands in
    that case. So his surname is «Кузнецов», and the feminine forms «Кузнецовою», «Кузнецовій» are not his. For other
    surnames the other reading may be a clerk's slip («позов Сорока Гліба Аркадійовича»); for these it is another
    person."""
    out = [_analyses(w) for w in words]
    if gender == "m":
        endings = {m[1] for m in map(_MASCULINE_PATRONYMIC_RE.search, words) if m}
        last = {"а": "ая", "у": "ую", "ю": "ую"}
        allowed = "".join(last.get(e, "") for e in endings)
        for k, word in enumerate(words):
            if word[-1] in allowed and _POSSESSIVE_AMBIGUOUS_RE.search(word):
                # Adjectival surnames keep their own oblique reading: «Садовому» is «Садовий», not «Садовом» + у.
                out[k] = frozenset(r for r in out[k] if _is_masculine_oblique(r)) or out[k]
    return tuple(out)


_MASCULINE_PATRONYMIC_CASES = {"": "N", "а": "GA", "у": "DL", "еві": "DL", "ю": "DL", "ем": "I", "і": "L"}
_FEMININE_PATRONYMIC_CASES = {"а": "N", "о": "N", "и": "G", "і": "DL", "у": "A", "ою": "I"}


def _strict_readings(words: tuple[str, ...], gender: str) -> list[tuple[str, tuple[_Readings, ...]]]:
    """The ways to read a mention whose words agree in case and gender: [(gender, the readings of every word)].

    - With a patronymic the case and the gender are known, and every other word keeps the readings that fit them:
      «Олександра» in «Гнатенка Олександра Сидоровича» is no woman's name, «Лобода» in «Лобода Богдан Юхимович» is
      no genitive of «Лобод», «Ярославі» in «Мазур Ярославі Степанівні» is «Ярослава». A word that has no such
      reading keeps them all: it is not declined («Седих») or the clerk left it so.
    - A surname and a first name without a patronymic are tried in every case and gender. «Заєць Валентина» is a
      nominative or a woman (a man's surname would be declined: «Зайця Валентина»), «Романа Притули» is the genitive
      of «Роман» (hers is «Романи Притули»), «Лобода Богдана» is the woman «Богдана» or the genitive of «Лобод
      Богдан», but never «Лобода Богдан». Each way is a reading of its own, with its gender when only one fits.
    - Anything else, and a pair that agrees in no way («Шевченко Олега»), is left as it is read word by word."""
    supports = [_analysed(w) for w in words]
    plain = tuple(frozenset(x) for x in supports)

    def fitting(code: str) -> tuple[_Readings, ...]:
        return tuple(frozenset(r for r, codes in x.items() if code in codes) for x in supports)

    cases = ""
    if gender == "f":
        cases = "".join(_FEMININE_PATRONYMIC_CASES[m[1]] for m in map(_FEMININE_PATRONYMIC_RE.search, words) if m)
    elif gender == "m":
        cases = "".join(_MASCULINE_PATRONYMIC_CASES[m[1]] for m in map(_MASCULINE_PATRONYMIC_RE.search, words) if m)
    if cases:
        kept = [frozenset().union(*(fitting(case + gender)[k] for case in cases)) for k in range(len(words))]
        return [(gender, tuple(kept[k] or plain[k] for k in range(len(words))))]
    if len(words) != 2:
        return [(gender, plain)]
    ways: dict[tuple[_Readings, ...], set[str]] = {}
    for one in (gender or "fm"):
        for case in "NGDAIL":
            readings = fitting(case + one)
            if all(readings):
                ways.setdefault(readings, set()).add(one)
    # A way that another one contains adds nothing: «Заєць Валентина» as a man called «Валентина» is part of the
    # woman's nominative, and the pair then tells no gender.
    for narrow in sorted(ways, key=lambda w: sum(map(len, w))):
        wider = next((w for w in ways if w != narrow and all(a <= b for a, b in zip(narrow, w))), None)
        if wider is not None:
            ways[wider] |= ways.pop(narrow)
    if not ways:
        return [(gender, plain)]
    return [(gender or (min(genders) if len(genders) == 1 else ""), readings) for readings, genders in ways.items()]


def _name_items(fragment: str) -> list[tuple[str, bool]]:
    items: list[tuple[str, bool]] = []
    lower_case: list[tuple[str, bool]] = []
    for token in _WORD_RE.findall(fragment):
        token = _fix_script(_APOSTROPHE_RE.sub("", token))
        if not token:
            continue
        if len(token) == 1:
            # A lower-case single letter is a conjunction or a preposition (і, й, в, з), not an initial.
            if token.isupper():
                items.append((token, False))
            continue
        word = token.lower()
        if word in _TITLES:
            continue
        (items if token[0].isupper() else lower_case).append((word, True))
    # Names are capitalised; lower-case words are kept only when the span has nothing else to go by.
    return items or lower_case


_HYPHEN = "\x1f"  # a white-space character no text has
_JOINED_RE = re.compile(rf"(?<=[^\W\d_]{{2}}){_HYPHEN}(?=[^\W\d_]{{2}})")


@lru_cache(maxsize=65536)
def _parse_mention(raw: str) -> tuple[tuple[_Mention, ...], tuple[_Mention, ...], tuple[_Mention, ...]]:
    """The mention read with agreement between its words (one or several ways), read word by word, and read with no
    rule at all."""
    # Hyphenated surnames are compared part by part: Нечуй-Левицький - Нечуя-Левицького. The hyphen becomes a
    # separator that is still told from a space, to count the names.
    text = re.sub(r"\s*[-‐‑–—]\s*", _HYPHEN, raw)
    parts = [p for p in _NAME_SEPARATOR_RE.split(text) if p.strip()]
    items: list[tuple[str, bool]] = []
    chosen = text
    for part in parts:
        items = _name_items(part)
        if any(is_word for _, is_word in items):
            chosen = part
            break
    if not items:
        items = _name_items(text)
    words = tuple(v for v, is_word in items if is_word)
    gender = _mention_gender(words, len(words) - len(_JOINED_RE.findall(chosen)))
    shape = tuple(items)
    return (
        tuple(_Mention(shape, way_gender, readings) for way_gender, readings in _strict_readings(words, gender)),
        (_Mention(shape, gender, _loose_readings(words, gender)),),
        (_Mention(shape, gender, tuple(_analyses(w) for w in words)),),
    )


def _pair_words(ma: _Mention, mb: _Mention) -> list[tuple[int, int, int]]:
    """A largest set of (index in a, index in b, match kind) pairs of matching words, each word used once."""
    a, b = ma.words, mb.words
    best: list[tuple[int, int, int]] = []
    # Names have three or four words; anything much longer is a broken span and a greedy pairing is good enough.
    exhaustive = len(a) <= 5 and len(b) <= 5

    def walk(i: int, used: int, acc: list[tuple[int, int, int]]) -> None:
        nonlocal best
        if i == len(a):
            if len(acc) > len(best):
                best = list(acc)
            return
        if len(acc) + (len(a) - i) <= len(best):
            return
        for j, wb in enumerate(b):
            if used >> j & 1:
                continue
            kind = _match_words(a[i], wb, ma.readings[i], mb.readings[j])
            if kind:
                acc.append((i, j, kind))
                walk(i + 1, used | 1 << j, acc)
                acc.pop()
                if not exhaustive:
                    return
        walk(i + 1, used, acc)

    walk(0, 0, [])
    return best


def _is_run(short: list[str], long: list[str]) -> bool:
    n = len(short)
    return any(long[i:i + n] == short for i in range(len(long) - n + 1))


_CONFLICT, _NEUTRAL = -1, 0


@lru_cache(maxsize=262144)
def _compare(a: _Mention, b: _Mention) -> int:
    """_CONFLICT (different persons), _NEUTRAL (nothing in common to judge by) or a positive score of the match."""
    if a.gender and b.gender and a.gender != b.gender:
        return _CONFLICT
    words_a, words_b = a.words, b.words
    pairs = _pair_words(a, b)
    if (a.gender == "f" or b.gender == "f") and any(kind == _SAME_MASCULINE for _, _, kind in pairs):
        return _CONFLICT
    if words_a and words_b and not pairs:
        # Two mentions of one person with two words each must share a word (a name has three); «Гончаренка» and
        # «Віктора Павловича» may still be one person.
        return _CONFLICT if len(words_a) >= 2 and len(words_b) >= 2 else _NEUTRAL

    def rest(m: _Mention, matched: set[int]) -> tuple[list[str], bool]:
        letters, has_word, k = [], False, 0
        for value, is_word in m.items:
            if is_word:
                if k not in matched:
                    letters.append(value[0].upper())
                    has_word = True
                k += 1
            else:
                letters.append(value)
        return letters, has_word

    rest_a, word_left_a = rest(a, {i for i, _, _ in pairs})
    rest_b, word_left_b = rest(b, {j for _, j, _ in pairs})
    if word_left_a and word_left_b and pairs:
        return _CONFLICT
    short, long = (rest_a, rest_b) if len(rest_a) <= len(rest_b) else (rest_b, rest_a)
    if not pairs:
        # At least one side is initials only: «І. П.» may stand anywhere in «Коваль Іван Петрович».
        if not short or not long:
            return _NEUTRAL
        if not words_a and not words_b:
            return len(short) if long[: len(short)] == short else _CONFLICT
        return len(short) if _is_run(short, long) else _CONFLICT
    if long[: len(short)] != short:
        return _CONFLICT
    return 10 * len(pairs) + len(short)


def _covers(head: _Mention, m: _Mention) -> bool:
    """Is every word of `m` a word of `head`?"""
    return len(_pair_words(head, m)) == len(m.words)


def _compare_ways(a: tuple[_Mention, ...], b: tuple[_Mention, ...]) -> int:
    """`_compare` of two mentions that may each be read in several ways: the best result of any two of them."""
    return max(_compare(x, y) for x in a for y in b)


# A placeholder left in the text by an earlier anonymisation: «ОСОБА_1», «Особа2», «ОСОБА 1». A digit after a letter
# or an underscore marks one; a year that the span swallowed («Коваль І. П., 1980р.н.») does not.
_PLACEHOLDER_RE = re.compile(r"[^\W\d]\d")
_PLACEHOLDER_ALONE_RE = re.compile(r"([^\W\d_]+)([_\s]?)(\d+)")
_PLACEHOLDER_WORDS = frozenset((PERSON.lower(), ADDRESS.lower(), NUMBER.lower(), INFO.lower()))


def _placeholder_key(raw: str) -> str | None:
    """The key of a mention that is a placeholder, or None. «ОСОБА_1,» = «особа_1»; «ОСОБА_1» != «ОСОБА_2»."""
    alone = _PLACEHOLDER_ALONE_RE.fullmatch(re.sub(r"^[\W_]+|[\W_]+$", "", raw))
    # Written with a space only the four words of the task are placeholders: «Коваль 1980» is a name and a year.
    if alone and (not alone[2].isspace() or alone[1].lower() in _PLACEHOLDER_WORDS):
        return f"{alone[1].lower()}_{int(alone[3])}"
    if _PLACEHOLDER_RE.search(raw):
        return " ".join(raw.lower().split())  # several placeholders in one span, or one with words around it
    return None


# A mention read with agreement between its words (strict: one or several ways), word by word (loose), and with every
# reading of every word (plain).
_Parsed = tuple[tuple[_Mention, ...], tuple[_Mention, ...], tuple[_Mention, ...]]
_STRICT, _LOOSE, _PLAIN = 0, 1, 2
# The score of three paired words: the whole name.
_WHOLE_NAME = 30


@dataclass
class _Entity:
    head: _Parsed
    forms: set[_Parsed] = field(default_factory=set)
    positions: list[int] = field(default_factory=list)


def _recency(positions: list[int], position: int) -> tuple[int, int]:
    """Sort key: the entity mentioned latest before `position` first; then the one mentioned soonest after it."""
    before = [p for p in positions if p < position]
    if before:
        return 1, max(before)
    return 0, -min(positions)


@lru_cache(maxsize=65536)
def _stems(word: str) -> frozenset[str]:
    return frozenset(stem for stem, _, _ in _analyses(word))


class _HeadIndex:
    """Finds the entities whose head a mention can fit at all, so that a mention is not compared with every person.

    `_compare` is positive only when one of the two mentions has every word paired, and paired words share a stem. So
    the candidates are the heads that have a stem of every word of the mention, the heads whose every word has a stem
    of the mention, and the heads without words (initials). Nothing else can fit, in any reading: the result of the
    clustering is the same as without the index."""

    def __init__(self) -> None:
        self.by_stem: dict[str, list[tuple[int, int]]] = {}  # stem -> (entity, number of the word in its head)
        self.words: dict[int, int] = {}  # entity -> the number of words in its head
        self.wordless: list[int] = []

    def add(self, e: int, head: _Mention) -> None:
        self.words[e] = len(head.words)
        if not head.words:
            self.wordless.append(e)
        for k, word in enumerate(head.words):
            for stem in _stems(word):
                self.by_stem.setdefault(stem, []).append((e, k))

    def candidates(self, m: _Mention) -> list[int]:
        if not m.words:
            return sorted(self.words)
        mine: dict[int, set[int]] = {}  # entity -> the words of the mention that it has
        theirs: dict[int, set[int]] = {}  # entity -> the words of its head that the mention has
        for k, word in enumerate(m.words):
            for stem in _stems(word):
                for e, j in self.by_stem.get(stem, ()):
                    mine.setdefault(e, set()).add(k)
                    theirs.setdefault(e, set()).add(j)
        fitting = [e for e in mine if len(mine[e]) == len(m.words) or len(theirs[e]) == self.words[e]]
        return sorted(fitting + self.wordless)


def _best_entity(entities: list[_Entity], candidates: list[int], parsed: _Parsed, position: int, reading: int) -> tuple[int, int]:
    """The entity the mention joins when all mentions are taken in one reading (_STRICT, _LOOSE or _PLAIN) and the
    score of the match, or (-1, 0)."""
    m = parsed[reading]
    best, best_key = -1, (0, 0, 0)
    for e in candidates:
        entity = entities[e]
        head = entity.head[reading]
        score = _compare_ways(head, m)
        if score <= 0:
            continue
        # The other mentions are consulted only for what the head cannot vouch for: «Олег Шевченко» fits the head
        # «Шевченко О. В.» but not its «Олена Шевченко»; «Зіновієм Марчуком» and «Зіновія Устимовича» are both
        # parts of the head «Марчук Зіновій Устимович».
        covered = any(_covers(x, y) for x in head for y in m)
        if not covered and any(_compare_ways(f[reading], m) == _CONFLICT for f in entity.forms):
            continue
        key3 = (score, *_recency(entity.positions, position))
        if key3 > best_key:
            best, best_key = e, key3
    return best, best_key[0]


def _link_persons(mentions: list[str]) -> list[int]:
    parsed = [_parse_mention(m) for m in mentions]
    entities: list[_Entity] = []
    # Mentions with no letters at all and placeholders are linked by their exact text: the word «особа» says nothing,
    # and «ОСОБА_1» is not «ОСОБА_2».
    by_text: dict[str, int] = {}
    exact = (_Mention((), ""),)
    cluster = [0] * len(mentions)
    index = _HeadIndex()
    richness = [p[_LOOSE][0].richness for p in parsed]
    order = sorted(range(len(mentions)), key=lambda i: (-richness[i][0], -richness[i][1], i))
    for i in order:
        m = parsed[i]
        key = _placeholder_key(mentions[i])
        if key is None and not m[_LOOSE][0].items:
            key = " ".join(mentions[i].lower().split())
        if key is not None:
            if key not in by_text:
                by_text[key] = len(entities)
                entities.append(_Entity((exact, exact, exact), set(), [i]))
            else:
                entities[by_text[key]].positions.append(i)
            cluster[i] = by_text[key]
            continue
        candidates = index.candidates(m[_PLAIN][0])
        best, strict_score = _best_entity(entities, candidates, m, i, _STRICT)
        loose, loose_score = _best_entity(entities, candidates, m, i, _LOOSE)
        # Agreement only chooses between persons that fit equally well. A mention that fits nobody when read strictly
        # is read word by word before it starts a person: clerks do leave a man's surname undeclined («позов Шевченко
        # Олега»). And a word-by-word fit on the whole name («Седих Нодара» to «Седих Нодар Зурабович») is not given
        # up for a strict fit on the surname and an initial («Седих Н. В.»).
        if best < 0 or loose_score > strict_score:
            best = loose
        if best < 0:
            # «Щербина Устима Гордійовича» after «Щербини Устима Гордійовича»: a man's surname in -а left undeclined
            # looks like the genitive of «Щербин», and `_loose_readings` takes it so. Before it starts a person the
            # mention is read with no such rule, and joins only a person whose whole name it repeats: for anything
            # shorter («Нікітіною Д. Ю.» next to «Нікітіна Дениса Юрійовича») the possessive reading stands.
            plain, plain_score = _best_entity(entities, candidates, m, i, _PLAIN)
            if plain_score >= _WHOLE_NAME:
                best = plain
        if best < 0:
            best = len(entities)
            entities.append(_Entity(m))
            index.add(best, m[_PLAIN][0])
        entities[best].forms.add(m)
        entities[best].positions.append(i)
        cluster[i] = best
    return cluster


# --------------------------------------------------------------------------------------------------------------------
# Common words (institutions, street and settlement names)
# --------------------------------------------------------------------------------------------------------------------

_COMMON_ENDINGS = ("ами", "ями", "ими", "ам", "ям", "ах", "ях", "их", "ів", "їв", "ей", "е", "є", "и", "і")


@lru_cache(maxsize=65536)
def _common_stems(word: str) -> frozenset[str]:
    stems = {stem for stem, _, _ in _analyses(word)}
    for ending in _COMMON_ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            stems.add(word[: len(word) - len(ending)].replace("ь", ""))
    return frozenset(stems)


def _same_common_word(a: str, b: str) -> bool:
    if a == b:
        return True
    # Anything with a digit is an identifier (А1234, 8-го): only the exact token counts.
    if any(c.isdigit() for c in a) or any(c.isdigit() for c in b):
        return False
    if "-" in a and "-" in b:
        # Both halves are declined: Кам'янець-Подільський - Кам'янця-Подільського.
        halves_a, halves_b = a.split("-"), b.split("-")
        return len(halves_a) == len(halves_b) and all(_same_common_word(x, y) for x, y in zip(halves_a, halves_b))
    return not _common_stems(a).isdisjoint(_common_stems(b))


def _same_phrase(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    return len(a) == len(b) and all(_same_common_word(x, y) for x, y in zip(a, b))


def _greedy_clusters(records: list, same: Callable[[object, object], bool]) -> list[int]:
    """Give each record the cluster of the first earlier representative it equals; `same` need not be transitive."""
    representatives: list = []
    out: list[int] = []
    for record in records:
        for k, other in enumerate(representatives):
            if same(other, record):
                out.append(k)
                break
        else:
            out.append(len(representatives))
            representatives.append(record)
    return out


# --------------------------------------------------------------------------------------------------------------------
# НОМЕР
# --------------------------------------------------------------------------------------------------------------------

_PHONE_SHAPE_RE = re.compile(r"^\+?[\d\s()\-.]+$")
_PHONE_LABEL_RE = re.compile(r"^(?:тел(?:ефон)?|моб(?:ільний)?)\s*[.:]?\s*", re.IGNORECASE)


def _phone_key(raw: str) -> str | None:
    """+380 67 123 45 67 = 8 067 123 45 67 = 067 123-45-67: Ukrainian phone numbers by their last nine digits."""
    raw = _PHONE_LABEL_RE.sub("", raw)  # «тел. 067 111 22 33»
    if not _PHONE_SHAPE_RE.match(raw):
        return None
    digits = re.sub(r"\D", "", raw)
    international = len(digits) == 12 and digits.startswith("380")
    # 8 0XX is the long-distance prefix of the old numbering plan; the local form has no «+».
    local = "+" not in raw and ((len(digits) == 11 and digits.startswith("80")) or (len(digits) == 10 and digits.startswith("0")))
    return "tel:" + digits[-9:] if international or local else None


def _number_key(raw: str) -> str:
    phone = _phone_key(raw.strip())
    if phone:
        return phone
    # The slash is kept: it parts the fields of a case number, and «755/1234/21» is not «755/123/421». Spaces, dots
    # and hyphens are only a way of writing («АВ 123456» = «АВ-123456», a card number in groups of four).
    key = "".join(c for c in raw.upper().translate(_CYRILLIC_TO_LATIN) if c.isalnum() or c == "/").strip("/")
    return key or " ".join(raw.split())


# --------------------------------------------------------------------------------------------------------------------
# ІНФОРМАЦІЯ
# --------------------------------------------------------------------------------------------------------------------

_MONTH_PREFIXES = ("січ", "лют", "берез", "квіт", "трав", "черв", "лип", "серп", "верес", "жовт", "листопад", "груд")
_DATE_ISO_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_DATE_NUMERIC_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4}|\d{2})(?!\d)")
_DATE_WORDS_RE = re.compile(r"(?<!\d)(\d{1,2})\s+([^\W\d_]+)\s+(\d{4})(?!\d)")
# What may stand around a date without making the mention something else: «року народження», «р.н.», «від».
_DATE_FILLER = frozenset("року рік р рр народження рн н від дата дати день".split())
_QUOTES_RE = re.compile(r"[\"«»“”„]")
_UNIT_RE = re.compile(r"\bв\s*[/.]\s*ч\b\.?", re.IGNORECASE)  # «в/ч», «в.ч.»
_UNIT_WORDS = frozenset("частина частини частині частину частиною".split())
# Short words that stand before a number without being the letters of its code: «З 2019 року», «ДО 2020 року».
_SHORT_WORDS = frozenset("у в з і й о до на за по із зі та чи не".split())
_ORDINAL_RE = re.compile(r"(\d+)-[а-яіїє]{1,3}")
# An account or a page: «https://instagram.com/name_1», «@name_1», «name.surname». Only its exact spelling counts.
_HANDLE_RE = re.compile(
    r"(?:https?://|www\.|@)\S+|[a-z0-9/-]*[a-z][a-z0-9/-]*[._][a-z0-9._/-]+|[^\W\d_]+_\w+", re.IGNORECASE | re.ASCII
)
_INFO_TOKEN_RE = re.compile(rf"[^\W_]+(?:[-/{_APOSTROPHES}][^\W_]+)*")
# Abbreviations that courts use next to the full name of the same body. An expansion only has to be the one the same
# decision would spell out: a wrong guess («ДП» as «дочірнє підприємство») leaves two spellings apart, as before.
_ABBREVIATIONS = {
    short: tuple(full.split())
    for short, full in {
        "вч": "військова частина",
        "тцк": "територіальний центр комплектування",
        "ртцк": "районний територіальний центр комплектування",
        "сп": "соціальної підтримки",
        "моу": "міністерство оборони україни",
        "зсу": "збройні сили україни",
        "тов": "товариство з обмеженою відповідальністю",
        "дп": "державне підприємство",
        "кп": "комунальне підприємство",
        "пп": "приватне підприємство",
        "ат": "акціонерне товариство",
        "пат": "публічне акціонерне товариство",
        "прат": "приватне акціонерне товариство",
        "міноборони": "міністерство оборони україни",
    }.items()
}


def _full_year(year: int) -> int:
    # Two-digit years in court decisions are birth years or recent dates; split them at a fixed pivot.
    if year >= 100:
        return year
    return 1900 + year if year > 30 else 2000 + year


def _date_key(raw: str) -> tuple[int, int, int] | None:
    low = _QUOTES_RE.sub("", raw.lower())  # «17» квітня 1992 року
    found: tuple[int, int, int] | None = None
    span = (0, 0)
    m = _DATE_ISO_RE.search(low)
    if m:
        found, span = (int(m[1]), int(m[2]), int(m[3])), m.span()
    if found is None:
        m = _DATE_NUMERIC_RE.search(low)
        if m:
            found, span = (_full_year(int(m[3])), int(m[2]), int(m[1])), m.span()
    if found is None:
        m = _DATE_WORDS_RE.search(low)
        if m:
            month = next((k + 1 for k, p in enumerate(_MONTH_PREFIXES) if m[2].startswith(p)), 0)
            if month:
                found, span = (int(m[3]), month, int(m[1])), m.span()
    if found is None or not (1 <= found[1] <= 12 and 1 <= found[2] <= 31):
        return None
    around = low[: span[0]] + " " + low[span[1]:]
    if any(c.isdigit() for c in around) or any(w not in _DATE_FILLER for w in _WORD_RE.findall(around)):
        return None
    return found


def _info_record(raw: str) -> tuple:
    """("date", y, m, d) | ("mail", address) | ("tel", digits) | ("handle", address) | ("name", (words...))."""
    text = raw.strip()
    if "@" in text[1:]:  # «@name» is an account, not an address
        return "mail", re.sub(r"[\s<>,;]+", "", text).strip(".").lower()
    date = _date_key(text)
    if date:
        return ("date", *date)
    phone = _phone_key(text)
    if phone:
        return "tel", phone
    if _HANDLE_RE.fullmatch(text.rstrip(".,;")):
        return "handle", re.sub(r"^(?:https?://)?(?:www\.)?@?", "", text.lower()).rstrip("/.,;")
    plain = _UNIT_RE.sub("військова частина ", _QUOTES_RE.sub(" ", text))
    words: list[str] = []
    code_letters = ""  # the previous token when it was the letters of a code written apart
    for raw_token in _INFO_TOKEN_RE.findall(plain):
        token = _APOSTROPHE_RE.sub("", raw_token)
        low = _fix_script(token).lower()
        ordinal = _ORDINAL_RE.fullmatch(low)
        # «в/ч А 1234» = «в/ч А1234». After «частина» any letter is the letter of the code («в/ч а 1234»); elsewhere
        # only capitals that are no short word are, so that «у 2019 році» and «З 2019 року» keep their preposition.
        letters = len(token) <= 2 and token.isalpha() and low not in _ABBREVIATIONS and (
            (len(token) == 1 and bool(words) and words[-1] in _UNIT_WORDS) or (token.isupper() and low not in _SHORT_WORDS)
        )
        if token.isdigit() and code_letters:
            words[-1] = _number_key(code_letters + token)
        elif low in _ABBREVIATIONS:
            words.extend(_ABBREVIATIONS[low])
        elif ordinal:
            words.append(ordinal[1])  # «110-та бригада» = «110 бригада» = «110-ї бригади»
        elif any(c.isdigit() for c in token):
            words.append(_number_key(token))
        else:
            words.append(low)
        code_letters = token.upper() if letters else ""
    if not words:
        return "text", " ".join(text.lower().split())
    return "name", tuple(words)


def _same_info(a: tuple, b: tuple) -> bool:
    if a[0] != b[0]:
        return False
    if a[0] == "name":
        return _same_phrase(a[1], b[1])
    return a == b


# --------------------------------------------------------------------------------------------------------------------
# АДРЕСА
# --------------------------------------------------------------------------------------------------------------------

_SETTLEMENT, _STREET, _HOUSE, _BUILDING = "settlement", "street", "house", "building"
_APARTMENT, _REGION, _DISTRICT = "apt", "region", "district"


def _markers(kind: str, value: str, forms: str) -> dict[str, tuple[str, str]]:
    return {form: (kind, value) for form in forms.split()}


_ADDRESS_MARKERS: dict[str, tuple[str, str]] = {
    **_markers(_SETTLEMENT, "", "м місто міста місті містом с село села селі смт сел селище селища селищі с-ще хутір х"),
    **_markers(_STREET, "вул", "вул вулиця вулиці вулицю вулицею"),
    **_markers(_STREET, "просп", "просп проспект проспекту проспекті пр-т пр-кт пр"),
    **_markers(_STREET, "пров", "пров провулок провулку пр-к"),
    **_markers(_STREET, "бульв", "бульв бульвар бульвару бульварі б-р"),
    **_markers(_STREET, "пл", "пл площа площі площу"),
    **_markers(_STREET, "шосе", "шосе ш"),
    **_markers(_STREET, "узвіз", "узвіз узвозу узвозі"),
    **_markers(_STREET, "наб", "наб набережна набережної набережній"),
    **_markers(_STREET, "майдан", "майдан майдану майдані"),
    **_markers(_STREET, "проїзд", "проїзд проїзду"),
    **_markers(_STREET, "тупик", "тупик тупику туп"),
    **_markers(_STREET, "мкр", "мкр мікрорайон мікрорайону мікрорайоні м-н мкрн"),
    **_markers(_STREET, "кв-л", "кв-л квартал кварталу"),
    **_markers(_HOUSE, "", "буд будинок будинку будинком б дім д"),
    **_markers(_BUILDING, "", "корп корпус корпусу к"),
    **_markers(_APARTMENT, "", "кв квартира квартири квартирі квартиру оф офіс офісу кімн кімната кімнати кім"),
    **_markers(_REGION, "", "обл область області"),
    **_markers(_DISTRICT, "", "р-н район району районі"),
}
_ADDRESS_NOISE = frozenset("по на у в за адресою адреса адресу індекс".split())
_COUNTRY = frozenset(("україна", "україни"))
_ADDRESS_TOKEN_RE = re.compile(
    rf"\d+(?:\s?[-/]\s?[^\W_]+)*[^\W\d_]?(?![^\W\d_])|[^\W\d_]+(?:[-{_APOSTROPHES}][^\W_]+)*|\d+|[,;\n]"
)


@dataclass
class _Address:
    settlement: tuple[str, ...] = ()
    street_type: str = ""
    street: tuple[str, ...] = ()
    house: str = ""
    building: str = ""
    apartment: str = ""
    region: tuple[str, ...] = ()
    district: tuple[str, ...] = ()
    fallback: str = ""


def _unit(tokens: list[str]) -> str:
    """House / apartment numbers as written keys: «1-А», «1 а», «1А» are one house; «1-2» is not house 12."""
    text = re.sub(r"(?<=\d)-(?=\d)", "~", "".join(tokens).upper().translate(_CYRILLIC_TO_LATIN))
    return "".join(c for c in text if c.isalnum() or c in "/~").replace("~", "-")


def _is_numeric(token: str) -> bool:
    return token[:1].isdigit()


def _parse_address(raw: str) -> _Address:
    addr = _Address()
    # Prepositions are told from initials inside street names («вул. В. Стуса») by their case, so drop them first.
    found = [t if t not in _ADDRESS_NOISE else "," for t in _ADDRESS_TOKEN_RE.findall(raw)]
    tokens = [_APOSTROPHE_RE.sub("", t.lower()) for t in found]
    tokens = [re.sub(r"\s", "", t) if _is_numeric(t) else t for t in tokens]
    parts: list[list[str]] = []  # [kind, value, token, token, ...]; kind "" = no marker seen
    current: list[str] | None = None
    for token in tokens + [","]:
        if token in ",;\n":
            current = None
            continue
        marker = None if _is_numeric(token) else _ADDRESS_MARKERS.get(token)
        in_street = current is not None and current[0] == _STREET
        if marker and in_street and (marker[0] == _STREET or (len(token) == 1 and len(current) == 2)):
            # Inside a street name these are words of the name: «вул. Героїв Майдану», «площа Майдан Незалежності»,
            # and the initial of «вул. М. Грушевського» is no «місто».
            marker = None
        if token in _COUNTRY and not (in_street and len(current) > 2):
            current = None  # the country says nothing, unless a street is named after it: «вул. Героїв України»
            continue
        if marker and marker[0] in (_REGION, _DISTRICT):
            # The name stands before the marker: «Одеська область», «Кельменецький район».
            name = [current.pop()] if current and len(current) > 2 and not _is_numeric(current[-1]) else []
            parts.append([marker[0], "", *name])
            current = None
        elif marker:
            current = [marker[0], marker[1]]
            parts.append(current)
        else:
            if current is None:
                current = ["", ""]
                parts.append(current)
            current.append(token)

    def set_unit(tokens_: list[str]) -> None:
        if not addr.house:
            addr.house = _unit(tokens_)
        elif not addr.apartment:
            addr.apartment = _unit(tokens_)

    for kind, value, *body in parts:
        if not body:
            continue
        if kind == _STREET or kind == _SETTLEMENT:
            # «вул. Хрещатик 1»: a trailing number is the house, unless it is all there is («вул. 8 Березня»).
            trailing: list[str] = []
            while len(body) > 1 and _is_numeric(body[-1]):
                trailing.insert(0, body.pop())
            if kind == _STREET and not addr.street:
                addr.street, addr.street_type = tuple(body), value
            elif kind == _SETTLEMENT and not addr.settlement:
                addr.settlement = tuple(body)
            if trailing:
                set_unit(trailing)
        elif kind == _HOUSE:
            addr.house = addr.house or _unit(body)
        elif kind == _BUILDING:
            addr.building = addr.building or _unit(body)
        elif kind == _APARTMENT:
            addr.apartment = addr.apartment or _unit(body)
        elif kind == _REGION:
            addr.region = addr.region or tuple(body)
        elif kind == _DISTRICT:
            addr.district = addr.district or tuple(body)
        elif _is_numeric(body[0]):
            if len(body) == 1 and len(body[0]) == 5 and body[0].isdigit():
                continue  # postal index: optional, never tells two addresses apart
            set_unit(body)
        elif not addr.settlement and not addr.street:
            addr.settlement = tuple(body)
        elif not addr.street:
            addr.street = tuple(body)
    if not (addr.settlement or addr.street or addr.house):
        addr.fallback = _number_key(raw)
    return addr


def _same_optional(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    return not a or not b or _same_phrase(a, b)


def _same_street(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """«вул. Мазепи» = «вул. Івана Мазепи» = «вул. І. Мазепи»: a street named after a person is known by the surname."""
    if _same_phrase(a, b):
        return True
    if not a or not b or not _same_common_word(a[-1], b[-1]):
        return False
    first_a, first_b = a[:-1], b[:-1]
    if not first_a or not first_b:
        # One word may be added, a first name or its initial; a number is another street («8 Березня»).
        added = first_a or first_b
        return len(added) == 1 and not _is_numeric(added[0])
    if len(first_a) != len(first_b):
        return False
    return all(_same_common_word(x, y) or (1 in (len(x), len(y)) and x[0] == y[0]) for x, y in zip(first_a, first_b))


def _same_address(a: _Address, b: _Address) -> bool:
    if a.fallback or b.fallback:
        return a.fallback == b.fallback
    if (a.house, a.building, a.apartment) != (b.house, b.building, b.apartment):
        return False
    if bool(a.street) != bool(b.street) or not _same_street(a.street, b.street):
        return False
    if a.street_type and b.street_type and a.street_type != b.street_type:
        return False
    return (
        _same_optional(a.settlement, b.settlement)
        and _same_optional(a.region, b.region)
        and _same_optional(a.district, b.district)
    )


# --------------------------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------------------------


def _link_exact(mentions: list[str]) -> list[int]:
    keys: dict[str, int] = {}
    return [keys.setdefault(" ".join(m.lower().split()), len(keys)) for m in mentions]


def _link_numbers(mentions: list[str]) -> list[int]:
    keys: dict[str, int] = {}
    return [keys.setdefault(_number_key(m), len(keys)) for m in mentions]


def _link_info(mentions: list[str]) -> list[int]:
    return _greedy_clusters([_info_record(m) for m in mentions], _same_info)


def _link_addresses(mentions: list[str]) -> list[int]:
    records = [_parse_address(m) for m in mentions]
    representatives: list[_Address] = []
    positions: list[list[int]] = []
    out = [0] * len(records)
    # Addresses with a settlement first. One without («вул. Миру, 5» after the same street in Харків and in Суми) is
    # then placed like a bare surname among persons: with the matching address mentioned last before it.
    for i in sorted(range(len(records)), key=lambda i: (not (records[i].settlement or records[i].fallback), i)):
        fits = [k for k, other in enumerate(representatives) if _same_address(other, records[i])]
        if fits:
            out[i] = max(fits, key=lambda k: _recency(positions[k], i))
        else:
            out[i] = len(representatives)
            representatives.append(records[i])
            positions.append([])
        positions[out[i]].append(i)
    return out


_LINKERS: dict[str, Callable[[list[str]], list[int]]] = {
    PERSON: _link_persons,
    ADDRESS: _link_addresses,
    NUMBER: _link_numbers,
    INFO: _link_info,
}


def _mention_text(text: str, span: dict) -> str:
    try:
        start, end = int(span.get("start", 0)), int(span.get("end", 0))
        return text[max(start, 0): max(end, 0)] if isinstance(text, str) else ""
    except Exception:  # noqa: BLE001 - an offset that is None, a string, NaN or infinite: no text, never an error
        return ""


def _position(spans: list, index: int) -> tuple[int, int, int, int]:
    """Sort key of a span: its place in the text; a span without usable offsets goes last, in the order given."""
    try:
        return 0, int(spans[index].get("start", 0)), int(spans[index].get("end", 0)), index
    except Exception:  # noqa: BLE001 - not a dict, or an offset that is no number
        return 1, 0, 0, index


def assign(text: str, spans: list[dict]) -> list[str]:
    """One pseudonym «<TYPE>_<n>» per span, in the order of `spans`. Equal pseudonyms mean the same entity.

    The entities do not depend on the order of `spans`: mentions are linked in the order of their offsets, because
    «the person named last before this mention» is a matter of the text. Only the numbers follow the list: n counts
    the entities of a type as they first appear in `spans`."""
    by_type: dict[str, list[int]] = {}
    for i, span in enumerate(spans):
        kind = span.get("type") if isinstance(span, dict) else None
        by_type.setdefault(str(kind) if kind is not None else "", []).append(i)
    out = [""] * len(spans)
    for kind, indices in by_type.items():
        in_text_order = sorted(indices, key=lambda i: _position(spans, i))
        mentions = [_mention_text(text, spans[i]) if isinstance(spans[i], dict) else "" for i in in_text_order]
        try:
            clusters = _LINKERS.get(kind, _link_exact)(mentions)
            if len(clusters) != len(mentions):
                raise ValueError("linker returned a wrong number of clusters")
        except Exception:  # noqa: BLE001 - a pseudonym for every span matters more than linking one odd document
            clusters = list(range(len(mentions)))
        cluster_of = dict(zip(in_text_order, clusters))
        numbers: dict[int, int] = {}
        for i in indices:
            out[i] = f"{kind}_{numbers.setdefault(cluster_of[i], len(numbers) + 1)}"
    return out
