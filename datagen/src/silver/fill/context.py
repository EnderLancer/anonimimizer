"""Context heuristics: grammatical case for a person mention and sub-type of НОМЕР / ІНФОРМАЦІЯ
placeholders, decided from the text around the placeholder (which still contains placeholders)."""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

LEFT = 45

# --- case of a person mention ---------------------------------------------------------------
_GENT_PREP = re.compile(
    r"(?:\b(?:до|від|у|в|з|із|зі|без|для|біля|щодо|стосовно|проти|відносно|після|крім|окрім|замість|внаслідок|"
    r"коло|серед|поблизу|задля|з боку|на користь|в інтересах|за участю|за участі|за позовом|за заявою|за скаргою|"
    r"місце проживання|проживання|перебування|реєстрації|народження|житла|дитини|доньки|сина|утримання|виховання|"
    r"опіки|піклування|усиновлення|батьківства|заборгованість|заборгованості|вини|винуватості|дії|бездіяльність|"
    r"за клопотанням|на ім'я|на ім’я|позов|позову|позовом|скарга|скаргу|скарги|скаргою|заява|заяву|заяви|заявою|"
    r"клопотання|клопотанням|вимоги|вимогу|показання|пояснення|представник|представника|представником|захисник|"
    r"захисника|захисником|спадкоємець|спадкоємця|спадщина|спадщину|спадщини|майно|майна|смерті|частка|частку|"
    r"частки|права|прав|інтереси|інтересів|борг|боргу|дохід|доходу|заробітк[уа]|пенсії|пенсію|аліменти|аліментів|"
    r"стягнути з|стягнення з|стягнуто з|відносно|син|сина|дочка|доньки|дочки|дружина|дружини|чоловік|чоловіка|"
    r"батько|батька|мати|матері|брат|брата|сестра|сестри|онук|онука|опікун|опікуна|рідним батьком|рідною матір'ю)"
    r"[,:]?\s*$)",
    re.I,
)
_GENT_ROLE = re.compile(
    r"\b(?:обвинувачен(?:ого|ої)|засуджен(?:ого|ої)|підозрюван(?:ого|ої)|потерпіл(?:ого|ої)|позивача|позивачки|"
    r"відповідача|відповідачки|померл(?:ого|ої)|неповнолітн(?:ого|ьої)|малолітн(?:ого|ьої)|боржника|стягувача|"
    r"заявника|заявниці|скаржника|апелянта|третьої особи|свідка|експерта|перекладача|адвоката|прокурора|судді|"
    r"нотаріуса|виконавця|громадянина|громадянки|фізичної особи|особи|гр\.|гр)[,:]?\s*$",
    re.I,
)
_DATV = re.compile(
    r"\b(?:передати|надати|повернути|присудити|повідомити|вручити|роз'яснити|роз’яснити|призначити|надано|"
    r"належить|належало|повернення|відшкодувати|сплатити|виплатити|компенсувати|зарахувати|"
    r"обвинувачен(?:ому|ій)|засуджен(?:ому|ій)|підозрюван(?:ому|ій)|потерпіл(?:ому|ій)|позивачу|позивачці|"
    r"відповідачу|відповідачці|боржнику|стягувачу|заявнику|заявниці|неповнолітн(?:ьому|ій)|малолітн(?:ьому|ій))[,:]?\s*$",
    re.I,
)
_ACCS = re.compile(
    r"\b(?:визнати|притягнути|допитати|засудити|звільнити|виселити|зобов'язати|зобов’язати|оголосити|затримати|"
    r"взяти|викликати|попередити|помістити|доставити|етапувати|виправдати|поновити|відсторонити|усиновити|"
    r"встановити над|на|про|через|за|під)\s*$",
    re.I,
)
_ABLT = re.compile(
    r"\b(?:з|зі|із|разом з|разом із|між|над|під|перед|за|укладен\w+ між|підписан\w+|поданий|подана|поданих|"
    r"обвинувачен(?:им|ою)|засуджен(?:им|ою)|підозрюван(?:им|ою)|потерпіл(?:им|ою)|позивачем|позивачкою|"
    r"відповідачем|відповідачкою|представленого|головуючим|суддею|адвокатом|прокурором|нотаріусом|секретарем)[,:]?\s*$",
    re.I,
)
_NOMN_ROLE = re.compile(
    r"\b(?:позивач|відповідач|суддя|головуючий|адвокат|прокурор|нотаріус|секретар|заявник|скаржник|боржник|"
    r"стягувач|обвинувачений|обвинувачена|засуджений|засуджена|підозрюваний|підозрювана|потерпілий|потерпіла|"
    r"свідок|експерт|перекладач|спадкоємець|громадянин|громадянка|гр\.|фізична особа|особа|третя особа|"
    r"представник|захисник|ФОП|фізична особа-підприємець|фізична особа - підприємець|директор|керівник)[,:]?\s*$",
    re.I,
)


def guess_case(left: str) -> str | None:
    """Return a pymorphy case tag or None when the context does not tell."""
    ctx = left[-LEFT:]
    tail = ctx.rstrip()
    if not tail or tail.endswith((":", ";", "\n", ".", "—", "-", "–")):
        return "nomn"
    # "з ... " is ambiguous (gent vs ablt); ablt only after "разом з" / "між" handled first
    if re.search(r"\b(?:разом з|разом із|між|укладен\w+ між)\s*$", tail, re.I):
        return "ablt"
    if _GENT_ROLE.search(tail):
        return "gent"
    if _DATV.search(tail):
        return "datv"
    if _NOMN_ROLE.search(tail):
        return "nomn"
    if _GENT_PREP.search(tail):
        return "gent"
    if _ACCS.search(tail):
        return "accs"
    if _ABLT.search(tail):
        return "ablt"
    if tail.endswith(","):
        return None
    return None


# --- sub-type of НОМЕР -----------------------------------------------------------------------
_NUM_RULES = [
    ("rnokpp", re.compile(r"(рнокпп|іпн|ідентифікац|податков|обліков\w* картк|реєстраційн\w* номер\w* обл)", re.I)),
    ("iban", re.compile(r"(рахун|iban|р/р|картк\w* рахун|на карт)", re.I)),
    ("phone", re.compile(r"(тел\.|телефон|моб\.|мобільн|номер\w* тел|контактн)", re.I)),
    ("imei", re.compile(r"(imei|імеі)", re.I)),
    ("passport", re.compile(r"(паспорт|посвідч|id-?карт|ід-?карт)", re.I)),
    # «серії» right before the placeholder asks for letters + digits, whatever the document is
    ("svidotstvo", re.compile(r"(свідоцтв|диплом|трудов\w* книжк|атестат|посвідчення водія|водійськ|сері[яїю])", re.I)),
    ("military", re.compile(r"(частин|в/ч|в\.ч\.)", re.I)),
    ("vin", re.compile(r"(vin|він-?код|кузов|шасі|\bрам[иі]\b)", re.I)),
    ("vehicle", re.compile(r"(д\.?\s?н\.?\s?з|номерн\w* знак|реєстрац\w* номер|державн\w* номер|автомоб|транспортн|"
                            r"мотоцикл|причіп|заводськ)", re.I)),
    # «у ВП № НОМЕР_2», «АСВП НОМЕР_1»: the bare abbreviation right before the placeholder; «ЄДРПОУ ВП НОМЕР_1» is the code
    # of a company unit (відокремлений підрозділ), not a proceeding
    ("provadzhennia", re.compile(r"((?i:провадж)|(?<![\w/])(?<!ЄДРПОУ )(?:АС|З)?ВП(?=\s*(?:№|N)?\s*$))")),
    ("card", re.compile(r"(банківськ\w* карт|картк[аи] №|платіжн\w* карт)", re.I)),
]


def number_subtype(left: str, right: str) -> str:
    """The cue closest to the placeholder decides: «номер кузова …, реєстраційний номер НОМЕР_2» is a plate, not a VIN."""
    ctx = left[-LEFT:]
    best, pos = "generic", -1
    for name, rx in _NUM_RULES:
        for m in rx.finditer(ctx):
            if m.start() > pos:
                best, pos = name, m.start()
    return best


# --- sub-type of ІНФОРМАЦІЯ ------------------------------------------------------------------
# info_subtype() tries five stages in a fixed order: (A) the placeholder stands in quotes, (B) checks of the right
# context, (C) rules that must come before the general ones, (D) the general rules, (E) fallbacks for what is left.
# The rules come from the v6 audit (hand labels + four verifiers); each one cuts the window it was measured with,
# so giving info_subtype() more context than INFO_LEFT / INFO_RIGHT changes nothing.
# hints["mask"] is True where a rule is not reliable enough to teach a label: the fill step writes a value there
# but records it as a mask.
#   sub-type        value                                       hints
#   dob             birth date                                  child, no_roku, style ("numeric": «р.н.» follows)
#   death           date of death                               no_roku; weak + mask for the loose cue
#   derived         date computed from a child's birth date     until: "majority" / "age23" / "age3"; no_roku
#   date_event      a recent date, neither birth nor death      no_roku; always masked
#   email, url      e-mail / web address                        url_kind: "site" / "page" / "profile"; a link masked where the text
#                                                               names a video-conference system or already gave the link
#   nick            social-network account                      nick_kind: "handle" / "page"; always masked (0.61 on the hand labels)
#   ident           login or system identifier                  always masked
#   military_name   what follows «військова частина»
#   org             institution, declined by guess_org_case()   org_kind: "party" / "unit" / "short" / "holder" / "official" (masked)
#   quoted          bare name inside the quotes of the text     quote_kind: "company" / "bank" / "formation" / "institution" /
#                                                               "person" (masked) / "unknown" (masked)
#   place, other    settlement / most likely an institution     always masked
INFO_LEFT, INFO_RIGHT = 110, 70  # characters of context that info_subtype() is given
Subtype = tuple[str, dict]

# ---- A. inside the quotes of the text ----
# «ІНФОРМАЦІЯ_3», “ ІНФОРМАЦІЯ_3 ”: the opening quote right before the placeholder
_QUOTE_LEFT = re.compile(r"[«\"„“]\s*$")
# «ІНФОРМАЦІЯ_3», „ІНФОРМАЦІЯ_3“, “ ІНФОРМАЦІЯ_3 ”: the closing quote right after it
_QUOTE_RIGHT = re.compile(r"^\s*[»\"“”]")
# «ТОВ «X»»: a legal form right before the opening quote: the quotes hold a name, whatever cue stands further left
_Q_FORM = re.compile(r"(?<![а-яіїєa-z])(?:ТОВ|ТзОВ|ПрАТ|ПАТ|АТ|ГО|ДП|КП|КНП|ПП|ФГ|ДУ|БФ|ОСББ|товариств\w+|підприємств\w+|"
                     r"організаці\w+|банк\w*)\s*[«\"„“]\s*$", re.I)
# «зазначивши вірно дату народження боржника - ОСОБА_2, «X»»
_Q_DOB = re.compile(r"дат\w+\s+(?:\w+\s+)?народження(?:[^.;]|\b[А-ЯІЇЄ]\.){0,40}[«\"„“]\s*$", re.I)
# «на електронну адресу суду «X»»
_Q_EMAIL = re.compile(r"(?i:електрон\w*\s+(?:адрес\w*|пошт\w*)|e-?mail|ел\.\s*(?:адрес|пошт)\w*)[^«\"„“]{0,25}[«\"„“]\s*$")
# «рік народження з «X»» (not «місце народження «X»»)
_Q_DATE_BIRTH = re.compile(r"(?i:(?<!місце )(?<!місця )народження[^«\"„“]{0,30})[«\"„“]\s*$")
# «виправити дату «X»» («дат…» as a whole word: not «надати «X»»)
_Q_DATE_BARE = re.compile(r"(?i:(?<!\w)дат[аиую]\w{0,2}[\s:\-–—]{0,3})[«\"„“]\s*$")
# «змінити прізвище дитини з «X»», «по батькові «X»»
_Q_PERSON = re.compile(r"(?i:прізвищ\w+|по\s?-?\s?батькові)[^.;]{0,45}[«\"„“]\s*$")
# «оперативного командування «X»», «впс «X»» (the abbreviations are case-sensitive: «ОК» is not «ок»)
_Q_FORMATION = re.compile(r"(?<!\w)(?:(?i:командуванн\w+|угруп[оу]ванн\w+(?:\s+військ)?|прикордонн\w+\s+служб\w+|"
                          r"контрольно-пропускн\w+\s+пункт\w*)|ОТУ|ОСУВ|ОУВ|ОК|ВПС|впс|віпс|ВіПС)\s*[«\"„“]\s*$")
# «КНП «X»», «комунального закладу Львівської обласної ради «X»», «ДУ «X»»
_Q_INSTITUTION = re.compile(r"(?<!\w)(?:КНП|КЗ|КУ|ДУ|КЗПСО|КЗОЗ|(?i:некомерційн\w+\s+підприємств\w+|комунальн\w+\s+(?:заклад\w*|установ\w+)|"
                            r"державн\w+\s+установ\w+))(?:\s+[А-ЯІЇЄҐ]{2,5})?(?:\s+\w+\s+(?i:обласної|міської|селищної|сільської|районної)"
                            r"\s+ради)?\s*[«\"„“]\s*$")
# «ТОВ «X»», «Товариство з обмеженою відповідальністю «X»», «АТ КБ «X»», «магазину «X»»; group 1 is the cue
_Q_COMPANY = re.compile(r"(?<!\w)(ТОВ|ТзОВ|ТОB|TOB|ПрАТ|ПАТ|АТ|AT|ПП|ДП|КП|СГК|СВК|БФ|НПП|ФГ|СФГ|ТДВ|ВАТ|ЗАТ|ГО|ОСББ|ТЦ|ТРЦ|КБ|АКБ|АБ|АГ|"
                        r"(?i:товариств\w+|відповідальністю|(?<!некомерційного )(?<!некомерційне )підприємств\w+|банк\w*|магазин\w*|"
                        r"кафе(?:-бар\w*)?|компані\w+|готел\w+|клуб\w*|концерн\w*|корпораці\w+|господарств\w+|кооператив\w*|фірм\w+|"
                        r"супермаркет\w*|ресторан\w*|аптек\w+|торгов\w+\s+марк\w+))\s*[«\"„“]\s*$")
# «банку», «АТ КБ»: the company cue names a bank
_Q_BANK = re.compile(r"(?i:банк\w*)|КБ|АКБ|АБ")
# «на банківську картку АТ «X»», «рахунку НОМЕР_2, відкритому в АТ «X»», «у володінні банку Акціонерного товариства «X»»:
# a joint-stock company named after a card, an account or a bank word is the bank (45 of 45 read)
_Q_BANK_CONTEXT = re.compile(r"(?:банк\w*|карт[кц]\w+|карт[иіуа]|картков\w+|рахун\w+|емітован\w+|емітент\w*)[^.;«»\"]{0,60}?(?<!\w)"
                             r"(?:АТ|AT|ПАТ|ПрАТ|акціонерн\w+\s+товариств\w+)\s*[«\"„“]\s*$", re.I)
# the opening quote itself, removed before the general e-mail cue is tried on a quoted placeholder
_Q_OPEN = re.compile(r"[«\"„“]\s*$")

# ---- B. right-context checks ----
# «ІНФОРМАЦІЯ_1 р.н.»
_DOB_RIGHT_RN = re.compile(r"^\s*р\.\s?н\.", re.I)
# «X (військова частина НОМЕР_1)», «X - в/ч НОМЕР_2»: the unit's code stands in the open, X is the unit's own name
_UNIT_RIGHT = re.compile(r"^\s*(?:\(АДРЕСА_\d+\)\s*)?[(\-–—]\s*(?:військов\w+\s+частин\w+|в/ч|ВЧ)(?!\w)", re.I)
# «X Державної прикордонної служби України», «X прикордонного загону Державної …»: the parent body is in the open
_PARENT_RIGHT = re.compile(r"^\s*\)?\s*(?:Державної прикордонної служби|Національної гвардії|Збройних Сил|Міністерства оборони|"
                           r"(?:прикордонн\w+\s+)?заг[іо]н\w*\s+Державної)", re.I)
# «НОМЕР_1 прикордонного загону X (військова частина …)», «загін імені X»: the head noun is open, X is only the tail of the name
_UNIT_HEAD_LEFT = re.compile(r"(?<!\w)(?:заг[оі]н\w*|імені|ім\.)\s*$", re.I)

# ---- C. before the general rules ----
# «уродженця X», «місце народження: X», «мешканця X»
_PLACE = re.compile(r"(уродж\w*|місце народження[\s:\-–—]*|народи\w+\s+(?:в|у)|мешканц\w+)\s*$", re.I)
# «ОСОБА_1 (X, РНОКПП НОМЕР_1)»: the bracket after a person …
_REQ_BRACKET_LEFT = re.compile(r"ОСОБА_\d+\s*[(/]\s*$")
# … and a requisite after the date: «, РНОКПП», «р.н., паспорт», «, остання відома адреса», « ІПН НОМЕР_1»
_REQ_BRACKET_RIGHT = re.compile(r"^\s*(?:р\.\s?н\.|року народження)?\s*(?:[,;]\s*(?:РНОКПП|РНКОПП|ІПН|ІН\b|і\.?\s?н\.|ідентифікац|"
                                r"реєстраційний номер|паспорт|закордонний|адреса|місце|зареєстр|останн|прожива|жител|мешкан|уродж|"
                                r"громадян|як(?:ий|а)\s+помер|дата смерті)|\s+(?:РНОКПП|РНКОПП|ІПН)\b)", re.I)
# «ОСОБА_3, АДРЕСА_1, X, РНОКПП»: a list of requisites after a person (or the bracket again) …
_REQ_LIST_LEFT = re.compile(r"ОСОБА_\d+\s*(?:\(|(?:,\s*(?:Р\s?|РНОКПП:?\s*|ІПН:?\s*)?(?:АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_\d+)+\s*,)\s*$", re.I)
# … continued by another requisite: «, РНОКПП НОМЕР_2», «, зареєстрований», «) паспорт»
_REQ_LIST_RIGHT = re.compile(r"^\s*[,;)]?\s*(?:РНОКПП|ІПН|адрес|зареєстр|паспорт|уродж|прожива|місце|громадян|року народж|р\.\s?н|"
                             r"реєстраційн\w+\s+номер|ідентифікаційн|ід\.\s?(?:код|номер)|податков\w+\s+номер|мешка|житель|жител[яі]|"
                             r"НОМЕР_\d)", re.I)
# the age a derived date is tied to: «повноліття», «23 років», «трьох років», «трирічного віку», «закінчення навчання»
_DERIVED_NOUN = (r"(?:(?<!не)повнол\w+|(?:\d+|двадцяти\s+трьох|трьох|вісімнадцяти)\s*[-‑]?\s*(?:х|ти)?\s*(?:\([^)]{0,20}\)\s*)?рок\w+|"
                 r"річного\s+віку|закінчення\s+навчання)")
# «до досягнення дитиною повноліття, тобто до X», «до 23 років, а саме по X»
_DERIVED_THAT_IS = re.compile(_DERIVED_NOUN + r"(?:\s+[\w’'`]+){0,2}\s*,?\s*\(?\s*(?:тобто|а\s+саме)\s*,?\s*(?:до|по)?\s*$", re.I)
# «до повноліття - X», «трирічного віку – X» (the bare «річного віку X» was a birth date and is left out)
_DERIVED_DASH = re.compile(r"(?<!не)повнолітт[яю]\s*[-–—]?\s*$|річного\s+віку\s*[-–—]\s*$", re.I)
# «до 23 років», «двадцяти трьох років», «до закінчення навчання»
_UNTIL_AGE23 = re.compile(r"23|двадцяти\s+трьох|закінчення\s+навчання", re.I)
# «трирічного віку», «трьох років», «3-х років»
_UNTIL_AGE3 = re.compile(r"трирічн|(?<!двадцяти\s)трьох|(?<!\d)3\s*[-‑]?\s*(?:х\s*)?(?:\([^)]{0,20}\)\s*)?(?:рок|річн)", re.I)
# «повноліття», «18 років», «вісімнадцяти років»
_UNTIL_MAJORITY = re.compile(r"повнол|18|вісімнадцят", re.I)
# «який помер X», «після смерті (X», «дружині загиблого X»: the death cue right before the placeholder
_DEATH_ADJACENT = re.compile(r"(?:помер(?:ла|лий|лого|лої|лому|лій|лим|лою)?|загинув|загинула|загибл\w+|смерті|смерть|смертю)"
                             r"[\s,:\-–—(]*$", re.I)
# «що X ОСОБА_3 помер», «X у місті Києві померла ОСОБА_2» (not «X помер 16.01.2021»: there X is the birth date)
_DEATH_RIGHT = re.compile(r"^\s*(?:(?:у|в)\s+[^,.;()]{0,40}?\s+)?(?:(?:гр\.\s*)?ОСОБА_\d+\s+)?(?:помер(?:ла)?|загинув|загинула)(?![\w])"
                          r"(?!\s+(?:ІНФОРМАЦІЯ_\d|\d{1,2}[\s.]))", re.I)
# «ОСОБА_7, X помер …»: after a person and a comma X is the birth date in apposition
_PERSON_COMMA = re.compile(r"ОСОБА_\d+\s*,\s*$")
# «смерть настала X», «дата смерті - X», «по день смерті ОСОБА_2 X», «до дня загибелі (смерті) останнього - X»
_DEATH_LINK = re.compile(r"(?:смерт\w+|загибел\w+)[^.;()]{0,35}?(?:настала|сталася|стався|мала\s+місце|вважати|останнього|останньої|"
                         r"зазначена|тобто\s+(?:з|по|до)|а\s+саме)\s*[-–—:]?\s*$|(?:дат\w+|день|дня|днем)\s+смерті(?:\s+ОСОБА_\d+)?"
                         r"\s*[-–—:]?\s*$|(?:смерті|загибелі)\s*\((?:смерті|загибелі)\)\s+\w+\s*[-–—]\s*$", re.I)
# «народився», «народилась»: a birth verb near a death cue makes the date ambiguous
_BORN = re.compile(r"народи\w+", re.I)
# «X року народився син», «X у сторін народилася донька» (no comma between the placeholder and the verb)
_BORN_RIGHT = re.compile(r"^\s*(?:року|р\.)?\s*(?:у\s+(?:сторін|них|подружжя|шлюбі)\s+)?народи(?:вся|лася|лась|лися)(?!\w)", re.I)
# «(далі - відповідач, X)»: the short name of a party …
_ALIAS_LEFT = re.compile(r"\(\s*дал[іi]\s*(?:також\s*)?[-–—]?\s*[^()]{0,40}[,;]\s*$", re.I)
# … closed by the bracket
_ALIAS_RIGHT = re.compile(r"^\s*\)")
# «(ВКЗ) за обліковим записом: X», «ID користувачем – X», «платформи «Електронний суд» - X»: such an account is an e-mail address
_VC_ACCOUNT = re.compile(r"(?:обліков\w+\s+запис\w*|ID\s+користувач\w*|користувач\w*\s+систем\w+\s+[«\"]?EasyCon[»\"]?|"
                         r"(?:vkz\.court\.gov\.ua|ВКЗ|ВК3|EasyCon)[»\")]*[,\s]+користувач\w*|"
                         r"зареєстрован\w+\s+в\s+Єдиній\s+судовій\s+інформаційно-телекомунікаційній\s+системі)"
                         r"(?:[^.;]|\b[А-ЯІЇЄ]\.){0,30}$|(?:платформ\w+|підсистем\w+|сервіс\w+)\s+[«\"]Електронний суд[»\"]\s*[-–—:]\s*$", re.I)
# «X року народження», «X р. н.»
_ROKU_BIRTH = re.compile(r"^\s*(?:року|р\.)\s*(?:народження|нар\.|н\.)", re.I)
# «X року» without «народження»: «оприлюднену X року інформацію»
_ROKU_PLAIN = re.compile(r"^\s*року(?!\s*народж)(?![\w])", re.I)
# «ОСОБА_3, X року», «народився X року»: here the plain «року» still follows a birth date
_ROKU_DOB_LEFT = re.compile(r"(ОСОБА_\d+\s*,\s*$|народ)", re.I)
# «X року», «X р.»: the text already has the word, so the value must not repeat it
_ROKU_RIGHT = re.compile(r"^\s*(?:року(?!\w)|р\.)", re.I)

# ---- D. the general rules ----
# the role words of the birth-date cue must start a word: «відносин,» is not «син,»
_ROLE = (r"(?<![а-яіїєґ'’`])(?:позивач|відповідач|заявник|скаржник|засуджен|обвинувачен|потерпіл|підозрюван|боржник|стягувач|дитин|"
         r"син|доньк|дочк|громадян)\w*")
# «ОСОБА_3, X», «народився X», «дата народження X», «позивач, X»
_DOB = re.compile(rf"(ОСОБА_\d+,?\s*$|народ|р\.\s?н\.|дата народж|дн\.|д\.н\.|{_ROLE},\s*$)", re.I)
# «померла», «смерті», «загинув» anywhere in the window: right every second time only, so its date is masked
_DEATH_WEAK = re.compile(r"(помер|смерт|загин|відкрил\w+\s+спадщин|спадщин\w+,\s+яка відкрилась)", re.I)
# «адрес ІНФОРМАЦІЯ_4, ІНФОРМАЦІЯ_5, X»: the placeholders listed right before ours, skipped before the e-mail cue is tried
_MAIL_LIST_TAIL = re.compile(r"(?:(?:ІНФОРМАЦІЯ|НОМЕР|АДРЕСА)_\d+\s*[,;/]?\s*(?:та|і|й)?\s*)+$")
# «на електронну адресу X», «e-mail: X», «ел. пошта X», «пошта: a@b.com, X» (the cue must name a mailbox: not «поштового зв'язку»)
_EMAIL = re.compile(r"(?:[еэ]л[еэ]ктрон+\w*\s+(?:поштов\w+\s+)?(?:адрес\w*|пошт\w*|с?криньк\w*|синьк\w*|лист\w*)|(?<![а-яіїє])по\s+пошті|"
                    r"(?<![а-яіїє])е\.\s?пошт\w*|[еэ]л[еэ]ктрон+у\s*(?=\(\s*$)|(?<![а-яіїє])ел(?:ектр)?\.\s?(?:адрес\w*|пошт\w*)\.?|"
                    r"(?<![а-яіїє])е-пошт\w*|поштов\w+\s+с?криньк\w+|особист\w+\s+пошт\w+|адрес\w+\s+ел\w*\.?\s*пошт\w*|"
                    r"(?<![a-zа-яіїє])[eе]\s?-?\s?mail\w*|емейл\w*|імейл\w*|(?<![а-яіїє])пошта:?(?=\s*$))"
                    r"(?:[\w.+-]+@[\w-]+(?:\.[\w-]+)+|[^.;)]|\b[А-ЯІЇЄA-Z]\.){0,45}$", re.I)
# «X (електронна пошта ОСОБА_9)» (not «X (електронна пошта: ІНФОРМАЦІЯ_3)»: there the next placeholder is the address)
_EMAIL_RIGHT = re.compile(r"^\s*\(\s*(?:ел[еэ]ктрон+\w*\s+(?:пошт\w+|адрес\w+)|[eе]-?mail)(?![^)]{0,25}ІНФОРМАЦІЯ)", re.I)
# «на сайті X», «за посиланням X», «https://X» (not «з посиланням на те, що X», «Посилання скаржника X»)
_URL = re.compile(r"(сайт|https?|www\.|url|веб|інтернет|сторінк[аи] за адресою|"
                  r"(?<!з )(?<!із )(?<!зі )(?<![а-яіїє])посиланн(?:ями|ям|ю|і|я)(?![а-яіїє])"
                  r"(?!\s+(?:на\s+(?:те|відсутн|неправ|правов|лист|ст\.|п\.|норм|висновк|постанов|рішен|практик)|начальник|скаржник|"
                  r"відповідач|позивач|заявник|апелянт|суду)))", re.I)
# «у мережі Facebook за посиланням X», «на сторінці користувача X», «Telegram-каналі X»: the address of a profile
# («канал» as a word: not «телеканалу»)
_URL_PROFILE = re.compile(r"facebook|фейсбук|instagram|інстаграм|telegram|телеграм|tiktok|профіл|сторін[кц]\w+\s+користувач|"
                          r"(?<![а-яіїє])канал", re.I)
# «відео розміщене на youtube каналі за посиланням: X»: the link of a video, not of the channel
_URL_VIDEO = re.compile(r"(?:youtube|ютуб)[^.;]*(?:посиланн|адрес)|відео[^.;]*(?:youtube|ютуб)", re.I)
# «на сайті X», «веб-сайт X», «інтернет-ресурсі X», «інтернет ресурсах X»: the address of a site, not of one page
_URL_SITE = re.compile(r"сайт|веб-?\s?(?:адрес|сторін|ресурс|портал)|www\.|інтернет-?\s?ресурс", re.I)
# «Відеоконференція буде проведена за допомогою додатку «EasyCon», за веб-посиланням X», «системи відеоконференцзв'язку,
# розміщеної на сайті X»: the hand labels call it an account as often as a link, so the value is masked
_URL_VIDEO_CONFERENCE = re.compile(r"відео-?\s?конференц|відоконференц|(?<!\w)ВКЗ(?!\w)|EasyCon", re.I)
# «за посиланням: ІНФОРМАЦІЯ_3 «X», «посилання, а саме для ІНФОРМАЦІЯ_3 та для X»: the link is the placeholder before ours and
# other words follow it, so ours is a title or a name (not a list: «на ресурсах ІНФОРМАЦІЯ_5, ІНФОРМАЦІЯ_6, X»)
_URL_ALREADY_GIVEN = re.compile(r"ІНФОРМАЦІЯ_\d+[^А-Яа-яІіЇїЄєҐґA-Za-z]*?(?:(?:та|і|й|або|чи)\s+)?"
                                r"(?:(?!(?:та|або|чи)\s)[а-яіїєґ'’`]{2,}\s[^.;]*|[«\"„“]\s*)$")
# «нік X», «акаунт X», «користувач Telegram X», «профіль X», «каналі X» («нік» as a word: not «техніка», «клініка»)
_NICK = re.compile(r"(?<![а-яіїєґa-z])(?:нік(?:нейм\w*|ом|у|а|и|ами)?(?![а-яіїєґ])|nickname|ак{1,2}аунт\w*|"
                   r"користувач\w*\s+(?:телеграм\w*|вайбер\w*|«?telegram|«?viber|«?instagram|«?facebook|«?tiktok)|"
                   r"telegram|[«\"]телеграм[»\"]|instagram|facebook|фейсбук\w*|viber|tiktok|профіл\w*|канал\w*)", re.I)
# «на власній сторінці X», «Telegram-каналі X», «у групі X», «профіль X»: a page, not an account handle
_NICK_PAGE = re.compile(r"сторін[кц]|канал|профіл|(?<![а-яіїє])груп", re.I)
# «3 - відповідачу (office@firm.ua); 4 - третій особі (X)»: an e-mail cue further left and an opening bracket or dash …
_EMAIL_LIST_LEFT = re.compile(r"(?:@|елект|e-?mail|е-?mail)[^«»]*[(\-–—:;]\s*$", re.I)
# … and the item closed right after the placeholder: «X);», «X,», «X в електронній формі»
_EMAIL_LIST_RIGHT = re.compile(r"^\s*(?:[);,]|\n|в\s+електр|на\s+електр|$)", re.I)
# «командира військової частини X», «в/ч X» (not «чергової частини X», «запчастини X»)
_MILITARY_NAME = re.compile(r"(?<!\w)(?:військов\w+\s+частин\w*|в/ч|в\.\s?ч\.)\s*(?:№\s*)?$", re.I)
# «військової частини НОМЕР_1 (X», «в/ч А1234 X»: the unit's code in the open on the left, X is the unit's own name
_UNIT_LEFT = re.compile(r"(?<!\w)(?:військов\w+\s+частин\w+|в/ч|ВЧ)\s*(?:№\s*)?(?:НОМЕР_\d+|[А-ЯA-Z]?\d{4})\s*"
                        r"(?:\((?:ВЧ\s*|в/ч\s*)?НОМЕР_\d+\)\s*)?\(?\s*$", re.I)
# «відповідач: X», «відповідачі - ІНФОРМАЦІЯ_1, ІНФОРМАЦІЯ_2 та X», «співвідповідача X»
_RESPONDENT_LIST = re.compile(r"(?<!\w)(?:спів)?відповідач\w*[\s\-–—:,(|]*(?:ІНФОРМАЦІЯ_\d+\s*(?:,|та|і)\s*)*$", re.I)
# «(далі - X», «(далі також - відповідач, X»: the short name of the body hidden just before
_SHORT_NAME = re.compile(r"далі(?:\s+також)?\s*[-–—]\s*(?:відповідач[\s-]?\d?\s*,\s*)?$", re.I)
# «позовом ОСОБА_1 в інтересах неповнолітньої ОСОБА_2 до X»
_SUIT_TO = re.compile(r"(?<!\w)позов\w*[^.;\n]{0,60}?\s+до\s*$", re.I)
# a party to the case, cue right before the placeholder: «за позовом X», «зобов'язати X», «третя особа - X», «стягнути з X»;
# the clerks anonymise a military body (ministry, ТЦК, unit) or a company in this position
_ORG_STRICT = re.compile(r"(?<!\w)(позов\w*(?:\s+(?:ОСОБА|ІНФОРМАЦІЯ)_\d+\s*,?)*\s+до|за позовом|скарг\w+|заяв\w+|клопотанн\w+|відзив\w*|"
                         r"зобов[’'`\"]яза\w*|(?:протиправн\w+\s+)?(?:дії|діями|бездіяльн\w+)|рішенн\w+|в особі|на користь|стягн\w+ з|"
                         r"бюджетних асигнувань(?:\s+суб[’'`\"]єкта владних повноважень)?|(?:трет\w+|заінтересован\w+) ос[оі]б\w*[^,]{0,40}?|"
                         r"далі\s*[\-–—]|відповідач\w*[\s\-–—:]*|представник\w*|залучи\w+ до участі у справі|надіслан\w+|видан\w+|"
                         r"укладен\w+ між|повернут\w+|проти|щодо|стосовно|з боку|начальник\w*|командир\w*|керівник\w*|директор\w*)\s*$", re.I)
# a word of an institution anywhere in the window: «начальника відділу кадрів X», «тцк», «служби»
_ORG_LOOSE = re.compile(r"(начальник|командир|керівник|директор|голов[аи]\s|управлінн|відділ|департамент|тцк|служб|військов)", re.I)

# ---- E. fallbacks ----
# «електронна адреса представника позивача адвоката Коваля: X»: an e-mail cue up to 70 characters back
_EMAIL_CUE_FAR = re.compile(r"(?:електронн\w+\s+(?:адрес|пошт)\w*|\bел\.?\s*(?:адрес\w*|пошт\w*)|\b[eе]-?mail|\bел\.)[^.;\n]{0,70}$", re.I)
# «1 - позивачу (a@b.ua); 2 - відповідачу (X»: the previous item of a mailing list shows an address …
_EMAIL_AFTER_AT = re.compile(r"(?:[\w.]+@[\w.]+\w\)?\s*[;,]?\s*(?:\d\s*-\s*)?[^@;\n]{0,75}[-–—(:]\s*$)", re.I)
# … but not «- X (izmail@dpsu.gov.ua»: X is the name of the body, nor «(X, АДРЕСА_1)»: a person's details
_EMAIL_AFTER_AT_RIGHT = re.compile(r"^(?!\s*\()(?!\s*,\s*АДРЕСА_)", re.I)
# «2 - заявнику ОСОБА_1 (X»: an item of a list opened by a bracket, dash or colon …
_LIST_ITEM_LEFT = re.compile(r"[-–—(:]\s*$", re.I)
# … whose next item shows an address: «X); 3 - Товариству (lawyer@firm.ua)», also with a line break between the items
_EMAIL_BEFORE_AT = re.compile(r"^\)?\s*[;,]\s*[^@\n]{0,45}[-–—(:]\s*[\w.]+@|^\)?[ \t]*(?:[;,]|\n)\s*[^@;]{0,60}[-–—(:]\s*[\w.\-]+@", re.I)
# «Залізняка І.І. (X» …
_EMAIL_RIGHT_CUE_LEFT = re.compile(r"[(:\-–—]\s*$", re.I)
# … «X) у режимі відеоконференції», «X) на електронну адресу»
_EMAIL_RIGHT_CUE = re.compile(r"^\)?\s*(?:на електронну адресу|(?:у|в) режимі відеоконференц)", re.I)
# «підсистеми ЄСІТС (ідентифікатор X», «за номером - X», «ресурс: X»
_IDENT = re.compile(r"(?:ідентифікатор\w*(?:\s+[\w.]+)?|за номер\w+\s*[-–—]?|зареєстрованим ім.ям:|ресурс:)\s*\(?\s*$", re.I)
# «X помер її чоловік», «яка X у лікарні померла», «X - загинув»
_DEATH_VERB_RIGHT = re.compile(r"^\s*[-–—]?\s*(?:[^.,;()\n]{0,45}?\s)?(?:помер(?:ла|ли)?|загину(?:в|ла|ли))(?!\w)", re.I)
# «X у шлюбі народилася донька», «X, народився син» (no comma between the placeholder's clause and the verb)
_BIRTH_VERB_RIGHT = re.compile(r"^\s*,?\s*(?:[^.,;()\n]{0,45}?\s)?народи(?:вся|лася|лась|ла|лися)(?!\w)", re.I)
# «ОСОБА_1 (X», «ОСОБА_2 (ОСОБА_3), (X», «ОСОБА_1 / X»
_DOB_PERSON_BRACKET = re.compile(r"ОСОБА_\d+\s*(?:\(ОСОБА_\d+\)\s*)?,?\s*[(/]\s*$", re.I)
# «ОСОБА_10 (ОСОБА_11), X»: a second spelling of the name, then the birth date
_DOB_PERSON_ALIAS = re.compile(r"ОСОБА_\d+\s*\(ОСОБА_\d+\)\s*,\s*$", re.I)
# «ОСОБА_4 – X» …
_DOB_PERSON_DASH_LEFT = re.compile(r"ОСОБА_\d+\s*[-–—.]\s*$", re.I)
# … «X, уродженця смт …», «X та ОСОБА_4 – ІНФОРМАЦІЯ_2»
_DOB_PERSON_DASH_RIGHT = re.compile(r"^\s*(?:[,;]\s*(?:уродж|жител|громадян|прожива|зареєстр|як(?:ий|а) прожива|з усіх)|\s+у вигляді|"
                                    r",\s+як[іа] |\s+та ОСОБА_\d+\s*[-–—]\s*ІНФОРМАЦІЯ|,\s*для)", re.I)
# «X (код ЄДРПОУ 12345678)», «X (місцезнаходження: …)», «X, що знаходиться за адресою»
_ORG_RIGHT_EDRPOU = re.compile(r"^\s*[,(]?\s*\(?(?:код\s+(?:за\s+)?)?(?:ЄДРПОУ|ЄРДПОУ)|^\s*\((?:місцезнаходження|юридична адреса|адреса:|"
                               r"за адресою|військова частина|далі\s*[-–—])|^\s*,?\s*(?:за адресою|(?:який|яка|яке|що)\s+(?:розташован|"
                               r"знаходиться))|^\s*,?\s*місцезнаходження", re.I)
# «загін імені X (код ЄДРПОУ …)»: X is a person's name inside the name of a unit
_NAMED_AFTER_LEFT = re.compile(r"імені\s*$", re.I)
# «документи знаходяться у володінні X», «зберігаються в X», «працювала у X»: the holder of documents, a place of work or custody
_ORG_HOLDER = re.compile(r"(?:(?:у|в)\s+(?:володінні|провадженні|приміщенн[іяю]|користуванні)(?:\s+посадових осіб)?|(?:зберіга\w+|утриму\w+|"
                         r"перебува\w+|працювала?|гауптвахт\w*|знаходяться)\s+(?:у|в)|гауп\w*вахт\w+|які належать)\s*$", re.I)
# «X про визнання протиправною», «X, ІНФОРМАЦІЯ_2, третя особа», «X звернулося до суду», «X із заявою»
_ORG_PARTY_RIGHT = re.compile(r"^\s*(?:,?\s*(?:ІНФОРМАЦІЯ_\d+|військов\w+ частин\w+ НОМЕР_\d+|в/ч НОМЕР_\d+)\s*,?\s*){0,3},?\s*(?:[-–—]\s*)?"
                              r"(?:про\s+(?:визнання|встановлення|стягнення|зобов|притягнення|скасування|залишення)|трет[яі] особ|"
                              r"(?:звернувся|звернулося|звернулась|подав|подало|подала|оскаржив|оскаржило|зазначає|повідомив|встановили)(?!\w)|"
                              r"(?:із|з)\s+(?:відповідною\s+)?заявою)", re.I)
# «заінтересована особа - X», «позовом ОСОБА_1 до ІНФОРМАЦІЯ_1, X», «Міністерства оборони України, X»
_ORG_PARTY_LEFT = re.compile(r"(?:заінтересован\w+\s+ос[оі]б\w*(?:\s+ОСОБА_\d+\s+та)?\s*[-–—:]?|(?:позов\w*|скарг\w+)\s+ОСОБА_\d+[^.;\n]{0,25}"
                             r"\sдо\s+(?:[^.;\n]{2,70},\s*)|Міністерств\w+ оборони України(?:,|\s+та)|(?:із|з) рішенням[^.,;\n]{0,40},)\s*$", re.I)
# «довідки X», «наказом X», «ухвали X»: the author of a document …
_ORG_DOC_AUTHOR_LEFT = re.compile(r"(?<!\w)(?:довідк\w+|лист[ауі]?|наказ\w*|ухвал\w+|повідомлення|постанов\w+|рішень|відповід[іь]\s+з|"
                                  r"складен\w+|надан\w+|що видав|видан\w+(?:\s+КНП)?|про розмір грошового забезпечення)\s*$", re.I)
# … followed by the document's date or number: «X від 05.07.2023», «X № 123», «X,»
_ORG_DOC_AUTHOR_RIGHT = re.compile(r"^\s*(?:(?:від\s+\d|№|N\s?\d|справа\s+№|нової довідки|довідки|до Головного)|[;,.]|\s+станом на)", re.I)
# «звернувся до X», «надійшло з X», «лист з X», «на адресу X», «через X» (not «інформація від X»: a date as often as a sender)
_ORG_ADDRESSEE = re.compile(r"(?<!\w)(?:(?:зверн\w+|зверта\w+|прибув|доставлен\w+|повернен\w+|повернути|направ\w+|передан\w+|подан\w+ її|"
                            r"доводить його|запит\w*|вилучено)\s+до|(?:надійш\w+|надісла\w+|витребува\w+)\s+(?:і?з|від|зі)|"
                            r"(?:інформаці[а-яіїє]+|лист)\s+(?:і?з|зі)|на адресу|від тимчасової|через|експерт\w+|органами|органів|"
                            r"підпорядкованих|між|призваний)\s*$", re.I)
# «слідчого судді X», «прокурора X», «на картку X»: the body of an official or a bank …
_ORG_OFFICIAL_LEFT = re.compile(r"(?<!\w)(?:слідч\w+(?:\s+судд\w+)?|прокурор\w*|судд\w+|працівник\w+|співробітник\w+|заступник\w*|виконавц\w+|"
                                r"реєстратор\w+|чергового|командувач\w*|банк\w*|картк\w+)\s*$", re.I)
# … followed by the official's name or the end of the phrase: «X ОСОБА_3», «X НОМЕР_1», «X.»
_ORG_OFFICIAL_RIGHT = re.compile(r"^\s*(?:ОСОБА_\d+|НОМЕР_\d+|№|,|\.|\d|від\s|у складі|генерал|полковник|встановили|надано)", re.I)

# ---- the «child» hint of a birth date ----
# «малолітньої ОСОБА_3, X», «сина ОСОБА_2, X», «дітей: ОСОБА_4, X»
_CHILD = re.compile(r"(малолітн|(?<![а-яіїє])неповнолітн|дитин|дитя|\bдітей\b|\bдіти\b|\bдітьми\b|\bсин[ау]?\b|\bсин(?:ом|ові)\b|доньк|дочк|"
                    r"\bдочок\b|\bдоньок\b|онук[уи]?\b|онучк|трирічного)", re.I)
# «дітей: ОСОБА_3, ІНФОРМАЦІЯ_1, та ОСОБА_4, »: only names and dates between a child cue further left and the placeholder
_CHILD_LIST_TAIL = re.compile(r"\w*[\s:,\-–—]*(?:(?:ОСОБА_\d+|ІНФОРМАЦІЯ_\d+|та|і|й)[\s,;:()\-–—]*)+", re.I)
# «повнолітнього сина ОСОБА_2, X»: an adult child
_ADULT = re.compile(r"(?<!не)повнолітн", re.I)
# «є донькою ОСОБА_2, X», «доводиться сином ОСОБА_1, X»: X is the parent's date
_KIN_PREDICATE = re.compile(r"(?:\bє|являється|доводиться|\bбула|\bбув|рідн\w+)\s+(?:\w+\s+){0,2}?(?:сином|донькою|дочкою|доньками|онуко[мю]|"
                            r"онучкою)\s+(?:\w+\s+)?ОСОБА_\d+,?\s*$", re.I)
# «матір'ю ОСОБА_1, X», «від ОСОБА_2, X», «стягнути з ОСОБА_1, X»: the parent, the payer
_PARENT = re.compile(r"(?:матір\w*|матер\w+|\bмати|батьк\w+|\bвід|усиновлювач\w*|опікун\w*|бабус\w+|дідус\w+|\bз|стягн\w+ з)\s*[-–—:]?\s*"
                     r"ОСОБА_\d+,?\s*$", re.I)

_DATE_SUBTYPES = frozenset({"dob", "death", "derived", "date_event"})
# sub-types that are never reliable enough for a label
_ALWAYS_MASKED = frozenset({"date_event", "ident", "nick", "place", "other"})
_QUOTE_MASKED = frozenset({"person", "unknown"})


def inside_quotes(left: str, right: str) -> bool:
    """The placeholder stands inside the quotes of the text: «ІНФОРМАЦІЯ_3», “ ІНФОРМАЦІЯ_3 ”."""
    return bool(_QUOTE_LEFT.search(left) and _QUOTE_RIGHT.search(right))


def bank_in_quotes(left: str) -> bool:
    """The quotes that open at the end of `left` hold the name of a bank: «на банківську картку АТ «», «банку «», «АТ КБ «»."""
    company = _Q_COMPANY.search(left[-60:])
    return bool(company and (_Q_BANK.fullmatch(company.group(1)) or _Q_BANK_CONTEXT.search(left[-90:])))


def _is_child(left: str) -> bool:
    """The birth date belongs to a minor: a child cue within 45 characters, or further left with only a list of names
    and dates in between; not an adult child, a parent or a payer."""
    ctx = left[-110:]
    if _ADULT.search(ctx[-60:]) or _KIN_PREDICATE.search(ctx) or _PARENT.search(ctx):
        return False
    cues = list(_CHILD.finditer(ctx))
    if not cues:
        return False
    last = cues[-1]
    return len(ctx) - last.start() <= 45 or bool(_CHILD_LIST_TAIL.fullmatch(ctx[last.end():]))


def _until(span: str) -> str:
    """Which age a derived date is tied to: the cue closest to the placeholder decides."""
    best, pos = "majority", -1
    for name, rx in (("majority", _UNTIL_MAJORITY), ("age23", _UNTIL_AGE23), ("age3", _UNTIL_AGE3)):
        for m in rx.finditer(span):
            if m.start() > pos:
                best, pos = name, m.start()
    return best


def _quoted(left: str, right: str) -> Subtype | None:
    """A. The placeholder stands inside the quotes of the text: a date, an e-mail or a bare name of some kind."""
    if not inside_quotes(left, right):
        return None
    l60 = left[-60:]
    form = _Q_FORM.search(l60)  # after a legal form the quotes hold a name, whatever stands further left
    if not form and _Q_DOB.search(l60):
        return "dob", {}
    if not form and (_Q_EMAIL.search(l60) or _EMAIL.search(_Q_OPEN.sub("", _MAIL_LIST_TAIL.sub("", l60)))):
        return "email", {}
    if _Q_DATE_BIRTH.search(l60):
        return "dob", {}
    if _Q_DATE_BARE.search(l60):
        return "date_event", {}
    if _Q_PERSON.search(left[-70:]):
        return "quoted", {"quote_kind": "person"}
    if _Q_FORMATION.search(left[-45:]):
        return "quoted", {"quote_kind": "formation"}
    if _Q_INSTITUTION.search(left[-80:]):
        return "quoted", {"quote_kind": "institution"}
    if _Q_COMPANY.search(l60):
        return "quoted", {"quote_kind": "bank" if bank_in_quotes(left) else "company"}
    return "quoted", {"quote_kind": "unknown"}


def _right_context(left: str, right: str) -> Subtype | None:
    """B. What follows the placeholder decides, whatever stands before it."""
    if _DOB_RIGHT_RN.search(right[:25]):
        return "dob", {"style": "numeric"}
    if _UNIT_RIGHT.search(right[:40]) or _PARENT_RIGHT.search(right[:45]):
        return "org", {"org_kind": "unit"}
    return None


def _pre_rules(left: str, right: str) -> Subtype | None:
    """C. Rules that must be tried before the general ones: a cue next to the placeholder beats a cue further left."""
    ctx = left[-LEFT:]
    if _PLACE.search(ctx):
        return "place", {}
    if (_REQ_BRACKET_LEFT.search(left[-20:]) and _REQ_BRACKET_RIGHT.search(right[:45])) or (
            _REQ_LIST_LEFT.search(left[-60:]) and _REQ_LIST_RIGHT.search(right[:30])):
        return "dob", {}
    l90 = left[-90:]
    derived = _DERIVED_THAT_IS.search(l90) or _DERIVED_DASH.search(l90)
    if derived:
        return "derived", {"until": _until(l90[max(0, derived.start() - 12):])}  # 12: «двадцяти три» before «річного віку»
    if _DEATH_ADJACENT.search(left[-30:]):
        return "death", {}
    if _DEATH_RIGHT.search(right[:60]) and not _PERSON_COMMA.search(left[-15:]):
        return "death", {}
    if _DEATH_LINK.search(left[-60:]) and not _BORN.search(left[-60:]):
        return "death", {}
    if _BORN_RIGHT.search(right[:40]):
        return "dob", {"child": True}  # «X народився син»: the date of a child of the parties
    if _ALIAS_LEFT.search(ctx) and _ALIAS_RIGHT.search(right[:40]):
        return "org", {"org_kind": "short"}
    if _VC_ACCOUNT.search(left[-60:]):
        return "email", {}
    if _ROKU_BIRTH.search(right[:25]):
        return "dob", {}
    if _ROKU_PLAIN.search(right[:20]):
        # plain «року» beats the org / url / nick cues («розповсюджену відповідачем X року»); it keeps a birth date only
        # after «ОСОБА_n,» or «народ…»
        return ("dob", {}) if _ROKU_DOB_LEFT.search(ctx) else ("date_event", {})
    return None


def _url(left: str) -> Subtype:
    """A web address: of a profile, of a site or of one page; masked where the context leaves its kind in doubt."""
    ctx = left[-LEFT:]
    last = list(_URL.finditer(ctx))[-1]
    if _URL_PROFILE.search(left[-60:]) and not _URL_VIDEO.search(left[-60:]):
        kind = "profile"
    else:
        # the cue closest to the placeholder with the words after it: «інтернет-ресурс », not «сайті суду за посиланням: »
        kind = "site" if _URL_SITE.match(ctx, last.start()) else "page"
    hints = {"url_kind": kind}
    if _URL_VIDEO_CONFERENCE.search(left) or _URL_ALREADY_GIVEN.search(ctx[last.end():]):
        hints["mask"] = True
    return "url", hints


def _general_rules(left: str, right: str) -> Subtype | None:
    """D. The general cues, most of them searched anywhere in the last 45 characters; the first that matches wins."""
    ctx = left[-LEFT:]
    if _DOB.search(ctx):
        return "dob", {}
    if _DEATH_WEAK.search(ctx):
        return "death", {"weak": True, "mask": True}
    if _EMAIL.search(_MAIL_LIST_TAIL.sub("", left[-60:])) or _EMAIL_RIGHT.search(right[:40]):
        return "email", {}
    if _URL.search(ctx):
        return _url(left)
    if _NICK.search(ctx):
        return "nick", {"nick_kind": "page" if _NICK_PAGE.search(ctx) else "handle"}
    if _EMAIL_LIST_LEFT.search(left[-110:]) and _EMAIL_LIST_RIGHT.search(right[:20]):
        return "email", {}
    if _MILITARY_NAME.search(left[-30:]):
        return "military_name", {}
    if _UNIT_LEFT.search(left[-60:]):
        return "org", {"org_kind": "unit"}
    if _RESPONDENT_LIST.search(left[-60:]):
        return "org", {"org_kind": "party"}
    if _SHORT_NAME.search(left[-40:]):
        return "org", {"org_kind": "short"}
    if _SUIT_TO.search(left[-80:]) or _ORG_STRICT.search(ctx) or _ORG_LOOSE.search(ctx):
        return "org", {"org_kind": "party"}
    return None


def _fallbacks(left: str, right: str) -> Subtype | None:
    """E. Tried only when nothing above matched: wider windows and right-hand cues, in the order they were measured."""
    if _EMAIL_CUE_FAR.search(left[-110:]):
        return "email", {}
    if _EMAIL_AFTER_AT.search(left[-110:]) and _EMAIL_AFTER_AT_RIGHT.search(right[:20]):
        return "email", {}
    if _LIST_ITEM_LEFT.search(left[-20:]) and _EMAIL_BEFORE_AT.search(right[:70]):
        return "email", {}
    if _EMAIL_RIGHT_CUE_LEFT.search(left[-5:]) and _EMAIL_RIGHT_CUE.search(right[:40]):
        return "email", {}
    if _IDENT.search(left[-45:]):
        return "ident", {}
    if _DEATH_VERB_RIGHT.search(right[:70]):
        return "death", {}
    if _BIRTH_VERB_RIGHT.search(right[:70]):
        return "dob", {}
    if _DOB_PERSON_BRACKET.search(left[-45:]) or _DOB_PERSON_ALIAS.search(left[-45:]) or (
            _DOB_PERSON_DASH_LEFT.search(left[-45:]) and _DOB_PERSON_DASH_RIGHT.search(right[:40])):
        return "dob", {}
    if _ORG_RIGHT_EDRPOU.search(right[:40]) and not _NAMED_AFTER_LEFT.search(left[-10:]):
        return "org", {"org_kind": "party"}
    if _ORG_HOLDER.search(left[-45:]):
        return "org", {"org_kind": "holder"}
    if (_ORG_PARTY_RIGHT.search(right[:70]) or _ORG_PARTY_LEFT.search(left[-110:])
            or (_ORG_DOC_AUTHOR_LEFT.search(left[-45:]) and _ORG_DOC_AUTHOR_RIGHT.search(right[:40]))
            or _ORG_ADDRESSEE.search(left[-45:])):
        return "org", {"org_kind": "party"}
    if _ORG_OFFICIAL_LEFT.search(left[-45:]) and _ORG_OFFICIAL_RIGHT.search(right[:40]):
        return "org", {"org_kind": "official"}
    return None


def _with_hints(st: str, hints: dict, left: str, right: str) -> Subtype:
    """The hints that do not depend on the rule that decided, and the mask of the sub-types that teach nothing reliable."""
    hints = dict(hints)
    if st in _DATE_SUBTYPES and _ROKU_RIGHT.search(right[:25]):
        hints["no_roku"] = True
    if st == "dob" and _is_child(left):
        hints["child"] = True
    # F. «прикордонного загону X (військова частина НОМЕР_2)»: a whole unit name would repeat the head noun
    unit_tail = st == "org" and hints.get("org_kind") == "unit" and _UNIT_HEAD_LEFT.search(left[-20:])
    if (st in _ALWAYS_MASKED or unit_tail or hints.get("org_kind") == "official" or hints.get("quote_kind") in _QUOTE_MASKED):
        hints["mask"] = True
    return st, hints


def info_subtype(left: str, right: str) -> Subtype:
    """Sub-type of one occurrence and the hints for its value (the table above the rules lists them)."""
    left, right = left[-INFO_LEFT:], right[:INFO_RIGHT]
    for stage in (_quoted, _right_context, _pre_rules, _general_rules, _fallbacks):
        found = stage(left, right)
        if found:
            return _with_hints(*found, left, right)
    return "other", {"mask": True}


def _vote(votes: list[Subtype]) -> Subtype:
    """The most frequent sub-type, ties to the one seen last; the hints of its last occurrence."""
    counts = Counter(st for st, _ in votes)
    top = max(counts.values())
    winner = next(st for st, _ in reversed(votes) if counts[st] == top)
    return next((st, dict(hints)) for st, hints in reversed(votes) if st == winner)


def choose_subtype(votes: list[Subtype]) -> Subtype:
    """Sub-type of a placeholder index from the sub-types of its occurrences: a majority vote over the occurrences
    that are neither «other» nor masked; only when there is none, the same vote over the masked ones."""
    known = [v for v in votes if v[0] != "other"]
    labelled = [v for v in known if not v[1].get("mask")]
    if not known:
        return "other", {"mask": True}
    return _vote(labelled or known)


def info_subtypes(doc: dict) -> dict[int, Subtype]:
    """Sub-type of every ІНФОРМАЦІЯ placeholder index of a decision, from all its occurrences."""
    text, votes = doc["text"], {}
    for p in sorted(doc["placeholders"], key=lambda p: p["start"]):
        if p["type"] == "ІНФОРМАЦІЯ":
            votes.setdefault(p["idx"], []).append(
                info_subtype(text[max(0, p["start"] - INFO_LEFT): p["start"]], text[p["end"]: p["end"] + INFO_RIGHT]))
    return {idx: choose_subtype(v) for idx, v in votes.items()}


# --- gender hints -----------------------------------------------------------------------------
_FEMN = re.compile(r"\b(?:донька|доньки|дочка|дочки|дружина|дружини|дружиною|мати|матері|матір'ю|сестра|сестри|"
                   r"бабуся|бабусі|онука|онуки|громадянка|громадянки|позивачка|позивачки|відповідачка|відповідачки|"
                   r"потерпіла|потерпілої|обвинувачена|обвинуваченої|засуджена|засудженої|підозрювана|підозрюваної|"
                   r"заявниця|заявниці|скаржниця|померла|померлої|неповнолітня|малолітня|малолітньої|неповнолітньої|"
                   r"вдова|вдови|спадкоємиця|жінка|жінки|пані|гр-ка|громадянці)[,:]?\s*$", re.I)
_MASC = re.compile(r"\b(?:син|сина|синові|чоловік|чоловіка|батько|батька|брат|брата|дідусь|онук|онука|громадянин|"
                   r"громадянина|позивач|позивача|відповідач|відповідача|потерпілий|потерпілого|обвинувачений|"
                   r"обвинуваченого|засуджений|засудженого|підозрюваний|підозрюваного|заявник|заявника|скаржник|"
                   r"померлий|померлого|неповнолітній|малолітній|малолітнього|неповнолітнього|вдівець|спадкоємець|"
                   r"чоловік|пан|гр-н|громадянину|військовослужбовець|військовослужбовця)[,:]?\s*$", re.I)


# --- case of an institution mention ---------------------------------------------------------
# «ІНФОРМАЦІЯ_1, », «ІНФОРМАЦІЯ_1 (військова частина НОМЕР_1) та »: the parties listed before ours share its case
_LIST_TAIL = re.compile(r"(?:(?:ОСОБА|ІНФОРМАЦІЯ|НОМЕР)_\d+(?:\s*\([^()]{0,60}\))?\s*(?:,|\s(?:та|і)(?!\w))\s*)+$")
# «Державної прикордонної служби України, »: a party named in the open before ours in a list
_OPEN_NAME_ITEM = re.compile(r"(?!ОСОБА_|ІНФОРМАЦІЯ_|НОМЕР_|АДРЕСА_)[А-ЯІЇЄҐ][^,;:.()\n]{2,80}(?:\s*\([^()]{0,60}\))?\s*,")
# «… України в », «… служби у »: a preposition of place inside an open name, not a cue before it
_INSIDE_NAME = re.compile(r"(?<!\w)(?:в|у)$", re.I)
# a capital letter that starts a word: where an open name may begin
_NAME_START = re.compile(r"(?<![\w«\"„])[А-ЯІЇЄҐ]")
# every cue is a whole word: «скаргу» must not match the «у» of the accusative cues
_ORG_GENT = re.compile(r"(?<!\w)(?:скарг\w+|заяв\w+|клопотанн\w+|відзив\w*|дії|діями|бездіяльн\w+|рішенн\w+|асигнувань|представник\w*|"
                       r"в особі|з боку|трет\w+ особ\w*|начальник\w*|командир\w*|керівник\w*|директор\w*|відповідач[аи]|позивач[аи]|"
                       r"відповід\w+|лист\w*|довідк\w+|наказ\w+|розпорядженн\w+|постанов\w+|ухвал\w+|"
                       r"щодо|стосовно|проти|від|для|з|із|зі|до|на користь|стягн\w+ з)\s*$", re.I)
_ORG_ACCS = re.compile(r"(?<!\w)(?:зобов[’'`\"]яза\w*|залучи\w+ до участі у справі|визнати|через|на|про|в|у)\s*$", re.I)
_ORG_DATV = re.compile(r"(?<!\w)(?:повернут\w+|надати|надано|направити|надіслати|надіслано|повідомити|відмовити|відмовлено|"
                       r"роз[’'`\"]яснити|вручити|належить|продовжено|продовжити)\s*$", re.I)
_ORG_ABLT = re.compile(r"(?<!\w)(?:надіслан\w+|видан\w+|укладен\w+ між|між|підписан\w+|подан\w+|разом з|разом із)\s*$", re.I)
# «комісії в », «службу у », «комісією при »: a place after a noun (group 1) is in the locative; «при» always is
_ORG_LOCT = re.compile(r"(?<!\w)(?:([\w’'`]+)\s+(?:в|у)|при)\s*$", re.I)
# «взводу охорони », «служби «ІНФОРМАЦІЯ_1» »: the word the institution's name is attached to (group 1)
_HEAD_WORD = re.compile(r"(?<![\w’'`])([А-Яа-яІіЇїЄєҐґ’'`]{3,})(?:\s+«[^«»]{1,40}»)?$")
# «військової частини НОМЕР_1 », «військовій частині НОМЕР_1 (НОМЕР_3) »: the unit's own name agrees with «частина»
# (group 1: its ending)
_UNIT_CODE_HEAD = re.compile(r"(?<!\w)частин(а|и|і|у|ою)\s+(?:НОМЕР_\d+|[А-ЯA-Z]?\d{4})(?:\s*\(НОМЕР_\d+\))?$", re.I)
_UNIT_ENDING_CASE = {"а": "nomn", "и": "gent", "і": "datv", "у": "accs", "ою": "ablt"}
# after these words the slot starts a new phrase or holds a name in the nominative: «06 лютого 2024 року X подано»,
# «представника компанії X», «щодо предмета спору X про визнання», «звернулась до суду X»
_NOT_A_HEAD = re.compile(r"(?:року|році|рік|спору|суду|компані\w+|товариств\w+|підприємств\w+|фірм\w+|банк\w*)", re.I)
# «заінтересовані особи Міністерство оборони України, X», «заінтересовану особу X»: the role word names the slot itself
# (apposition), and «особи» is as often a nominative plural as a genitive
_ROLE_APPOSITION = re.compile(r"(?<!\w)(?:заінтересован\w+|трет\w+)\s+ос[оі]б\w*$", re.I)
_NOUN_LIKE = frozenset({"NOUN", "ADJF"})


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3  # loaded on first use: the sub-typing rules above do not need it

    return pymorphy3.MorphAnalyzer(lang="uk")


@lru_cache(maxsize=20_000)
def _is_noun(word: str, adjective_too: bool = False) -> bool:
    """Every reading of the word is a noun (or an adjective): «охорони», «комісії», not «що», «та», «дій», «видано»."""
    parses = _morph().parse(word.lower().replace("`", "'").replace("’", "'"))
    kinds = _NOUN_LIKE if adjective_too else {"NOUN"}
    # a passive participle («наданої X довідки») takes its agent in the instrumental, so it is not a head
    return bool(parses) and all(p.tag.POS in kinds and "pssv" not in p.tag for p in parses)


def _org_cue_case(tail: str) -> str | None:
    """Case asked for by the cue right before the slot; None when there is no cue."""
    if _ORG_ABLT.search(tail):
        return "ablt"
    if _ORG_DATV.search(tail):
        return "datv"
    place = _ORG_LOCT.search(tail)
    if place and (place.group(1) is None or _is_noun(place.group(1))):
        return "loct"
    if _ORG_ACCS.search(tail):
        return "accs"
    if _ORG_GENT.search(tail):
        return "gent"
    return guess_case(tail)


def _open_list_case(tail: str) -> str | None:
    """«позовом ОСОБА_1 до Державної прикордонної служби України, X»: the cue stands before the open names of the list."""
    if not tail.endswith(","):
        return None
    for start in reversed([m.start() for m in _NAME_START.finditer(tail)]):
        before = _LIST_TAIL.sub("", tail[:start].rstrip()).rstrip()
        # «… фонду України в Житомирській області, X»: «в» belongs to the open name, which starts further left
        if _OPEN_NAME_ITEM.fullmatch(tail, start) and not _INSIDE_NAME.search(before) and not _ROLE_APPOSITION.search(before):
            case = _org_cue_case(before) or _open_list_case(before)  # another open name may stand before this one
            if case and case != "nomn":
                return case
    return None


def _head_case(tail: str) -> str | None:
    """No cue: a name right after a noun is its attribute in the genitive («взводу охорони X», «комісії X»)."""
    head = _HEAD_WORD.search(tail)
    if not head or _ROLE_APPOSITION.search(tail) or _NOT_A_HEAD.fullmatch(head.group(1)):
        return None
    return "gent" if _is_noun(head.group(1), adjective_too=True) else None


def guess_org_case(left: str) -> str:
    """Case of an institution: «позовом до ІНФОРМАЦІЯ_1, X» -> genitive (the list before X is skipped),
    «зобов'язати X» -> accusative, «повернути X» -> dative, «виданої X» -> instrumental, «комісії в X» -> locative,
    «взводу охорони X» -> genitive, «військовій частині НОМЕР_1 X» -> the case of «частина», a subject -> nominative."""
    before = left[-(LEFT + 40):].rstrip()
    tail = _LIST_TAIL.sub("", before).rstrip()
    unit = _UNIT_CODE_HEAD.search(before)
    if unit:
        return _UNIT_ENDING_CASE[unit.group(1).lower()]
    if not tail or tail.endswith((":", ";", ".", "—", "-", "–", ")", "\n")):
        return "nomn"
    # a noun is the head of our name only when nothing stands between them: not «Житлова комісія ІНФОРМАЦІЯ_1, X»
    return _org_cue_case(tail) or _open_list_case(before) or (_head_case(tail) if tail == before else None) or "nomn"


def guess_gender(left: str) -> str | None:
    tail = left[-LEFT:].rstrip()
    if _FEMN.search(tail):
        return "femn"
    if _MASC.search(tail):
        return "masc"
    return None
