"""All regular expressions used by the labelling layers."""
from __future__ import annotations

import regex as re

UC = r"[А-ЯІЇЄҐ]"
LC = r"[а-яіїєґ'’ʼ\-]"
# Прізвище І.І. | Прізвище І. І. | Прізвище І.
INI = rf"{UC}{LC}+(?:[ \t]*{UC}[ \t]?\.){{1,2}}"
# І.І. Прізвище | І. І. Прізвище
INI_R = rf"(?:{UC}[ \t]?\.[ \t]?){{1,2}}[ \t]*{UC}{LC}+"
PATR_SUFFIX = r"(?:(?:ович|йович|ьович|евич|євич|ич|іч)(?:а|у|ем|ом|і|е|ю)?|(?:івн|ївн|ічн|овн)(?:а|и|і|у|ою))"
SP = r"[ \t]+"
FULL = rf"{UC}{LC}{{2,}}{SP}{UC}{LC}{{2,}}{SP}{UC}{LC}+{PATR_SUFFIX}(?![а-яіїєґ])"
CAPS = rf"(?:{UC}{LC}{{2,}}\s+{UC}{{3,}}(?:-{UC}{{3,}})?|{UC}{{3,}}\s+{UC}{LC}{{2,}}\s+{UC}{LC}+{PATR_SUFFIX}[а-яіїєґ]*)"
NAME = rf"(?:{FULL}|{CAPS}|{INI}|{INI_R})"

ROLE = (
    r"(?i:судд\w*|головуюч\w*|секретар\w*|прокурор\w*|адвокат\w*|захисник\w*|представник\w*|нотаріус\w*|"
    r"(?:державн\w+|приватн\w+)\s+виконав\w+|виконав\w+|слідч\w*|експерт\w*|перекладач\w*|спеціаліст\w*|ФОП|"
    r"директор\w*|керівник\w*|голов[аиуою](?!\s+суду)|заявник\w*|скаржник\w*|позивач\w*|відповідач\w*|свідок|"
    r"свідк\w*|потерпіл\w+|обвинувачен\w+|підозрюван\w+|засуджен\w+|боржник\w*|стягувач\w*|апелянт\w*|"
    r"арбітражн\w+\s+керуюч\w+|ліквідатор\w*|доповідач\w*|помічник\w*|інспектор\w*|лікар\w*|опікун\w*|"
    r"громадянин\w*|громадянк\w*|гр\.|гр-н|гр-ка|особ[аиу]|фізичн\w+\s+особ\w+)"
)
ROLE_NAME = re.compile(rf"\b{ROLE}\s+(?:{LC}+\s+){{0,3}}({NAME})", re.U)
IN_PERSON_OF = re.compile(rf"\b(?i:в\s+особі)\s+(?:{LC}+\s+){{0,3}}({NAME})", re.U)
HEADER = re.compile(rf"(?i:у\s+складі|складі\s+колегії|суддів?|головуючого(?:\s+судді)?)\s*:?\s*({NAME}(?:\s*,\s*{NAME})*)", re.U)
SIGNATURE = re.compile(rf"^\s*(?:{ROLE}\s+)?({NAME})\s*$", re.U | re.M)
FULL_ANY = re.compile(FULL, re.U)
NAME_ANY = re.compile(NAME, re.U)

# contexts where a capitalised sequence is NOT a person
NOT_PERSON_LEFT = re.compile(
    r"(?:вул\.|вулиц\w+|просп\w*\.?|пров\.|провулок|провулку|площ\w+|пл\.|бульвар\w*|бул\.|набережн\w+|узвіз|"
    r"імені|ім\.|ім'я|ім’я|академіка|акад\.|проф\.|професора|справ[аі]\s+[«\"]|ВАТ|ТОВ|ПрАТ|ПАТ|ПП|КП|ФГ|СФГ|ДП|НВП|ТДВ)\s*$",
    re.I,
)

# --- numbers ---
RNOKPP = re.compile(r"(?:РНОКПП|ІПН|реєстраційн\w+\s+номер\w*\s+обліков\w+\s+картк\w+|ідентифікаційн\w+\s+(?:номер|код)\w*|"
                    r"податков\w+\s+номер\w*)\W{0,25}?(\d{10})\b", re.I)
PASSPORT_SN = re.compile(r"\b([А-ЯІЇЄҐA-Z]{2}\s?\d{6})\b")
PASSPORT_CTX = re.compile(r"(паспорт|посвідч\w+\s+особ|документ\w*,?\s+що\s+посвідч|ID-?карт|ід-?карт)", re.I)
NOT_PASSPORT_CTX = re.compile(r"(купюр|банкнот|грн|гривень|акт[аіу]?\s|держ\w*\s+акт|протокол|ліценз|припис|сертифікат|свідоцтв|"
                              r"диплом|квит\w+|талон|бланк|серії\s+та\s+номер|поліс|довіреност|ордер|посвідчення\s+водія|техпаспорт)", re.I)
ID_CARD = re.compile(r"\b(\d{9})\b")
ID_CTX = re.compile(r"(паспорт|ID-?карт|ід-?карт|№\s*запису|УНЗР|унікальн\w+\s+номер\w*\s+запису)", re.I)
UNZR = re.compile(r"\b(\d{8}-\d{5})\b")
IBAN = re.compile(r"\bUA\d{27}\b")
# --- enforcement-proceeding numbers; every number of a match is a capture of the group «n»: m.spans("n") ---
# «код ЄДРПОУ » / «Код за ЄДРПОУ » / «код в ЄДР-» / «ЄДРПОУ – » / «код » right before «ВП»: there ВП is «відокремлений підрозділ»
# and the number is a company code. Only link words may stand in between: «з кодом доступу ВП № …» is a proceeding
_VP_CODE = r"(?:ЄДРПОУ|\b(?i:код(?:у|ом|а|і)?))(?:\s+(?:за|в|у)(?!\w)|\s+(?i:ЄДР\w*)|\s*[:\-–—])*\s*"
# «ВП» / «АСВП» / «ЗВП» / «виконавчому провадженні» (not «ЄДРПОУ ВП», not «ВПО»); «ВП75258642» is allowed, a digit may follow «ВП».
# The company-code lookbehind stands behind the literal: in front of the alternation it ran at every position (13 times slower)
_VP_CUE = rf"(?:\bВП(?<!{_VP_CODE}ВП)(?![^\W\d])|\b(?:АСВП|ЗВП)(?!\w)|\b(?i:виконавч\w+\s+провадженн\w+)(?!\w))"
# « від 23.07.2021» / « від 22 грудня 2023 року» between the cue and the number
_VP_DATE = r"(?:\s+від\s+\d{1,2}(?:\.\d{2}\.|\s+[а-яіїєґ]+\s+)\d{4}(?:\s*(?:р\.|року))?)?"
# « № » / «№№» / « за № » / « по № » / « N » / « N?» (a mangled «№») / «: № » — a number sign
_VP_NO = r"(?:\s*(?:за|по)(?!\w))?\s*:?\s*(?:(?:№\s*)+|N\s*[?°º]?\s*):?\s*"
# «74591493»; not «755/19637/17» (a case number), not «1434048,11» (an amount), not the head of a longer number
_VP_NUM = r"(?<![\d/\-])(?P<n>\d{7,10})(?![\w/]|-\d|[.,]\d{2}(?!\d))"
# the same without a number sign before it: not «1500000 гривень» / «2500000 грн» (an amount), not «0501234567» (a phone;
# no proceeding number of the corpus starts with 0)
_VP_BARE = rf"(?!0){_VP_NUM}(?!\s*(?:тис|млн|грн|грив|коп|дол|євро))"
# «№ 74591493» / «: 68293414» / « 62884789» — one number with or without the sign
_VP_ITEM = rf"(?:{_VP_NO}{_VP_NUM}|\s*:?\s*{_VP_BARE})"
# « та № 74591949» / «, 69474005» / «; №63958006» — the next numbers of a list
_VP_MORE = rf"(?:\s*(?:[,;]|та\b|і\b|й\b)\s*(?:№\s*{_VP_NUM}|{_VP_BARE}))"
# «виконавчі провадження № 74591493 та № 74591949», «ВП 62884789», «провадження від 23.07.2021 № 66181350», «№ АСВП: 68293414»
VP = re.compile(rf"{_VP_CUE}{_VP_DATE}{_VP_ITEM}{_VP_MORE}*")
# «виконавчі провадження: № » right before a filled НОМЕР placeholder: use as VP_LIST_HEAD.search(text, start - VP_HEAD_WINDOW, start)
VP_LIST_HEAD = re.compile(rf"{_VP_CUE}{_VP_DATE}(?:{_VP_NO}|\s*:?\s*)\Z")
VP_HEAD_WINDOW = 100  # the longest head: «виконавчому провадженні від 22 грудня 2023 року за № »
# «, № 58365604, № 61493933 та 61876187» right after that placeholder: use as VP_LIST_MORE.match(text, end)
VP_LIST_MORE = re.compile(rf"{_VP_MORE}+")
# «Фізична особа-підприємець » / «ФОП » / «боржник - » / «стягувача: » right before an open name: a party, not an official
VP_PARTY_LEFT = re.compile(r"(?:\bФОП|(?i:\bфізичн\w+\s+особ\w+\s*[-–—]?\s*підприєм\w+|\bборжни\w+|\bстягувач\w*))\s*[-–—:,]?\s*\Z")
# «третя особа, яка не заявляє самостійних вимог на предмет спору, на стороні відповідача Фізична особа-підприємець » before
# that name: a third party of a dispute between companies (decision 119839903), not the debtor
VP_THIRD_LEFT = re.compile(r"(?i:\bтрет\w+\s+ос[оі]б\w*)[^.;\n]{0,130}\Z")
CARD16 = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b")
CARD_CTX = re.compile(r"(картк\w*|карт[аи]\b|card)", re.I)
IBAN_PERSON = re.compile(r"(на\s+користь|стягувач\w*|позивач\w*|відповідач\w*|боржник\w*|"
                         r"картков\w+\s+рахун\w+|особов\w+\s+рахун\w+|рахун\w+\s+(?:фізичної\s+особи|громадянина|громадянки)|"
                         r"ОСОБА_\d+|заявник\w*|спадкоємц\w+|вклад\w*|депозит\w*|заробітн\w+)", re.I)
IBAN_INSTITUTION = re.compile(r"(UA\d{2}899998|ДКСУ|Казначейств\w*|казначейськ\w+|бюджет\w*|ГУК|УК\s+у|судов\w+\s+збор\w*|судов\w+\s+збір|"
                              r"ЄДРПОУ|код\s+\d{8}|ТОВ|ПрАТ|ПАТ|АТ\s|ПП\s|КП\s|УДКСУ|ГУ\s+ДПС|ДПС|ПФУ|Пенсійн\w+\s+фонд\w*|"
                              r"виконавч\w+\s+служб\w+|державн\w+\s+виконавц\w+|суд[уі]\b|поштов\w+|банк\w*\s*\(|"
                              r"депозитн\w+\s+рахун\w+\s+суду|філі[яї]|відділенн\w+|управлінн\w+|департамент\w*)", re.I)

# --- masks / info ---
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PERSONAL_MAIL = re.compile(r"@(gmail\.com|ukr\.net|i\.ua|meta\.ua|outlook\.com|icloud\.com|yahoo\.com|email\.ua|online\.ua|"
                           r"bigmir\.net|mail\.ru|yandex\.ru|hotmail\.com|proton\.me|protonmail\.com)$", re.I)
GOV_MAIL = re.compile(r"\.gov\.ua$|court\.gov\.ua$|arbitr\.gov\.ua$", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?38)?\s?\(?0\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)")
PLATE = re.compile(r"\b[АВЕІКМНОРСТХABEIKMHOPCTX]{2}\s?\d{4}\s?[АВЕІКМНОРСТХABEIKMHOPCTX]{2}\b")
DATE_NUM = re.compile(r"\b(\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})\b")
DATE_WORDS = re.compile(r"\b(\d{1,2}\s+(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)\s+\d{4}(?:\s+року)?(?:\s+народження)?)")
DOB_CTX = re.compile(r"(народ\w*|р\.\s?н\.|уродж\w+|дата\s+народж\w+|д\.н\.|помер\w*|смерт\w*|загин\w+)", re.I)
_ADDR_ABBR = r"(?:м|с|смт|сел|сщ|вул|просп|пр-т|пр|пров|бульв|бул|пл|буд|б|кв|корп|оф|обл|р-н|пгт|х)"
_ADDR_CHUNK = (rf"(?:\d{{5}}|{_ADDR_ABBR}\.?\s*[^,;\n]{{1,45}}|(?:місто|село|селище|вулиця|проспект|провулок|бульвар|будинок|квартира|область|район)\s+[^,;\n]{{1,45}}|"
               rf"[А-ЯІЇЄҐ][^,;\n]{{2,40}}?(?:область|обл\.|район|р-н)|[А-ЯІЇЄҐ][а-яіїєґ'’\-]+(?:\s+[А-ЯІЇЄҐ]?[а-яіїєґ'’\-]+)?(?:\s+\d+[А-Яа-яA-Za-z/\-]*)?|\d+[А-Яа-яA-Za-z0-9/\-]{{0,6}})")
ADDRESS_CTX = re.compile(r"(?i:проживає|проживаюч\w+|проживав|проживала|зареєстрован\w+|мешка\w+|місце\s+проживання|місце\s+реєстрації|"
                         r"адрес\w+\s+(?:проживання|реєстрації))[^.;\n]{0,40}?(?i:за\s+адресою|адресою|адреса)\s*:\s*"
                         rf"({_ADDR_CHUNK}(?:,\s*{_ADDR_CHUNK}){{1,8}})")
NOT_ADDRESS_LEFT = re.compile(r"(ТОВ|ПрАТ|ПАТ|ПП|КП|ДП|ФОП|підприємств\w*|юридичн\w+|місцезнаходженн\w+|яке\s+зареєстроване|майно|квартир\w+,?\s+яка)", re.I)

INI_ANY = re.compile(rf"(?<![\w'’]){INI}(?!\w)|(?<![\w'’.]){INI_R}(?!\w)", re.U)
TIME_RIGHT = re.compile(r"^\s*(?:року\s+)?(?:біля|близько|о\s|об\s|приблизно|в\s+період|з\s+\d|по\s+\d|у\s+період|близько)", re.I)
DOB_CUE_LEFT = re.compile(r"(?:народ\w*|р\.\s?н\.|уродж\w+|дата\s+народж\w+|д\.н\.|помер\w*|смерт[іь]|загин\w+|народження\s*[-–—:]?)[\s,:\-–—]*$", re.I)
DOB_CUE_RIGHT = re.compile(r"^\s*(?:р\.\s?н\.|року\s+народж\w+|р\.\s+народж\w+|народження)", re.I)
NOT_PASSPORT_LEFT = re.compile(r"(№|упакуван\w*|квитанц\w*|пакет\w*|бланк\w*|накладн\w*|сейф|конверт\w*|пломб\w*|бирк\w*)\s*$", re.I)
NOT_DOB_LEFT = re.compile(r"(народжень|актов\w+\s+запис\w*|запис\w*\s+про\s+народження|свідоцтв\w*\s+про\s+народження|реєстрації\s+народжен\w*)[^,;.]{0,25}$", re.I)
COMPANY_LEFT = re.compile(r"(ЄДРПОУ|код\s+за|ТОВ|СТОВ|ПрАТ|ПАТ|АТ\s|ПП\s|КП\s|ДП\s|ОСББ|підприємств\w*|товариств\w*|компані\w+|банк\w*|управлінн\w+|відділ\w*|департамент\w*|суд[уі]?\b|кредитор\w*|організаці\w+|установ\w+|фонд\w*)", re.I)
