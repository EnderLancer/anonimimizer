"""Гібрид: spaCy uk_ner_web_trf_13class (PERS -> ОСОБА) + правила для НОМЕР / ІНФОРМАЦІЯ / АДРЕСА.

python predict_trf.py cases.jsonl out.jsonl [--limit-ids ids.txt]
Результат NER кешується в ner_cache.json (щоб правила можна було перезапускати без повторного інференсу).
"""
import json
import os
import re
import sys
import time
from collections import defaultdict

from predict import cluster_persons

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("UK_NER_MODEL") or os.path.join(HERE, "models", "uk_ner_web_trf_13class")  # шлях до spaCy-моделі можна задати змінною UK_NER_MODEL
CACHE = os.path.join(HERE, "ner_cache.json")

UP = "А-ЯІЇЄҐ"
LOW = "а-яіїєґ'’ʼ"

# ---- НОМЕР (v2): формати за стандартами + ключові слова ----
LAT = "A-Z"
CYR = "А-ЯІЇЄҐ"
PLATE_L = "АВЕІКМНОРСТХABEKMHOPCTXI"
NUM_PATTERNS = [
    # телефони: +380.., 380.., 0XX з різними роздільниками, дужки
    r"(?<![\w/.])\+?3?8?[\s-]?\(?0\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?![\d/])",
    r"(?<![\w/.])\+?380\d{9}(?!\d)",
    r"(?<![\w/.])0\d{9}(?![\d/])",
    # IBAN
    r"(?<![\w])UA\d{2}(?:\s?\d){25}(?!\d)",
    # номерні знаки (стандартні, діппредставництв, старі)
    rf"(?<![\w])[{PLATE_L}]{{2}}\s?\d{{4}}\s?[{PLATE_L}]{{2}}(?![\w])",
    rf"(?<![\w])\d{{5}}\s?[{CYR}]{{2}}(?![\w])",
    # номерні знаки з дефісами XX-0000-XX (кирилиця/латиниця)
    rf"(?<![\w-])[{CYR}{LAT}]{{2}}-\d{{4}}-[{CYR}{LAT}]{{2}}(?![\w-])",
    # паспорт: серія (2 літери) + 6 цифр
    rf"(?<![\w])[{CYR}]{{2}}\s?\d{{6}}(?!\d)",
    # закордонний паспорт, латиницею
    rf"(?<![\w])[{LAT}]{{2}}\s?\d{{6}}(?!\d)",
    # номер картки 16 цифр
    r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",
    # VIN
    r"(?<![A-Za-z0-9])(?=[A-HJ-NPR-Z0-9]*\d)(?=[A-HJ-NPR-Z0-9]*[A-HJ-NPR-Z])[A-HJ-NPR-Z0-9]{17}(?![A-Za-z0-9])",
]
NUM_RE = [re.compile(p) for p in NUM_PATTERNS]

_SEP = r"[\s:№N.,«»\"'()=-]{0,14}"
_VAL = rf"((?:[{CYR}{LAT}]{{1,3}}[\s-]?)?\d[\d\s-]{{2,22}}\d|[{CYR}{LAT}]{{1,3}}\d{{3,10}})"
KW_ID = (r"(?:РНОКПП|РНОКПП|ІПН|ідентифікаційн\w*\s+(?:номер|код)\w*|облікової\s+картки\s+платника\s+податків|"
         r"паспорт\w*|ID[- ]?картк\w*|ІD[- ]?картк\w*|посвідчен\w+(?:\s+водія)?|водійськ\w+\s+посвідчен\w+|"
         r"картк\w*|IMEI|ІМЕІ|імеі|VIN|номер\w*\s+(?:кузова|шасі)|р/р|IBAN|"
         r"номер\w*\s+телефон\w*|телефон\w*|тел\.?|моб\.?|абонентськ\w+\s+номер\w*|сім-картк\w*|"
         r"номерн\w+\s+знак\w*|держ\w*\s+номер\w*|реєстраційн\w*\s+номер\w*|ДНЗ|д\.н\.з\.?|р/н|н/з|знак\w*)")
CTX_NUM = re.compile(rf"(?i)\b{KW_ID}{_SEP}(?:(?:серії|серія|серією)\s*)?{_VAL}")
MIL = re.compile(rf"(?i)(?:в/ч|в\.ч\.|військов\w+\s+частин\w*|ВЧ)\s*[№N]?\s*([{CYR}{LAT}]{{0,2}}\s?-?\d{{3,5}})(?![\d/])")

# ---- ІНФОРМАЦІЯ ----
# номери виконавчих проваджень (стосуються конкретної особи-боржника), машиномісця
PROC_VAL = r"(\d{8,10}|[А-ЯІЇЄҐ]{2}/\d{3,5})(?![\d/])"
EXEC_PROC = re.compile(rf"(?i)виконавч\w+\s+провадж\w*\s*(?:від\s+[^№]{{0,30}})?[№N]?\s*{PROC_VAL}")
PROC_YEAR = re.compile(r"(?i)провадженн\w+[^№\n]{0,45}№\s*(\d{3,4}/(?:19|20)\d{2}/\d{2})(?![\d/])")
NUM_CODE = re.compile(r"(?<![\w/])№\s*([А-ЯІЇЄҐ]{2}/\d{4})(?![\d/])")
PARKING = re.compile(r"(?i)машино-?місц\w*\s*(?:№|номер)\s*(\d{1,4})(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)[^\s,;<>\"«»)]+")
DATE = (r"(?:\d{1,2}[./]\d{1,2}[./]\d{4}|\d{1,2}\s+(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|"
        r"вересня|жовтня|листопада|грудня)\s+\d{4})(?:\s*(?:р\.|року|рік))?")
BIRTH_CTX = re.compile(
    r"(?i)(?:народжен\w*|нар\.|року\s+народження|р\.\s*н\.|уродж\w+|дата\s+народження|помер\w*|смерт\w*|померл\w+)")
DATE_RE = re.compile(DATE)
# дата одразу після імені особи (у ЄДРСР це майже завжди дата народження)
DATE_ISO = r"(?<![\d-])(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?![\d-])"
DATE_ALL_RE = re.compile(rf"{DATE}|{DATE_ISO}")
# дати маскуємо лише з контекстом народження/смерті або одразу після особи (AFTER_PERSON); решта дат у судових рішеннях — процесуальні
DATE_ANY = rf"(?:{DATE}|{DATE_ISO}|\d{{1,2}}\.\d{{1,2}}\.\d{{2}}(?!\d)|(?:19|20)\d{{2}}\s*(?:року\s+народження|р\.\s*н\.|року\s+н\.|р\.\s*народження))(?:\s*(?:року\s+народження|р\.\s*н\.|р\.н\.|року\s+н\.|року|р\.|рік))?"
AFTER_PERSON = re.compile(rf"\s*[,(]\s*({DATE_ANY})")
YEAR_BIRTH = re.compile(r"(?<!\d)(?:19|20)\d{2}\s*(?:року\s+народження|р\.\s*н\.|р\.н\.|року\s+н\.|р\.\s*народження)")
DEATH_CTX = re.compile(r"(?i)(?:помер\w*|померл\w+|смерті|смерть|загинув\w*|похован\w+)")
HANDLE = re.compile(r"(?<![\w.@])@[A-Za-z0-9_.]{3,32}(?![\w@])")
NICK = re.compile(r"(?i)(?:нік\w*|нікнейм\w*|акаунт\w*|аккаунт\w*|логін\w*|username)\s*[:=«\"'(]?\s*([A-Za-z0-9_.-]{3,32})(?![\w])")
SOCIAL_URL = re.compile(r"(?i)(?<![\w/@.])(?:facebook|instagram|t|telegram|tiktok|twitter|vk|ok|youtube|linkedin)\.(?:com|me|ru|ua|org)/[^\s,;<>\"«»)]+")


# ---- ІНФОРМАЦІЯ: організації (юрособи, військові формування, держорганізації) ----
_Q = r"[«\"“„]"
_QE = r"[»\"”“]"
ORG_LEGAL = re.compile(
    rf"(?:Товариств\w+\s+з\s+обмеженою\s+відповідальністю|(?:Приватн\w+|Публічн\w+)?\s*акціонерн\w+\s+товариств\w+|ТОВ|ПрАТ|ПАТ|АТ|ПП|ТзОВ)\s*{_Q}[^«»\"“”„\n]{{1,60}}{_QE}", re.I)
ORG_MIL = re.compile(
    rf"(?:Військов\w+\s+навчальн\w+\s+центр\w*|Відділенн\w+\s+прикордонн\w+\s+служби)\s*{_Q}[^«»\"“”„\n]{{1,40}}{_QE}"
    r"|Військово-лікувальн\w+\s+комісі\w+(?:\s+[А-ЯІЇЄҐ]\w+\s+госпітал\w+)?(?:\s+Міністерства\s+оборони\s+України)?"
    r"|(?<![\w-])ТЦК(?:\s+та\s+СП)?(?![\w-])"
    r"|(?:[А-ЯІЇЄҐ][\w'’-]+\s+(?:районн\w+\s+)?)?територіальн\w+\s+центр\w*\s+комплектування\s+та\s+соціальної\s+підтримки")
_ORG_HEAD = (r"(?:Міністерств\w+|Департамент\w*|(?:Головн\w+|Центральн\w+|Територіальн\w+)\s+управлінн\w+|Управлінн\w+|Відділ\w*|Відділенн\w+)")
ORG_GOV = re.compile(rf"(?<![\w-]){_ORG_HEAD}(?:\s+[\w'’-]+){{0,12}}?\s+(?:України|Україні|області|Києві|адміністрації)(?![\w-])")

# ---- АДРЕСА (v5): структурний шаблон індекс/область/район/місто/вулиця/будинок/квартира ----
# v5: відмінкові форми вулиці/міста (вулиці, місті...), складені назви (Івано-Франківська),
#     хвіст після будинку/квартири (село, район, область, індекс у довільному порядку), в'їзд, гараж/ділянка.
W = r"[А-ЯІЇЄҐA-Z][а-яіїєґa-z'’ʼ]+(?:-[А-ЯІЇЄҐA-Zа-яіїєґa-z'’ʼ]+)*"
IDX = r"\d{5}(?!\d)(?!\s*(?:грн|коп|євро|дол))"
REGION = rf"{W}\s+(?:область|області|обл\.)"
DIST = rf"{W}\s+(?:район|району|р-н)"
CITY = rf"(?:м\.|с\.|смт\.?|с-ще|місто|місті|міста|місту|село|селі|села|селище|селищі)\s*{W}(?:[ -]{W})?"
STREET = (r"(?:вул\.|вулиц\w*|просп\.|проспект\w*|пров\.|провулок|провулк\w*|бул\.|бульвар\w*|б-р|пл\.|площ(?:а|і|у)(?=\s+[А-ЯІЇЄҐ])|шосе|наб\.|набережн\w*|узвіз\w*|тупик\w*|в['’ʼ]їзд\w*)"
          r"\s*(?:[А-ЯІЇЄҐA-Z0-9][^\s,;()]*\.?\s?){1,4}?")
HOUSE = r"(?:буд\.|будинок|б\.)?\s*№?\s*\d{1,4}(?:[-/]\d{1,3})?[А-ЯІЇЄҐа-яіїєґA-Za-z]?(?![\d])"
CORP = r"(?:корп\.|корпус)\s*№?\s*\d{1,3}[А-Яа-я]?"
APT = r"(?:кв\.|квартира|оф\.|офіс|приміщення|гараж|ділянка|земельна\s+ділянка)\s*№?\s*\d{1,4}[А-Яа-я]?"
TAIL = rf"(?:\s*,\s*(?:{IDX}|{REGION}|{DIST}|{CITY})){{0,5}}"
ADDR_STREET = re.compile(
    rf"(?:{IDX}\s*,?\s*)?(?:{REGION}\s*,?\s*)?(?:{DIST}\s*,?\s*)?(?:{CITY}\s*,?\s*)?"
    rf"{STREET}\s*,?\s*{HOUSE}(?:\s*,?\s*{CORP})?(?:\s*,?\s*{APT})?{TAIL}")
# населений пункт (село/смт) + район [+ область] без вулиці: структура досить специфічна
ADDR_VILLAGE = re.compile(
    rf"(?:{IDX}\s*,?\s*)?(?:{REGION}\s*,?\s*)?(?:с\.|смт\.?|с-ще|село|селище)\s*{W}(?:[ -]{W})?\s*,\s*{DIST}(?:\s*,\s*{REGION})?")
# без вулиці (лише населений пункт/область) — тільки після тригера проживання/реєстрації
TRIG = (r"(?i)(?:проживає|проживаючий|проживаючого|проживаючої|мешкає|мешканц\w+|зареєстрован\w+|місц\w+\s+(?:проживання|реєстрації|перебування)|"
        r"адрес\w+\s+(?:проживання|реєстрації|місця\s+\w+)|за\s+адресою)\s*(?:за\s+адресою)?\s*[:,]?\s*(?:а\s+саме\s*:?\s*)?")
ADDR_NOSTREET = re.compile(
    rf"{TRIG}((?:{IDX}\s*,?\s*)?(?:{REGION}\s*,?\s*)?(?:{DIST}\s*,?\s*)?{CITY}(?:\s*,\s*{DIST})?)")
ORG_CTX = re.compile(r"(?i)(?:суд\w*|прокуратур\w*|управлін\w*|відділ\w*|банк\w*|(?-i:(?<!\w)(?:ТОВ|АТ|ПАТ|ПрАТ)(?!\w))|нотаріус\w*|адреса\s+суду|АЗС|поліці\w+)")
PERSONAL_CTX = re.compile(r"(?i)(?:проживає|мешкає|мешканц\w+|зареєстрован\w+|проживаюч\w+|місц\w+\s+(?:проживання|реєстрації|перебування)|адрес\w+\s+(?:проживання|реєстрації))")


# ---- НОМЕР (generic): «усе, де є цифри, окрім винятків» (низький пріоритет) ----
G_TOK = re.compile(r"[A-Za-zА-Яа-яІЇЄҐіїєґ]{0,3}[\s-]?\d[\d\-/.,:]*\d|\d")
G_MONEY = re.compile(r"^\s*(?:грн|гривн|тис|млн|млрд|коп|%|відсот|кв\.?\s*м|м2|м²|га\b|кг|км|год|хв|р\.|року|рік|років|євро|дол|usd|eur)", re.I)
G_PRE = re.compile(r"(?i)(?:ст\.|ст\s|статт\w*|ч\.|п\.|пп\.|абз\.|№\s*справ\w*|справ\w*\s*№?|провадженн\w*\s*№?|за\s*№|від)\s*$")
G_ORG = re.compile(r"(?i)(?:ЄДРПОУ|код\w*|ЄДР|реєстраційн\w+\s+номер\s+об['`’]?єкта\w*|об['`’]?єкта\s+нерухомого\s+майна|номер\s+запис\w*|"
                   r"кадастров\w+\s+номер\w*|договор\w*|контракт\w*|рахунок\w*\s+(?:ДКСУ|казначейств\w*)|платіжн\w+\s+інструкці\w*|P\.O\. Box|"
                   r"[Іі]ндекс|накладн\w+|ухвал\w+|постанов\w+|рішенн\w+|наказ\w*|акт\w*|протокол\w*|свідоцтв\w+\s+про\s+державн\w+)\W{0,14}$")


def generic_numbers(text):
    out = []
    for m in G_TOK.finditer(text):
        tok = m.group(0)
        d = re.sub(r"\D", "", tok)
        if len(d) <= 4 or "/" in tok:
            continue
        if re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}:\d{2}", tok):
            continue
        if re.search(r"[.,]\d{1,2}$", tok) and len(d) <= 9:
            continue
        if G_MONEY.match(text[m.end():m.end() + 14]) or G_PRE.search(text[max(0, m.start() - 18):m.start()]):
            continue
        if G_ORG.search(text[max(0, m.start() - 60):m.start()]):
            continue
        if len(d) == 5 and re.search(r"(?i)(?:м\.|місто|суд|області|обл\.)", text[m.end():m.end() + 20]):
            continue
        s0, e0 = m.start(), m.end()
        frag = text[s0:e0]
        s0 += len(frag) - len(frag.lstrip(" -"))
        e0 = s0 + len(text[s0:e0].rstrip(".,:-/ "))
        out.append((s0, e0, "НОМЕР"))
    return out


def spans_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]



# організація-сторона справи: маскується, якщо перед нею тригер ролі (позов до…, скарга…, відповідач…) або після неї «(далі - …)»;
# далі маскуються всі повтори тієї ж організації в документі
MIL_UNIT = re.compile(r"(?<![\w-])Військов\w+\s+частин\w+\s+[А-ЯA-Z]\s?\d{3,5}(?!\d)")
PARTY_PRE = re.compile(r"(?i)(?:позов\w*|скарг\w*|заяв\w*)\s[^.;]{0,160}?(?<![\w-])до\s*$|\(далі\s*[-–—]?\s*$|зобов['’]язати\s*$|дії\s*$")
PARTY_POST = re.compile(r"\s*\(далі\b")
def _org_key(t):
    q = re.search(r"[«\"“„]([^«»\"“”„]+)", t)
    if q:
        return "q:" + q.group(1).strip().lower()
    w = re.findall(r"[\w'’-]+", t.lower())
    return "g:" + " ".join(x[:6] for x in w[:3])
def party_orgs(text):
    cands = [(m.start(), m.end()) for R in (ORG_GOV,) for m in R.finditer(text)]
    keys = set()
    for s, e in cands:
        if PARTY_PRE.search(text[max(0, s - 200):s]) or PARTY_POST.match(text, e):
            keys.add(_org_key(text[s:e]))
    return [(s, e, "ІНФОРМАЦІЯ") for s, e in cands if _org_key(text[s:e]) in keys]

VP_NUM = re.compile(r"(?<![\w])ВП\s*№\s*(\d{8,10})(?!\d)")
ID10 = re.compile(r"(?i)(?:РНОКПП|ІПН|ідентифікаційн\w*\s+(?:код|номер)\w*|(?<!\w)код|(?<!\w)о/р)[^\d\n]{0,25}?(\d{10})(?!\d)")
# номери, що не стосуються особи: кадастрові, акти готовності, квитанції, записи реєстрів, поштові, номери справ
NUM_NEG = re.compile(r"(?i)(?:кадастров\w*\s+номер\w*(?:\s+земельної\s+ділянки)?|готовн\w+\s+(?:до\s+експлуатації\s+)?об['’]?єкта?\w*(?:\s+до\s+експлуатації)?|квитанці\w+|номер\w*\s+запис\w*|поштового\s+відправлення|справ\w*)\s*[:№]?\s*$")
ISSUED = re.compile(r"(?i)видан\w*\s*$")


def treasury_iban(frag):
    """Рахунки казначейства (код 899998) розмітка не маскує."""
    return "899998" in re.sub(r"\s", "", frag)[4:10]


def rule_spans(text, pers=()):
    out = []
    out += [(m.start(1), m.end(1), "НОМЕР") for m in VP_NUM.finditer(text)]
    out += [(m.start(1), m.end(1), "НОМЕР") for m in ID10.finditer(text)]
    for r in NUM_RE:
        for m in r.finditer(text):
            out.append((m.start(), m.end(), "НОМЕР"))
    for m in CTX_NUM.finditer(text):
        s, e = m.start(1), m.end(1)
        frag = text[s:e].rstrip(" -")
        digits = re.sub(r"\D", "", frag)
        if "/" in frag or len(digits) < 3:
            continue
        out.append((s, s + len(frag), "НОМЕР"))
    for m in MIL.finditer(text):
        out.append((m.start(1), m.end(1), "НОМЕР"))
    for R in (EXEC_PROC, PROC_YEAR, NUM_CODE, PARKING):
        for m in R.finditer(text):
            out.append((m.start(1), m.end(1), "НОМЕР"))
    for m in EMAIL.finditer(text):
        out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for m in URL.finditer(text):
        if "reyestr.court.gov.ua" in m.group(0) or "court.gov.ua" in m.group(0):
            continue
        out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for m in DATE_ALL_RE.finditer(text):
        ctx = text[max(0, m.start() - 45):m.start()] + " " + text[m.end():m.end() + 25]
        if ISSUED.search(text[max(0, m.start() - 12):m.start()]):
            continue
        if BIRTH_CTX.search(ctx) or DEATH_CTX.search(text[max(0, m.start() - 30):m.start()]):
            out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for m in YEAR_BIRTH.finditer(text):
        out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for (ps, pe) in pers:
        m = AFTER_PERSON.match(text, pe)
        if m:
            out.append((m.start(1), m.end(1), "ІНФОРМАЦІЯ"))
    for m in HANDLE.finditer(text):
        out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for m in NICK.finditer(text):
        out.append((m.start(1), m.end(1), "ІНФОРМАЦІЯ"))
    for m in SOCIAL_URL.finditer(text):
        out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for R in (ORG_MIL,):
        for m in R.finditer(text):
            out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    out.extend(party_orgs(text))
    for m in ADDR_STREET.finditer(text):
        frag = text[m.start():m.end()].rstrip(" ,")
        pre = text[max(0, m.start() - 90):m.start()]
        if ORG_CTX.search(pre[-60:]) and not PERSONAL_CTX.search(pre):
            continue
        out.append((m.start(), m.start() + len(frag), "АДРЕСА"))
    for m in ADDR_VILLAGE.finditer(text):
        pre = text[max(0, m.start() - 90):m.start()]
        if ORG_CTX.search(pre[-60:]) and not PERSONAL_CTX.search(pre):
            continue
        out.append((m.start(), m.end(), "АДРЕСА"))
    for m in ADDR_NOSTREET.finditer(text):
        out.append((m.start(1), m.end(1), "АДРЕСА"))
    return out


def trf_spans(nlp, texts):
    res = []
    for doc in nlp.pipe(texts, batch_size=1):
        res.append([(e.start_char, e.end_char, e.text) for e in doc.ents if e.label_ == "PERS"])
    return res


def clean_person(text, s, e):
    frag = text[s:e]
    frag = frag.rstrip(" ,;:)(«»\"")
    lead = len(frag) - len(frag.lstrip(" ,;:)(«»\""))
    return s + lead, s + len(frag)


def pseudonymize(text, pers, extra=(), generic=False, rules=True):
    spans = []
    for s, e, _ in pers:
        s, e = clean_person(text, s, e)
        if e > s and re.search(r"[^\W\d_]", text[s:e]):
            spans.append((s, e, "ОСОБА"))
    pers_spans = [(a, b) for a, b, t in spans if t == "ОСОБА"]
    cand = (list(rule_spans(text, pers_spans)) if rules else []) + [tuple(x) for x in extra]
    cand = [c for c in cand if not (c[2] == "НОМЕР" and (NUM_NEG.search(text[max(0, c[0] - 70):c[0]]) or treasury_iban(text[c[0]:c[1]])))]
    for sp in sorted(cand, key=lambda x: -(x[1] - x[0])):
        if not any(spans_overlap(sp, o) for o in spans):
            spans.append(sp)
    if generic:
        for sp in generic_numbers(text):
            if not any(spans_overlap(sp, o) for o in spans):
                spans.append(sp)
    spans.sort()
    labels, counters, seen = {}, defaultdict(int), {}
    pi = [k for k, sp in enumerate(spans) if sp[2] == "ОСОБА"]
    gids = cluster_persons([text[spans[k][0]:spans[k][1]] for k in pi]) if pi else []
    pmap = {}
    for k, g in zip(pi, gids):
        if g not in pmap:
            counters["ОСОБА"] += 1; pmap[g] = counters["ОСОБА"]
        labels[k] = f"ОСОБА_{pmap[g]}"
    for k, (s, e, t) in enumerate(spans):
        if t == "ОСОБА":
            continue
        key = (t, re.sub(r"\W", "", text[s:e].lower()))
        if key not in seen:
            counters[t] += 1; seen[key] = counters[t]
        labels[k] = f"{t}_{seen[key]}"
    return [{"start": s, "end": e, "pseudonym": labels[k]} for k, (s, e, t) in enumerate(spans)]


def main(src, dst):
    rows = [json.loads(l) for l in open(src, encoding="utf8")]
    cache = json.load(open(CACHE, encoding="utf8")) if os.path.exists(CACHE) else {}
    todo = [r for r in rows if str(r["id"]) not in cache]
    if todo:
        import spacy
        nlp = spacy.load(MODEL, disable=[])
        t0 = time.time()
        for i, r in enumerate(todo):
            cache[str(r["id"])] = trf_spans(nlp, [r["original"]])[0]
            if i % 10 == 0:
                json.dump(cache, open(CACHE, "w", encoding="utf8"), ensure_ascii=False)
                print(f"{i}/{len(todo)} {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
        json.dump(cache, open(CACHE, "w", encoding="utf8"), ensure_ascii=False)
    with open(dst, "w", encoding="utf8") as g:
        for r in rows:
            g.write(json.dumps({"id": r["id"], "annotations": pseudonymize(r["original"], cache[str(r["id"])])}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
