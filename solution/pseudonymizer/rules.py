"""Регулярні вирази для НОМЕР / ІНФОРМАЦІЯ / АДРЕСА.

Правила написані за описом завдання, стандартами форматів (ІПН, IBAN, номерні знаки, телефони тощо)
та ключовими словами зі статистики тренувального корпусу 2024-court-decisions. Тестова вибірка для
підбору правил не використовувалась. Усі правила повертають спани (start, end, type).
"""
import re

CYR = "А-ЯІЇЄҐ"
LAT = "A-Z"
PLATE_L = "АВЕІКМНОРСТХABEKMHOPCTXI"

# ---------------------------------------------------------------- НОМЕР: формати
NUM_PATTERNS = [
    # телефони: +380.., 380.., 0XX з різними роздільниками, дужки
    r"(?<![\w/.])\+?3?8?[\s-]?\(?0\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?![\d/])",
    r"(?<![\w/.])\+?380\d{9}(?!\d)",
    r"(?<![\w/.])0\d{9}(?![\d/])",
    # IBAN
    r"UA\s?\d{2}(?:\s?\d{4}){5}\s?\d{0,3}",
    # номерні знаки (стандартні, діппредставництв)
    rf"(?<![\w])[{PLATE_L}]{{2}}\s?\d{{4}}\s?[{PLATE_L}]{{2}}(?![\w])",
    rf"(?<![\w])\d{{5}}\s?[{CYR}]{{2}}(?![\w])",
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

# ---------------------------------------------------------------- НОМЕР: ключове слово + значення
_SEP = r"[\s:№N.,«»\"'()=-]{0,14}"
_VAL = rf"((?:[{CYR}{LAT}]{{1,3}}[\s-]?)?\d[\d\s-]{{2,22}}\d|[{CYR}{LAT}]{{1,3}}\d{{3,10}})"
KW_ID = (r"(?:РНОКПП|ІПН|ідентифікаційн\w*\s+(?:номер|код)\w*|облікової\s+картки\s+платника\s+податків|"
         r"паспорт\w*|ID[- ]?картк\w*|ІD[- ]?картк\w*|посвідчен\w+(?:\s+водія)?|водійськ\w+\s+посвідчен\w+|"
         r"картк\w*|IMEI|ІМЕІ|імеі|VIN|номер\w*\s+(?:кузова|шасі)|р/р|IBAN|"
         r"номер\w*\s+телефон\w*|телефон\w*|тел\.?|моб\.?|абонентськ\w+\s+номер\w*|сім-картк\w*|"
         r"номерн\w+\s+знак\w*|держ\w*\s+номер\w*|реєстраційн\w*\s+номер\w*|ДНЗ|д\.н\.з\.?|р/н|н/з|знак\w*)")
CTX_NUM = re.compile(rf"(?i){KW_ID}{_SEP}(?:(?:серії|серія|серією)\s*)?{_VAL}")
MIL = re.compile(rf"(?i)(?:в/ч|в\.ч\.|військов\w+\s+частин\w*|ВЧ)\s*[№N]?\s*([{CYR}{LAT}]?\s?-?\d{{3,5}})(?![\d/])")

# ---------------------------------------------------------------- ІНФОРМАЦІЯ
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)[^\s,;<>\"«»)]+")
DATE = (r"(?:\d{1,2}[./]\d{1,2}[./]\d{4}|\d{1,2}\s+(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|"
        r"вересня|жовтня|листопада|грудня)\s+\d{4})(?:\s*(?:р\.|року|рік))?")
BIRTH_CTX = re.compile(
    r"(?i)(?:народжен\w*|нар\.|року\s+народження|р\.\s*н\.|уродж\w+|дата\s+народження|помер\w*|смерт\w*|померл\w+)")
DATE_RE = re.compile(DATE)
# дата одразу після імені особи (у ЄДРСР це майже завжди дата народження)
DATE_ISO = r"\d{4}-\d{2}-\d{2}"
DATE_ANY = (rf"(?:{DATE}|{DATE_ISO}|\d{{1,2}}\.\d{{1,2}}\.\d{{2}}(?!\d)|(?:19|20)\d{{2}}\s*(?:року\s+народження|р\.\s*н\.|"
            rf"року\s+н\.|р\.\s*народження))(?:\s*(?:року\s+народження|р\.\s*н\.|р\.н\.|року\s+н\.|року|р\.|рік))?")
AFTER_PERSON = re.compile(rf"\s*[,(]\s*({DATE_ANY})")
YEAR_BIRTH = re.compile(r"(?<!\d)(?:19|20)\d{2}\s*(?:року\s+народження|р\.\s*н\.|р\.н\.|року\s+н\.|р\.\s*народження)")
DEATH_CTX = re.compile(r"(?i)(?:помер\w*|померл\w+|смерті|смерть|загинув\w*|похован\w+)")
HANDLE = re.compile(r"(?<![\w.@])@[A-Za-z0-9_.]{3,32}(?![\w@])")
NICK = re.compile(r"(?i)(?:нік\w*|нікнейм\w*|акаунт\w*|аккаунт\w*|логін\w*|username)\s*[:=«\"'(]?\s*([A-Za-z0-9_.-]{3,32})(?![\w])")
SOCIAL_URL = re.compile(r"(?i)(?<![\w/@.])(?:facebook|instagram|t|telegram|tiktok|twitter|vk|ok|youtube|linkedin)\.(?:com|me|ru|ua|org)/[^\s,;<>\"«»)]+")

# ---------------------------------------------------------------- АДРЕСА: індекс/область/район/місто/вулиця/будинок/квартира
W = r"[А-ЯІЇЄҐA-Z][а-яіїєґa-z'’ʼ-]+"
IDX = r"\d{5}"
REGION = rf"{W}\s+(?:область|обл\.)"
DIST = rf"{W}\s+(?:район|р-н)"
CITY = rf"(?:м\.|с\.|смт\.?|с-ще|місто|село|селище)\s*{W}(?:[ -]{W})?"
STREET = (r"(?:вул\.|вулиця|просп\.|проспект|пров\.|провулок|бул\.|бульвар|б-р|пл\.|площа|шосе|наб\.|набережна|узвіз|тупик)"
          r"\s*(?:[А-ЯІЇЄҐA-Z0-9][^\s,;()]*\.?\s?){1,4}?")
HOUSE = r"(?:буд\.|будинок|б\.)?\s*№?\s*\d{1,4}(?:[-/]\d{1,3})?[А-ЯІЇЄҐа-яіїєґA-Za-z]?(?![\d])"
CORP = r"(?:корп\.|корпус)\s*№?\s*\d{1,3}[А-Яа-я]?"
APT = r"(?:кв\.|квартира|оф\.|офіс)\s*№?\s*\d{1,4}[А-Яа-я]?"
ADDR_STREET = re.compile(
    rf"(?:{IDX}\s*,?\s*)?(?:{REGION}\s*,?\s*)?(?:{DIST}\s*,?\s*)?(?:{CITY}\s*,?\s*)?"
    rf"{STREET}\s*,?\s*{HOUSE}(?:\s*,?\s*{CORP})?(?:\s*,?\s*{APT})?")
# без вулиці (лише населений пункт/область) — тільки після тригера проживання/реєстрації
TRIG = (r"(?i)(?:проживає|проживаючий|проживаючого|проживаючої|мешкає|мешканц\w+|зареєстрован\w+|"
        r"місц\w+\s+(?:проживання|реєстрації|перебування)|"
        r"адрес\w+\s+(?:проживання|реєстрації|місця\s+\w+)|за\s+адресою)\s*(?:за\s+адресою)?\s*[:,]?\s*(?:а\s+саме\s*:?\s*)?")
ADDR_NOSTREET = re.compile(
    rf"{TRIG}((?:{IDX}\s*,?\s*)?(?:{REGION}\s*,?\s*)?(?:{DIST}\s*,?\s*)?{CITY}(?:\s*,\s*{DIST})?)")
ORG_CTX = re.compile(r"(?i)(?:суд\w*|прокуратур\w*|управлін\w*|відділ\w*|банк\w*|ТОВ|АТ|ПрАТ|нотаріус\w*|адреса\s+суду|АЗС|поліці\w+)")
PERSONAL_CTX = re.compile(r"(?i)(?:проживає|мешкає|мешканц\w+|зареєстрован\w+|проживаюч\w+|"
                          r"місц\w+\s+(?:проживання|реєстрації|перебування)|адрес\w+\s+(?:проживання|реєстрації))")


# рахунки установ (казначейство, судовий збір, бюджет, отримувач платежу) персональними даними не є
INST_ACCOUNT_CTX = re.compile(
    r"(?i)(?:казначейств\w*|ДКСУ|УДКСУ|бюджет\w*|отримувач\w*|судов\w+\s+збір\w*|судов\w+\s+витрат\w*|МФО|"
    r"код\s+банку|ЄДРПОУ|держав\w+\s+мит\w*|реквізит\w*)")
IBAN_RE = NUM_RE[3]


def spans_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def clean_person(text, s, e):
    """Прибирає зайві розділові знаки по краях спана ОСОБА."""
    frag = text[s:e]
    frag = frag.rstrip(" ,;:)(«»\"")
    lead = len(frag) - len(frag.lstrip(" ,;:)(«»\""))
    return s + lead, s + len(frag)


def rule_spans(text, pers=()):
    """pers — список (start, end) знайдених імен: дата одразу після імені вважається датою народження."""
    out = []
    for r in NUM_RE:
        for m in r.finditer(text):
            if r is IBAN_RE and INST_ACCOUNT_CTX.search(text[max(0, m.start() - 120):m.start()]):
                continue
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
    for m in EMAIL.finditer(text):
        out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for m in URL.finditer(text):
        if "court.gov.ua" in m.group(0):
            continue
        out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for m in DATE_RE.finditer(text):
        ctx = text[max(0, m.start() - 45):m.start()] + " " + text[m.end():m.end() + 25]
        if BIRTH_CTX.search(ctx) or DEATH_CTX.search(text[max(0, m.start() - 30):m.start()]):
            out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for m in YEAR_BIRTH.finditer(text):
        out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for (_, pe) in pers:
        m = AFTER_PERSON.match(text, pe)
        if m:
            out.append((m.start(1), m.end(1), "ІНФОРМАЦІЯ"))
    for m in HANDLE.finditer(text):
        out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for m in NICK.finditer(text):
        out.append((m.start(1), m.end(1), "ІНФОРМАЦІЯ"))
    for m in SOCIAL_URL.finditer(text):
        out.append((m.start(), m.end(), "ІНФОРМАЦІЯ"))
    for m in ADDR_STREET.finditer(text):
        frag = text[m.start():m.end()].rstrip(" ,")
        pre = text[max(0, m.start() - 90):m.start()]
        if ORG_CTX.search(pre[-60:]) and not PERSONAL_CTX.search(pre):
            continue
        out.append((m.start(), m.start() + len(frag), "АДРЕСА"))
    for m in ADDR_NOSTREET.finditer(text):
        out.append((m.start(1), m.end(1), "АДРЕСА"))
    return out
