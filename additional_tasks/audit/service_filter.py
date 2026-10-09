"""Conservative, explainable exclusions for the registry audit only."""
import re
from urllib.parse import urlsplit

import predict_trf as P
from regex_persons import persons

SERVICE_FILTER_VERSION = 11

# Match labels and complete values in the original text, independently of
# detector boundaries. NER may return only a fragment of the full number.
NUMBER_VALUE = r"[A-ZА-ЯІЇЄҐ]{0,3}\d+(?:[/-][A-ZА-ЯІЇЄҐа-яіїєґ\d]+)*"
NUMBER_SEPARATOR = r"\s*(?:[:№N]|номер\w*)?\s*"
CASE_NUMBER = re.compile(
    rf"\bсправ(?:а|и|і|у|ою|ам|ами|ах)?{NUMBER_SEPARATOR}"
    rf"(?P<value>{NUMBER_VALUE})(?!\w|[/\d-]|[.:]\d)", re.I)
COURT_PROCEEDING = re.compile(
    rf"\b(?:судов\w+\s+проваджен\w*{NUMBER_SEPARATOR}|"
    rf"проваджен\w*{NUMBER_SEPARATOR}(?=\d+[/-]))"
    rf"(?P<value>{NUMBER_VALUE})(?!\w|[/\d-]|[.:]\d)", re.I)
DOCUMENT_NUMBER = re.compile(
    rf"\b(?:ухвал\w*|постанов\w*|наказ\w*)\s*№\s*"
    rf"(?P<value>{NUMBER_VALUE})(?!\w|[/\d-]|[.:]\d)", re.I)
INSTITUTION_PHONE = re.compile(
    r"\b(?:телефон\w*|тел\.|факс|fax)\s+"
    r"(?:суду|канцелярії|прокуратури|нотаріальної\s+контори)\s*[:—–-]?\s*"
    r"(?P<value>\+?\(?\d[\d\s().-]{5,}\d)(?!\d)", re.I)
# Some published payment blocks concatenate the IBAN and the next Ukrainian
# field label without whitespace. Still reject longer Latin/digit identifiers.
TREASURY_ACCOUNT = re.compile(r"(?<![A-Z\d])(?P<value>UA\d{2}(?:\s?\d){25})(?![A-Z\d])", re.I)
PRIVATE_NUMBER = re.compile(
    rf"\b(?:РНОКПП|ІПН|паспорт\w*|VIN|IMEI|ІМЕІ|"
    rf"(?:виконавч\w+|кримінальн\w+|досудов\w+)\s+проваджен\w*|ВП|кадастров\w+\s+номер\w*|"
    rf"номер\w*\s+запис\w*|договор\w*|контракт\w*|квитанці\w*)"
    rf"{NUMBER_SEPARATOR}(?P<value>{NUMBER_VALUE})(?!\w|[/\d-])", re.I)

LETTERHEAD_COURT = re.compile(
    r"(?:[А-ЯІЇЄҐ’'ʼ-]+[ \t]+)*(?:СУД|СУД УКРАЇНИ)"
    r"(?:[ \t]+[А-ЯІЇЄҐ’'ʼ-]+)*")
LETTERHEAD_PHONE = re.compile(
    r"\b(?:телефон|тел\.|факс|fax)\s*[:—–-]?\s*"
    r"(?P<value>\+?\(?\d(?:[\d ().-]*\d)?)(?!\w)", re.I)


def letterhead_phone_regions(text):
    """Bind contacts to a standalone court heading and its contact lines only."""
    regions = []
    position = 0
    court_seen = False
    for line in text.splitlines(keepends=True):
        start = position
        position += len(line)
        if position > 1000:
            break
        value = line.strip()
        if not value:
            continue
        if not court_seen:
            if value == "УКРАЇНА":
                continue
            if (not LETTERHEAD_COURT.fullmatch(value)
                    or not re.search(r"(?:ГОСПОДАРСЬКИЙ|РАЙОННИЙ|МІСЬКИЙ|АПЕЛЯЦІЙНИЙ|"
                                     r"ОКРУЖНИЙ|ВЕРХОВНИЙ|АДМІНІСТРАТИВНИЙ)\b", value)):
                break
            court_seen = True
            continue
        # A split regional name belongs to the heading, not to the body.
        if re.fullmatch(r"[А-ЯІЇЄҐ’'ʼ -]+ ОБЛАСТІ", value):
            continue
        # Stop at prose, participant details or a document heading. Blank
        # lines are allowed because published letterheads often contain them.
        if (PARTY.search(value) or P.PERSONAL_CTX.search(value)
                or re.search(r"\b(?:адвокат\w*|судд\w*|представник\w*|особист\w*|"
                             r"мобільн\w*|розглянув\w*|повідом\w*|встанов\w*)\b", value, re.I)):
            break
        if not re.match(r"(?:\d{5}\s*,?\s*|вул\.|вулиця\b|майдан\b|"
                        r"проспект\b|площа\b|м\.|тел\.|телефон\b|"
                        r"факс\b|fax\b|E-mail\b|email\b)", value, re.I):
            break
        for match in LETTERHEAD_PHONE.finditer(line):
            if 6 <= len(re.sub(r"\D", "", match.group("value"))) <= 12:
                a, b = match.span("value")
                regions.append((start + a, start + b,
                                "Телефон або факс суду в блоці реквізитів шапки"))
    return regions


def service_number_regions(text):
    """Only explicit labels prove a service category; proximity does not."""
    regions = letterhead_phone_regions(text)
    for pattern, reason in (
        (CASE_NUMBER, "Номер судової справи"),
        (COURT_PROCEEDING, "Номер судового провадження"),
        (DOCUMENT_NUMBER, "Номер ухвали, постанови або наказу"),
        (INSTITUTION_PHONE, "Телефон або факс установи за явним контекстом"),
        (TREASURY_ACCOUNT, "Рахунок казначейства (код банку 899998)"),
    ):
        for match in pattern.finditer(text):
            value = match.group("value")
            if pattern is TREASURY_ACCOUNT and not P.treasury_iban(value):
                continue
            if pattern is INSTITUTION_PHONE and not 7 <= len(re.sub(r"\D", "", value)) <= 12:
                continue
            regions.append((*match.span("value"), reason))
    return regions


def service_number_reason(text, start, end, regions=None, protected=None):
    """Classify an occurrence; ambiguous or crossing fragments stay visible."""
    if protected is None:
        protected = [m.span("value") for m in PRIVATE_NUMBER.finditer(text)]
    if any(a < end and start < b for a, b in protected):
        return None
    if regions is None:
        regions = service_number_regions(text)
    return next((reason for a, b, reason in regions if a <= start < end <= b), None)

ROLES = re.compile(
    r"\b(?:(?P<judge>суддя-доповідач\w*|судд[яіюеївямх]*|головуюч\w*)|"
    r"(?P<lawyer>адвокат\w*|захисник\w*)|(?P<notary>нотаріус\w*)|"
    r"(?P<prosecutor>прокурор\w*)|(?P<secretary>секретар\w*)|(?P<juror>присяжн\w*))\b", re.I)
ROLE_NAMES = {"judge": "суддя", "lawyer": "адвокат / захисник",
              "notary": "нотаріус", "prosecutor": "прокурор", "secretary": "секретар",
              "juror": "присяжний"}
PARTY = re.compile(r"\b(?:позивач\w*|відповідач\w*|обвинувачен\w*|підозрюван\w*|"
                  r"засуджен\w*|потерпіл\w*|боржник\w*|свідок|свідк\w*|"
                  r"заявни[кц]\w*|стягувач\w*|скаржни[кц]\w*|"
                  r"трет(?:я|ю|ьої|ій|ьою|і|іх|ім|іми)\s+ос(?:об\w*|іб))\b", re.I)
REPRESENTATIVE = re.compile(
    r"\b(?:представни[кц]\w*|адвокат\w*|захисник\w*)"
    r"(?:\s+інтересів)?\s*$", re.I)
# Only institution descriptions may separate a role and its name. A new
# clause, participant role, or arbitrary prose must not bridge that relation.
GAP = re.compile(r"[А-ЯІЇЄҐа-яіїєґ'’ʼ\s,:()—–-]*")
INSTITUTION_WORD = re.compile(
    r"^(?:\w*(?:суд|район|облас|міськ|округ|апеляці|касаці|верхов|господар|"
    r"адміністратив|нотаріаль|держав|приват|прокуратур|засідан|ськ|цьк)\w*|"
    r"україни|міста|м|києва|львова|одеси|харкова|та|і|у|в)$", re.I)
SUFFIX_GAP = re.compile(r"[\s,():—–-]*(?:(?:приватн\w*|державн\w*)\s+)?", re.I)
LIST_GAP = re.compile(r"\s*[,;]\s*(?:(?:та|і)\s+)?", re.I)
TOKENS = re.compile(r"[^\W\d_]+", re.U)
ADDRESS = re.compile(
    r"(?:адрес\w*\s+(?:суду|прокуратури|банку|канцелярії|нотаріальної\s+контори)|"
    r"(?:суд|прокуратура|банк|нотаріальна\s+контора)\s+(?:знаходиться|розташован\w*)"
    r"\s+за\s+адресою)\s*[:—–-]?\s*$", re.I)
INSTITUTION_ADDRESS = re.compile(
    r"\b(?:суд(?:у|і|ом|а|ові)?|прокуратур\w*|банк\w*|канцелярі\w*|"
    r"нотаріальн\w+\s+контор\w*|(?:сільськ\w+|міськ\w+|селищн\w+)\s+рад\w*|"
    r"(?:головн\w+\s+)?управлінн\w*|департамент\w*|"
    r"відділ\w*\s+(?:державн\w+\s+)?реєстраці\w*|"
    r"інститут\w*\s+судових\s+експертиз|експертно-криміналістичн\w+\s+центр\w*|"
    r"центр\w*\s+з\s+надання\s+безоплатн\w+|казначейств\w*|"
    r"товариств\w*\s+з\s+обмеженою\s+відповідальністю|"
    r"(?:приватн\w+|державн\w+)\s+підприємств\w*|ТОВ|ПАТ|ПрАТ|АТ)\b", re.I)
# An institution mention must be linked to this address, not merely nearby.
ADDRESS_LINK = re.compile(
    r"(?:\(\s*(?:адрес\w*\s*:\s*)?|[,;]?\s*"
    r"(?:(?:юридичн\w+|поштов\w+)\s+)?(?:адрес\w*|місцезнаходження)\s*[:—–-]\s*|"
    r"(?:за\s+адресою)\s*:?\s*)$", re.I)
ADDRESS_NARRATIVE = re.compile(
    r"\b(?:прожив\w*|мешка\w*|жител\w*|квартир\w*|будин\w*|"
    r"майн\w*|земельн\w*|ділянк\w*|склад\w*|обшу\w*|"
    r"встановив|встановлено|позов\w*|позивач\w*|відповідач\w*|"
    r"зобов\w*|стягну\w*|повідомив|зазначив|відомо)\b", re.I)
ADDRESS_PERSON = re.compile(
    r"\b(?:ОСОБА_\d+|адвокат\w*|судд\w*|нотаріус\w*|"
    r"капітан\w*|інспектор\w*|воді\w*|начальник\w*)\b", re.I)
WORK_ADDRESS = re.compile(r"\bробоч\w+\s+адрес\w*[^;.!?]{0,100}[:—–-]\s*$", re.I)
HEARING_ADDRESS = re.compile(
    r"\b(?:судов\w+\s+засідан\w*|залі\s+суду)"
    r"[^;!?]{0,180}\bза\s+адресою\s*:?\s*$", re.I)

# Information exclusions use full source regions, not the boundaries returned
# by NER/CRF. Context is anchored to a value; mere proximity is not evidence.
EMAIL_LABEL = r"(?:email|e-mail|е-пошта|е-пош\.|електронн\w+\s+пошт\w+)"
INFORMATION_CONTACT = re.compile(
    rf"\b(?:{EMAIL_LABEL}|офіційн\w+\s+(?:сайт|вебсайт))\s+"
    r"(?:суду|канцелярії|прокуратури|нотаріальної\s+контори|"
    r"міністерства|департаменту|управління|міської\s+ради)\s*[:—–-]?\s*$", re.I)
INSTITUTION_EMAIL_LABEL = re.compile(rf"\b{EMAIL_LABEL}\s*[:—–-]?\s*$", re.I)
INSTITUTION_CONTACT_OWNER = re.compile(
    r"(?<!\w)(?:[А-ЯІЇЄҐ][\w’'ʼ-]+\s+){0,4}"
    r"(?i:(?:(?:окружна|районна|обласна|генеральна)\s+)?прокуратура|"
    r"(?:сільська|селищна|міська|районна|обласна)\s+рада|"
    r"(?:господарський|районний|міський|окружний|апеляційний|верховний)\s+суд)"
    r"\s*\(")
INFORMATION_PRIVATE = re.compile(
    r"\b(?:особист\w*|приватн\w+\s+(?:email|e-mail|пошт\w*|акаунт\w*)|"
    r"ОСОБА_\d+|РНОКПП|паспорт\w*|народжен\w*|народив\w*|нар\.|уродж\w*|"
    r"помер\w*|смерт\w*|загинув\w*|акаунт\w*|логін\w*|нікнейм\w*)", re.I)
INFORMATION_DATE_PREFIX = re.compile(
    r"\b(?:дата\s+(?:ухвалення|прийняття|складання|виготовлення)\s+"
    r"(?:рішення|ухвали|постанови)|дата\s+судового\s+засідання|"
    r"(?:рішення|ухвала|постанова|наказ|закон)\s+від|"
    r"судове\s+засідання\s+(?:призначено\s+на|відбудеться))\s*[:—–-]?\s*$", re.I)
INFORMATION_ORG_PREFIX = re.compile(
    r"\b(?:(?:найменування|назва)\s+(?:суду|установи|органу)|орган,?\s+що\s+видав\s+документ|"
    r"документ\s+видано|видано\s+органом)\s*[:—–-]?\s*$", re.I)
INFORMATION_MILITARY = re.compile(
    r"\b(?:ТЦК|військов\w*|комплектування|прикордонн\w*)\b", re.I)
# A structural name (ministry/department + country/region) also needs a
# public-body marker. "Головне управління" by itself does not prove ownership.
PUBLIC_AUTHORITY_MARKER = re.compile(
    r"\b(?:Міністерств\w+|Державн\w+|Національн\w+|"
    r"Пенсійн\w+\s+фонд\w+|Держ[а-яіїєґ]+|ДПС)\b", re.I)
PUBLIC_AUTHORITY_NAME = re.compile(P.ORG_GOV.pattern, P.ORG_GOV.flags | re.I)
PUBLIC_AUTHORITY_UNSAFE = re.compile(
    r"\b(?:ТОВ|ПП|АТ|ПАТ|ПрАТ|компані\w*|приватн\w*|"
    r"працівник\w*|співробітник\w*|керівник\w*|директор\w*|"
    r"телефон\w*|email|паспорт\w*|РНОКПП)\b", re.I)
AUTHORITY_REGION_SUFFIX = re.compile(
    r"\s+(?:в|у)\s+(?:[а-яіїєґ’'ʼ-]+\s+){1,3}?област\w*\b", re.I)
TCC_NAME = re.compile(
    r"(?<![\w-])(?:[А-ЯІЇЄҐ][\w’'ʼ-]+\s+)?"
    r"(?i:(?:районн\w+\s+)?територіальн\w+\s+центр\w*\s+"
    r"комплектування\s+та\s+соціальної\s+підтримки|ТЦК(?:\s+та\s+СП)?)"
    r"(?![\w-])")
COURT_NAME = re.compile(
    r"(?<!\w)(?:Верховн\w+\s+Суд\w*|(?:[А-ЯІЇЄҐ][\w’'ʼ-]+\s+){1,5}"
    r"(?:районн\w+|міськ\w+|апеляційн\w+|окружн\w+|господарськ\w+|"
    r"адміністративн\w+)\s+суд\w*)(?!\w)")
BARE_COURT_URL = re.compile(
    r"(?<![\w@./-])(?:[\w-]+\.)*(?:court|arbitr)\.gov\.ua(?![\w:-]|\.[\w-])"
    r"(?:[/?#][^\s,;<>\"«»)]*)?", re.I)


def information_context(text, start, end):
    """Use the local clause, preserving dots in dates and domain names."""
    pre = re.split(r"[;!?]|\n\s*\n|\.\s+(?=[А-ЯІЇЄҐA-Z])",
                   text[max(0, start - 180):start])[-1]
    after = re.split(r"[;!?]|\n|\.\s", text[end:end + 80])[0]
    return pre, after


def private_information_context(text, start, end):
    pre, after = information_context(text, start, end)
    # Postposed birth markers are common: "12.03.1980 року народження".
    birth_suffix = re.match(r"\s*(?:(?:року|р\.)\s*)?(?:народження|р\.?\s*н\.)", after, re.I)
    return bool(INFORMATION_PRIVATE.search(pre) or PARTY.search(pre)
                or P.PERSONAL_CTX.search(pre) or birth_suffix
                or PARTY.search(after) or INFORMATION_PRIVATE.search(after))


def institution_contact_blocks(text):
    """Bound contact ownership to the institution's balanced parentheses.

    An institution can itself be a plaintiff. That outer role does not make
    its general mailbox personal; individual roles inside the block still do.
    """
    blocks = []
    for match in INSTITUTION_CONTACT_OWNER.finditer(text):
        depth = 1
        for end in range(match.end(), min(len(text), match.end() + 1200)):
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
                if depth == 0:
                    blocks.append((match.end(), end))
                    break
    return blocks


def institution_email_reason(text, start, end, blocks):
    for a, b in blocks:
        if not a <= start < end <= b:
            continue
        # Inspect only this owner block, not the party title outside it.
        body = text[a:b]
        if (INFORMATION_PRIVATE.search(body) or PARTY.search(body)
                or P.PERSONAL_CTX.search(body)):
            continue
        if INSTITUTION_EMAIL_LABEL.search(text[a:start]):
            return "Email установи у блоці її реквізитів"
    return None


def official_information_link(value, is_email=False):
    """Validate complete hosts; do not hide URLs carrying personal parameters."""
    if is_email:
        host = value.rsplit("@", 1)[-1].casefold()
        if any(host == root or host.endswith("." + root)
               for root in ("court.gov.ua", "arbitr.gov.ua")):
            return "Офіційний сайт або email судової влади"
        return None
    try:
        parsed = urlsplit(value if "://" in value else "//" + value)
        host = parsed.hostname or ""
        # Login links, user info and query parameters can identify a person.
        if parsed.username is not None or parsed.port not in (None, 80, 443) or parsed.query:
            return None
    except ValueError:
        return None
    if re.search(r"(?:account|profile|user|login|cabinet|кабінет)", parsed.path, re.I):
        return None
    if any(host == root or host.endswith("." + root)
           for root in ("court.gov.ua", "arbitr.gov.ua")) and not parsed.fragment:
        return "Офіційний сайт або email судової влади"
    if host == "zakon.rada.gov.ua" and re.fullmatch(r"/laws/show/[\w%./-]+", parsed.path):
        return "Посилання на нормативний акт на офіційному сайті"
    if (host == "gov.ua" or host.endswith(".gov.ua")) and parsed.path in ("", "/") and not parsed.fragment:
        return "Офіційний сайт державної установи"
    return None


def service_information_regions(text):
    """Find only explicit service information; uncertain values stay visible."""
    regions = []
    contact_blocks = institution_contact_blocks(text)
    full_urls = list(P.URL.finditer(text))
    urls = full_urls + list(BARE_COURT_URL.finditer(text))
    for pattern, is_email in ((P.EMAIL, True), (P.URL, False), (BARE_COURT_URL, False)):
        for match in pattern.finditer(text):
            if is_email and any(m.start() <= match.start() < m.end() for m in urls):
                continue
            if pattern is BARE_COURT_URL and any(m.start() <= match.start() < m.end() for m in full_urls):
                continue
            value = match.group().rstrip(".,;)")
            # P.URL includes final sentence dots; keep that source boundary so
            # raw URL candidates and their contained fragments agree.
            start, end = match.span()
            if is_email:
                reason = institution_email_reason(text, start, end, contact_blocks)
                if reason:
                    regions.append((start, end, reason))
                    continue
            if private_information_context(text, start, end):
                continue
            reason = official_information_link(value, is_email)
            pre, _ = information_context(text, start, end)
            if reason is None and INFORMATION_CONTACT.search(pre):
                # Explicit institutional context permits an email on another
                # domain. For links, still reject user data and spoofed hosts.
                if is_email:
                    reason = "Контакт установи за явним контекстом"
            if reason:
                regions.append((start, end, reason))
    # A model can label an institutional phone as INFORMATION rather than
    # NUMBER. Keep the same explicit contact rule for either label.
    for start, end, reason in service_number_regions(text):
        if (reason in ("Телефон або факс установи за явним контекстом",
                       "Телефон або факс суду в блоці реквізитів шапки")
                and not private_information_context(text, start, end)):
            regions.append((start, end, "Контакт установи за явним контекстом"))
    for match in P.DATE_ALL_RE.finditer(text):
        start, end = match.span()
        pre, _ = information_context(text, start, end)
        if not private_information_context(text, start, end) and INFORMATION_DATE_PREFIX.search(pre):
            regions.append((start, end, "Дата процесуального або офіційного документа / засідання"))
    regions.extend(service_authority_regions(text))
    for pattern in (P.ORG_GOV, COURT_NAME):
        for match in pattern.finditer(text):
            start, end = match.span()
            if (private_information_context(text, start, end)
                    or INFORMATION_MILITARY.search(match.group())):
                continue
            pre, after = information_context(text, start, end)
            # A postposed participant role also makes an institution ambiguous.
            if re.match(r"\s*[,—–:-]?\s*(?:позивач|відповідач|заявник|скаржник)\w*\b", after, re.I):
                continue
            if INFORMATION_ORG_PREFIX.search(pre):
                regions.append((start, end, "Назва установи в реквізитах або як органу видачі документа"))
            else:
                line_start = text.rfind("\n", 0, start) + 1
                line_end = text.find("\n", end)
                if line_end == -1:
                    line_end = len(text)
                # Only a standalone first line proves letterhead, not a party
                # mention anywhere in the first 240 characters.
                if (not text[:line_start].strip() and not text[line_start:start].strip()
                        and not text[end:line_end].strip(" \t\r.:—–-")
                        and line_end < len(text)):
                    regions.append((start, end, "Назва установи в шапці документа"))
    return regions


def service_authority_regions(text):
    """Recognize classes of public names; party roles do not change the name.

    P.ORG_GOV supplies grammatical boundaries and case forms. Public markers
    distinguish governmental management from management of private companies.
    """
    regions = []
    for pattern, reason in (
        (PUBLIC_AUTHORITY_NAME, "Назва держоргану за структурою та ознаками державної установи"),
        (TCC_NAME, "Назва територіального центру комплектування та соціальної підтримки"),
    ):
        for match in pattern.finditer(text):
            start, end = match.span()
            if pattern is PUBLIC_AUTHORITY_NAME:
                suffix = AUTHORITY_REGION_SUFFIX.match(text, end)
                if suffix:
                    end = suffix.end()
                if not PUBLIC_AUTHORITY_MARKER.search(text[start:end]):
                    continue
                # TCC is explicitly allowed by audit policy; other military
                # formations retain the conservative treatment.
                if INFORMATION_MILITARY.search(text[start:end]):
                    continue
            value = text[start:end]
            if (PUBLIC_AUTHORITY_UNSAFE.search(value) or INFORMATION_PRIVATE.search(value)
                    or text[max(0, start - 1):start] in ('«', '"')):
                continue
            regions.append((start, end, reason))
    return regions


def service_information_reason(text, start, end, regions=None):
    """Classify this occurrence, including contained model fragments."""
    if regions is None:
        regions = service_information_regions(text)
    return next((reason for a, b, reason in regions if a <= start < end <= b), None)


def service_address_reason(text, start, end, addresses, people=()):
    """Classify each occurrence; never propagate by matching address values."""
    left = max(0, start - 400)
    # Stop at another detected address, including one already anonymized.
    left = max([left] + [c["end"] for c in addresses if c["end"] <= start])
    placeholders = list(re.finditer(r"АДРЕСА_\d+", text[left:start]))
    if placeholders:
        left += placeholders[-1].end()
    pre = text[left:start]
    pre = re.split(r"[;!?]|\n\s*\n", pre)[-1]
    if P.PERSONAL_CTX.search(pre) or re.search(r"\b(?:жител\w*|мешкан\w*)\b", pre, re.I):
        return None
    if ADDRESS.search(pre):
        return "Адреса установи за явним контекстом"
    if WORK_ADDRESS.search(pre):
        return "Явно зазначена робоча адреса"
    if HEARING_ADDRESS.search(pre):
        return "Адреса проведення судового засідання"
    for match in reversed(list(INSTITUTION_ADDRESS.finditer(pre))):
        gap = pre[match.end():]
        if (len(gap) > 260 or ADDRESS_NARRATIVE.search(gap)
                or ADDRESS_PERSON.search(gap)
                or re.search(r"\b(?:близько|знаходячись|перебуваючи)\b", gap, re.I)):
            continue
        absolute_end = start - len(pre) + match.end()
        if any(absolute_end <= c["start"] and c["end"] <= start for c in people):
            continue
        # A preceding sentence or closed parenthesis breaks the association.
        if re.search(r"[)]|\.\s+(?=[А-ЯІЇЄҐ][а-яіїєґ])", gap):
            continue
        if ADDRESS_LINK.search(gap):
            return "Адреса установи за явним контекстом"
        # Letterhead: institution name and address on consecutive lines at
        # the start of a document. Also allows NER to include the region name.
        if start < 240 and re.fullmatch(r"[\w\s.'’ʼ—–-]*", gap):
            if "\n" in pre or "\n" in text[start:end]:
                return "Адреса установи в шапці документа"
    return None


def name_key(value):
    return tuple(TOKENS.findall(value.casefold().replace("’", "'").replace("ʼ", "'")))


def normalize_role_context(value):
    """Normalize a copy only: candidate offsets always refer to original text."""
    return re.sub(r"\bм\.\s*", "м ", value.replace("|", " "), flags=re.I)


def role_gap(value):
    # Alignment padding is common in signatures. Measure meaningful context,
    # while keeping all offsets in the original text unchanged.
    value = re.sub(r"\s+", " ", value)
    if len(value) > 160:
        return False
    value = normalize_role_context(value)
    words = TOKENS.findall(value)
    return (GAP.fullmatch(value) is not None
            and all(INSTITUTION_WORD.fullmatch(w)
                    or (i > 0 and words[i - 1].lower() in ("м", "міста") and w[0].isupper())
                    for i, w in enumerate(words)))


def participant_context(text, start, anchors):
    """Associate a participant title with this name, not a nearby person.

    Newlines and the same city abbreviation accepted by role_gap are not
    clause boundaries. A previous detected name or a sentence boundary is.
    """
    left = max(0, start - 240)
    left = max([left] + [a["end"] for a in anchors if a["end"] <= start])
    context = normalize_role_context(text[left:start])
    clause = re.split(r"[;.!?]", context)[-1]
    for match in reversed(list(PARTY.finditer(clause))):
        # In "представник позивача" the participant title belongs to the
        # client; the following name belongs to their representative.
        if REPRESENTATIVE.search(clause[:match.start()]):
            continue
        gap = clause[match.end():]
        # Professional titles do not make a participant's name service data.
        # Reuse role_gap to accept line breaks, cities and table separators,
        # while rejecting unrelated prose between the title and the name.
        if len(gap) <= 200 and role_gap(ROLES.sub("", gap)):
            return True
    return False


def service_person_list(text, start, end, direct):
    """Prove that a merged model span contains only names with one service role."""
    parts = sorted((max(start, a), min(end, b), role)
                   for (a, b), role in direct.items() if a < end and start < b)
    if len(parts) < 2 or parts[0][0] != start or parts[-1][1] != end:
        return None
    roles = {part[2] for part in parts}
    if len(roles) != 1:
        return None
    role = parts[0][2]
    for left, right in zip(parts, parts[1:]):
        gap = normalize_role_context(text[left[1]:right[0]])
        if not (LIST_GAP.fullmatch(gap) or (role == "juror" and "\n" in gap and not gap.strip())):
            return None
    return role


def mark_service_candidates(text, candidates):
    """Keep every candidate; attach exclusions without changing detector spans."""
    addresses = [c for c in candidates if c["type"] == "АДРЕСА"]
    people = [c for c in candidates if c["type"] == "ОСОБА"]
    number_regions = service_number_regions(text)
    protected_numbers = [m.span("value") for m in PRIVATE_NUMBER.finditer(text)]
    information_regions = service_information_regions(text)
    address_reasons = {(c["start"], c["end"]): reason for c in addresses
                       if (reason := service_address_reason(text, c["start"], c["end"], addresses, people))}
    # Names matched in the source take precedence over model fragments that
    # can include titles, several people or the next paragraph.
    anchors = []
    source_names = [{"start": s, "end": e, "value": text[s:e]} for s, e, _ in persons(text)]
    for c in source_names + sorted(people, key=lambda c: -(c["end"] - c["start"])):
        if not any(c["start"] < a["end"] and a["start"] < c["end"] for a in anchors):
            anchors.append(c)
    anchors.sort(key=lambda c: c["start"])
    roles = list(ROLES.finditer(text))
    direct, ambiguous, participants = {}, set(), []
    role_regions = []
    previous = None
    for c in anchors:
        s, e = c["start"], c["end"]
        if participant_context(text, s, anchors):
            ambiguous.add(name_key(c["value"]))
            participants.append((s, e))
            previous = None
            continue
        role = None
        role_start = None
        for m in reversed(roles):
            if m.end() > s:
                continue
            if s - m.end() > 500:
                break
            gap = text[m.end():s]
            # "Адвокат позивача ПІБ" names the lawyer, not the plaintiff.
            # participant_context has already protected the reverse order.
            if m.lastgroup == "lawyer":
                gap = PARTY.sub("", gap)
            if role_gap(gap):
                # Never attach a role across another detected person.
                if not any(m.end() <= a["start"] < s for a in anchors):
                    role = m.lastgroup
                    role_start = m.start()
                    break
        if role is None:
            for m in roles:
                if e <= m.start() <= e + 60 and SUFFIX_GAP.fullmatch(normalize_role_context(text[e:m.start()])):
                    role = m.lastgroup
                    break
        if role is None and previous:
            gap = normalize_role_context(text[previous[0]:s])
            if LIST_GAP.fullmatch(gap) or (previous[1] == "juror" and "\n" in gap and not gap.strip()):
                role = previous[1]
        if role:
            direct[(s, e)] = role
            if role_start is not None:
                role_regions.append((role_start, s, e, role))
            previous = (e, role)
        else:
            previous = None

    known = {}
    for c in anchors:
        key = name_key(c["value"])
        role = direct.get((c["start"], c["end"]))
        if role and key not in ambiguous:
            known.setdefault(key, set()).add(role)
    for c in candidates:
        c.update(excluded=False, exclusion_reason=None, role=None)
        s, e, kind, value = c["start"], c["end"], c["type"], c["value"]
        if kind == "ОСОБА":
            key = name_key(value)
            # Reuse the anchor's decision for partial NER/CRF names, so a
            # participant prefix cannot disappear inside the longer name.
            if any(a < e and s < b for a, b in participants):
                continue
            role = next((r for (a, b), r in direct.items() if a <= s < e <= b), None)
            if role is None:
                # A title can be inside the candidate, but trailing prose or
                # another name must not be hidden along with the signature.
                role = next((r for a, name_start, b, r in role_regions
                             if a <= s < e <= b and e > name_start), None)
            reason = "Ім’я безпосередньо пов’язане зі службовою роллю"
            if role is None:
                role = service_person_list(text, s, e, direct)
                if role:
                    reason = "Фрагмент містить лише імена службових осіб однієї ролі"
            if role is None and len(key) >= 2 and len(known.get(key, ())) == 1:
                role = next(iter(known[key]))
                reason = "Повтор точно такого самого службового імені в цьому рішенні"
            if role:
                c.update(excluded=True, exclusion_reason=reason, role=ROLE_NAMES[role])
        elif kind == "АДРЕСА":
            reason = next((reason for (a, b), reason in address_reasons.items()
                           if a <= s < e <= b), None)
            if reason:
                c.update(excluded=True, exclusion_reason=reason)
        elif kind == "НОМЕР":
            reason = service_number_reason(text, s, e, number_regions, protected_numbers)
            if reason:
                c.update(excluded=True, exclusion_reason=reason)
        elif kind == "ІНФОРМАЦІЯ":
            reason = service_information_reason(text, s, e, information_regions)
            if reason:
                c.update(excluded=True, exclusion_reason=reason)
    return candidates
