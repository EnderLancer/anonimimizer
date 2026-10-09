"""Conservative, explainable exclusions for the registry audit only."""
import re
from urllib.parse import urlsplit

import predict_trf as P

SERVICE_FILTER_VERSION = 4

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


def service_number_regions(text):
    """Only explicit labels prove a service category; proximity does not."""
    regions = []
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
    r"(?P<prosecutor>прокурор\w*)|(?P<secretary>секретар\w*))\b", re.I)
ROLE_NAMES = {"judge": "суддя", "lawyer": "адвокат / захисник",
              "notary": "нотаріус", "prosecutor": "прокурор", "secretary": "секретар"}
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
    r"адміністратив|нотаріаль|держав|приват|прокуратур|засідан|ськ)\w*|"
    r"україни|міста|м|києва|львова|одеси|харкова|та|і|у|в)$", re.I)
SUFFIX_GAP = re.compile(r"[\s,():—–-]*(?:(?:приватн\w*|державн\w*)\s+)?", re.I)
LIST_GAP = re.compile(r"\s*[,;]\s*(?:(?:та|і)\s+)?", re.I)
TOKENS = re.compile(r"[^\W\d_]+", re.U)
CONTACT = re.compile(
    r"(?:телефон\w*|тел\.|факс|email|e-mail|електронн\w+\s+пошт\w+)\s+"
    r"(?:суду|канцелярії|прокуратури|нотаріальної\s+контори)\s*[:—–-]?\s*$", re.I)
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


def mark_service_candidates(text, candidates):
    """Keep every candidate; attach exclusions without changing detector spans."""
    addresses = [c for c in candidates if c["type"] == "АДРЕСА"]
    people = [c for c in candidates if c["type"] == "ОСОБА"]
    number_regions = service_number_regions(text)
    protected_numbers = [m.span("value") for m in PRIVATE_NUMBER.finditer(text)]
    address_reasons = {(c["start"], c["end"]): reason for c in addresses
                       if (reason := service_address_reason(text, c["start"], c["end"], addresses, people))}
    # Prefer full names to overlapping NER fragments for context association.
    anchors = []
    for c in sorted(people, key=lambda c: -(c["end"] - c["start"])):
        if not any(c["start"] < a["end"] and a["start"] < c["end"] for a in anchors):
            anchors.append(c)
    anchors.sort(key=lambda c: c["start"])
    roles = list(ROLES.finditer(text))
    direct, ambiguous, participants = {}, set(), []
    previous = None
    for c in anchors:
        s, e = c["start"], c["end"]
        if participant_context(text, s, anchors):
            ambiguous.add(name_key(c["value"]))
            participants.append((s, e))
            previous = None
            continue
        role = None
        for m in reversed(roles):
            if m.end() > s:
                continue
            if s - m.end() > 160:
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
                    break
        if role is None:
            for m in roles:
                if e <= m.start() <= e + 60 and SUFFIX_GAP.fullmatch(normalize_role_context(text[e:m.start()])):
                    role = m.lastgroup
                    break
        if role is None and previous and LIST_GAP.fullmatch(normalize_role_context(text[previous[0]:s])):
            role = previous[1]
        if role:
            direct[(s, e)] = role
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
        pre = text[max(0, s - 140):s]
        if kind == "ОСОБА":
            key = name_key(value)
            # Reuse the anchor's decision for partial NER/CRF names, so a
            # participant prefix cannot disappear inside the longer name.
            if any(a < e and s < b for a, b in participants):
                continue
            role = next((r for (a, b), r in direct.items() if a <= s < e <= b), None)
            reason = "Ім’я безпосередньо пов’язане зі службовою роллю"
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
            # Exact host boundaries prevent court.gov.ua.evil.example matches.
            url = value.rstrip(".,;)")
            try:
                host = urlsplit(url if "://" in url else "//" + url).hostname or ""
            except ValueError:
                host = ""
            email_host = url.rsplit("@", 1)[-1].casefold() if "@" in url else ""
            if any(h == "court.gov.ua" or h.endswith(".court.gov.ua") for h in (host, email_host)):
                c.update(excluded=True, exclusion_reason="Офіційний сайт або email судової влади")
            elif CONTACT.search(pre) and not P.PERSONAL_CTX.search(pre):
                c.update(excluded=True, exclusion_reason="Контакт установи за явним контекстом")
    return candidates
