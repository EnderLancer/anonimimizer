"""Conservative, explainable exclusions for the registry audit only."""
import re
from urllib.parse import urlsplit

import predict_trf as P

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
    people = [c for c in candidates if c["type"] == "ОСОБА"]
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
            if ADDRESS.search(pre) and not P.PERSONAL_CTX.search(pre):
                c.update(excluded=True, exclusion_reason="Адреса установи за явним контекстом")
        elif kind == "НОМЕР":
            compact = re.sub(r"\s", "", value)
            if re.fullmatch(r"UA\d{27}", compact, re.I) and P.treasury_iban(compact):
                c.update(excluded=True, exclusion_reason="Рахунок казначейства (код банку 899998)")
            elif P.NUM_NEG.search(pre[-70:]):
                c.update(excluded=True, exclusion_reason="Неособистий номер за контекстом документа")
            elif CONTACT.search(pre) and not P.PERSONAL_CTX.search(pre):
                c.update(excluded=True, exclusion_reason="Контакт установи за явним контекстом")
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
