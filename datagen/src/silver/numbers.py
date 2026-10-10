"""Layers 2–3: open identifiers → НОМЕР labels or masks."""
from __future__ import annotations

from silver import patterns as P
from silver.merge import any_overlap


def _free(span: dict, *lists: list[dict]) -> bool:
    return not any(any_overlap(span, lst) for lst in lists)


# An open enforcement-proceeding number is personal data only when the proceeding concerns a natural person
# (task statement: НОМЕР = identifiers of a specific natural person). «Person visible» = a party (vp_parties: a filled ОСОБА
# placeholder or an open name after a party word) within VP_PERSON_NEAR characters of the number, or in the first
# VP_PERSON_HEAD characters (the heading names the participants).
# Audit of 159 hand-read open numbers with these distances: TP 127, FP 1, FN 0, TN 5 (precision 0.992, recall 1.000);
# on the corpus the test leaves 106 of 606 open numbers without a person, mostly in commercial courts.
# Without the heading clause and a 500-character window the recall falls to 0.843.
VP_PERSON_NEAR = 300
VP_PERSON_HEAD = 1500


# Known cost of the heading clause: a heading whose only hidden persons are third parties or a representative of a legal
# entity makes the numbers of a proceeding between legal entities personal (decisions 121902057 and 120114987: 7 of 614
# labelled numbers). Not filtered: «третя особа ОСОБА_1» is as often the debtor, and «представника ОСОБА_1» names the party.
def person_visible(persons: list[dict], start: int, end: int) -> bool:
    return any(p["start"] < VP_PERSON_HEAD or (p["end"] > start - VP_PERSON_NEAR and p["start"] < end + VP_PERSON_NEAR)
               for p in persons)


def vp_parties(text: str, existing: list[dict]) -> list[dict]:
    """The natural persons that make an enforcement-proceeding number personal data."""

    def open_party(start: int) -> bool:
        return (bool(P.VP_PARTY_LEFT.search(text, max(0, start - 60), start))
                and not P.VP_THIRD_LEFT.search(text, max(0, start - 200), start))

    # Persons the clerk hid (filled placeholders) are the participants. The names left open are mostly officials (the judge
    # in every heading, the executor next to nearly every number) and would make the test always true; an open name counts
    # only right after a party word: «Фізична особа-підприємець Харченко Віктор Леонідович», «боржник Коваль І. П.»
    return [l for l in existing if l["type"] == "ОСОБА" and (l["source"] == "placeholder" or open_party(l["start"]))]


def vp_spans(text: str, existing: list[dict]) -> list[tuple[int, int]]:
    """Every open number that follows an enforcement-proceeding cue, in text order."""
    spans = {sp for m in P.VP.finditer(text) for sp in m.spans("n")}
    # «виконавчі провадження: № НОМЕР_1, № 58365604, № 61493933»: the clerk hid the first number of the list. The pattern
    # sees the list only when the fill value happens to be 7–10 digits, so the list is continued from the placeholder
    for l in existing:
        if l["type"] == "НОМЕР" and P.VP_LIST_HEAD.search(text, max(0, l["start"] - P.VP_HEAD_WINDOW), l["start"]):
            m = P.VP_LIST_MORE.match(text, l["end"])
            if m:
                spans.update(m.spans("n"))
    return sorted(spans)


def label_numbers(text: str, existing: list[dict], masks: list[dict]) -> tuple[list[dict], list[dict], dict]:
    labels: list[dict] = []
    new_masks: list[dict] = []
    stats = {"iban_person": 0, "iban_institution": 0, "iban_unknown": 0}

    def lab(s, e, source):
        sp = {"start": s, "end": e, "type": "НОМЕР", "pseudonym": None, "source": source}
        if _free(sp, existing, labels):
            labels.append(sp)

    def mask(s, e, reason):
        sp = {"start": s, "end": e, "reason": reason}
        if _free(sp, existing, labels, masks, new_masks):
            new_masks.append(sp)

    for m in P.RNOKPP.finditer(text):
        lab(m.start(1), m.end(1), "rnokpp")
    for m in P.UNZR.finditer(text):
        lab(m.start(1), m.end(1), "unzr")
    # local courts leave most of these numbers open, Supreme Court clerks hide 93 % of them; without a natural person
    # in sight the number is masked: neither taught as personal data nor as «not personal data»
    parties = vp_parties(text, existing)
    # a filled placeholder or a span that is masked already is left alone
    open_vp = [(s, e) for s, e in vp_spans(text, existing) if _free({"start": s, "end": e}, existing, masks)]
    # one number is one proceeding: when a person is visible at one occurrence the number is labelled at every occurrence,
    # otherwise the exported text would replace it in one place and leave it open in another
    personal = {text[s:e] for s, e in open_vp if person_visible(parties, s, e)}
    for s, e in open_vp:
        if text[s:e] in personal:
            lab(s, e, "vp")
        else:
            mask(s, e, "vp_no_person")
    for m in P.PASSPORT_SN.finditer(text):
        left = text[max(0, m.start() - 160): m.start()]
        if P.PASSPORT_CTX.search(left) and not P.NOT_PASSPORT_LEFT.search(left[-30:]) and not P.NOT_PASSPORT_CTX.search(left[-90:]):
            lab(m.start(1), m.end(1), "passport")
        else:
            mask(m.start(1), m.end(1), "series_no_ctx")
    for m in P.ID_CARD.finditer(text):
        if P.ID_CTX.search(text[max(0, m.start() - 60): m.start()]):
            lab(m.start(1), m.end(1), "id_card")
    for rx, source in ((P.IBAN, "iban"), (P.CARD16, "card")):
        for m in rx.finditer(text):
            if any_overlap({"start": m.start(), "end": m.end()}, existing):
                continue  # already a filled placeholder
            if source == "card" and not P.CARD_CTX.search(text[max(0, m.start() - 50): m.start()]):
                continue  # 16-digit ЄРДР case numbers etc. are not card numbers
            left, right = text[max(0, m.start() - 200): m.start()], text[m.end(): m.end() + 120]
            inst = P.IBAN_INSTITUTION.search(left[-120:] + " " + right) or P.IBAN_INSTITUTION.search(m.group())
            pers = P.IBAN_PERSON.search(left + " " + right) or any(
                l["type"] == "ОСОБА" and m.start() - 200 <= l["start"] <= m.end() + 120 for l in existing)
            if pers and not inst:
                lab(m.start(), m.end(), source)
                stats["iban_person"] += 1
            elif inst and not pers:
                stats["iban_institution"] += 1
            else:
                mask(m.start(), m.end(), f"{source}_unknown")
                stats["iban_unknown"] += 1
    return labels, new_masks, stats
