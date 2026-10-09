"""Layer 4: context-dependent ІНФОРМАЦІЯ / АДРЕСА labels and masks for everything PD-like we do not label."""
from __future__ import annotations

import re

from silver import patterns as P
from silver.merge import any_overlap


def label_masks(text: str, existing: list[dict], masks: list[dict]) -> tuple[list[dict], list[dict]]:
    labels: list[dict] = []
    new_masks: list[dict] = []
    persons = [l for l in existing if l["type"] == "ОСОБА"]

    def free(sp):
        return not any(any_overlap(sp, lst) for lst in (existing, labels, masks, new_masks))

    def lab(s, e, t, source):
        sp = {"start": s, "end": e, "type": t, "pseudonym": None, "source": source}
        if free(sp):
            labels.append(sp)

    def mask(s, e, reason):
        sp = {"start": s, "end": e, "reason": reason}
        if free(sp):
            new_masks.append(sp)

    for m in P.EMAIL.finditer(text):
        if P.GOV_MAIL.search(m.group()):
            continue
        # personal e-mail = personal domain + a person mentioned nearby + no institution markers in the block before it
        near_person = any(0 <= m.start() - p["end"] <= 120 for p in persons)
        if P.PERSONAL_MAIL.search(m.group()) and near_person and not P.COMPANY_LEFT.search(text[max(0, m.start() - 160): m.start()]):
            lab(m.start(), m.end(), "ІНФОРМАЦІЯ", "email")
        else:
            mask(m.start(), m.end(), "email_other")
    for m in P.PHONE.finditer(text):
        mask(m.start(), m.end(), "phone")
    for m in P.PLATE.finditer(text):
        mask(m.start(), m.end(), "plate")
    for rx in (P.DATE_NUM, P.DATE_WORDS):
        for m in rx.finditer(text):
            left = text[max(0, m.start() - 25): m.start()]
            right = text[m.end(): m.end() + 30]
            year = int(re.search(r"\d{4}", m.group(1)).group())
            # «ПІБ, 12.01.1980,» — birth date; «ПІБ 11 лютого 2002 року склав заповіт» — an event
            after_person = any(p["end"] <= m.start() <= p["end"] + 3 and text[p["end"]: m.start()].strip() == "," for p in persons) \
                and re.match(r"\s*(?:[,;)]|р\.|року\s+народж|р\.\s?н\.)", right) is not None
            # the cue must stand right before the date («помер 12.04.2019», «народився 3 травня 1980») or right after
            # («12.04.1980 р.н.», «... року народження»); «запис про народження № 24 від 3 листопада 1992» is a record date
            cue = (P.DOB_CUE_LEFT.search(left) and not P.NOT_DOB_LEFT.search(text[max(0, m.start() - 45): m.start()])) \
                or P.DOB_CUE_RIGHT.search(right)
            if P.TIME_RIGHT.search(right):
                continue  # «14 листопада 2023 року біля 10 години» is an event, not a birth date
            if cue or (after_person and year <= 2006):
                lab(m.start(1), m.end(1), "ІНФОРМАЦІЯ", "dob")
    for m in P.ADDRESS_CTX.finditer(text):
        s, e = m.start(1), m.end(1)
        if P.NOT_ADDRESS_LEFT.search(text[max(0, m.start() - 60): m.start()]):
            continue
        near_person = any(abs(p["end"] - m.start()) <= 120 for p in persons)
        if near_person:
            lab(s, e, "АДРЕСА", "address")
        else:
            mask(s, e, "address_no_person")
    return labels, new_masks
