"""Layers 2–3: open identifiers → НОМЕР labels or masks."""
from __future__ import annotations

from silver import patterns as P
from silver.merge import any_overlap


def _free(span: dict, *lists: list[dict]) -> bool:
    return not any(any_overlap(span, lst) for lst in lists)


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
    # local courts leave most of these numbers open, the Supreme Court decisions of the test hide all of them
    for m in P.VP.finditer(text):
        lab(m.start(1), m.end(1), "vp")
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
