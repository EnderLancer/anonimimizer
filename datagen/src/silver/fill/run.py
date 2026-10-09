"""Step 2: replace placeholders with synthetic values, producing text + labelled spans."""
from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter

from silver.fill.addresses import AddressGenerator
from silver.fill.context import (INFO_LEFT, INFO_RIGHT, guess_case, guess_gender, guess_org_case, info_subtype, info_subtypes,
                                 inside_quotes, number_subtype)
from silver.fill.info import DERIVED_CHILD_AGE, KNOWN_SUBTYPES, DocState, InfoGenerator
from silver.fill.numbers import NumberGenerator
from silver.fill.orgs import ORG_KINDS, Org, Seen, inner_name
from silver.fill.persons import PersonGenerator
from silver.io import SILVER, VARIANT, read_jsonl, step_path, write_jsonl

log = logging.getLogger("silver")
# how many times a Supreme Court decision is filled, each time with other synthetic values
VS_FILLS = {"base": 1, "large": 2}[VARIANT]
# … and a decision with a LABELLED institution / company (the rarest, hardest sub-type): the same contexts seen with
# other names teach the context, not the name. Data v5 filled such decisions 2 / 3 times and counted the unresolved
# «other» placeholders too: documents with a labelled institution were 29-31 % of those that mention an organisation
INFO_ORG_FILLS = {"base": 1, "large": 2}[VARIANT]
INFO_ORG_SUBTYPES = {"org", "quoted", "military_name"}
# names in the quotes of the text that are labelled; a surname or a name of unknown kind is masked
QUOTE_KINDS = ("company", "bank", "formation", "institution", "person", "unknown")
LABELLED_QUOTES = {"company", "bank", "formation", "institution"}
# sub-types whose value is never a label, whatever the hints say (section 2 of the v6 specification): the rule that
# finds them is unreliable or the kind of the hidden value is not known
ALWAYS_MASKED = {"date_event", "ident", "nick", "place", "other"}
MASK_REASON = "info_unsure"  # a generated value of uncertain kind: it makes the text read naturally, the loss ignores it
_MILITARY = re.compile(r"(військов|оборони|ТЦК|комплектування|мобілізац|грошового забезпечення)", re.I)
_COMMERCIAL = re.compile(r"(господарськ\w+\s+суд|\bГПК\b)", re.I)
_SOCIAL = re.compile(r"(пенсі[йяюї]|соціальн\w+\s+(?:захист|допомог|виплат)|державн\w+\s+допомог|інвалідн)", re.I)
# «(далі - », «(далі також - відповідач, »: the slot of a short name; the body it abbreviates stands right before the bracket
_SHORT_SLOT = re.compile(r"\(\s*дал[іi]\s*(?:також\s*)?[-–—]?\s*(?:[^()]{0,40}[,;]\s*)?$", re.I)
_PERSON_REF = re.compile(r"ОСОБА_(\d+)")
# «до досягнення ОСОБА_3, ІНФОРМАЦІЯ_2, повноліття, тобто до », «до закінчення ОСОБА_2, ІНФОРМАЦІЯ_4, навчання … (тобто »:
# the birth date named inside the clause of a derived date is the one it is computed from
_DERIVED_CLAUSE_DOB = re.compile(r"(?:досягнення|закінчення|повноліття)[^.;]{0,60}?ОСОБА_\d+\s*,?\s*\(?\s*ІНФОРМАЦІЯ_(\d+)[^.;]{0,90}$", re.I)
# «до досягнення ОСОБА_3 повноліття, тобто до », «до повноліття сина ОСОБА_4, тобто до »: the child, named without a date
_DERIVED_CLAUSE_PERSON = re.compile(r"(?:досягнення|повноліття|закінчення\s+навчання)\s+(?:[а-яіїє]+\s+){0,2}?ОСОБА_(\d+)[^.;]{0,60}$", re.I)
DERIVED_LOOK_BACK = 200  # characters before a derived date in which the child's birth date is looked for
TAIL = 200               # characters of the filled text kept for the values that depend on it


def resolve_subtype(st: str, hints: dict) -> tuple[str, dict]:
    """The sub-type and hints the generators work with: a sub-type they do not know becomes a masked «other», a name
    in quotes or an institution of an unknown kind becomes the masked kind, and `mask` is set wherever the value
    must not be a label."""
    hints = dict(hints or {})
    if st not in KNOWN_SUBTYPES:
        return "other", {"mask": True}
    if st == "quoted":
        kind = hints.get("quote_kind", "company")  # the rules always name the kind; a bare «quoted» is a company, as before v6
        hints["quote_kind"] = kind if kind in QUOTE_KINDS else "unknown"
        if hints["quote_kind"] not in LABELLED_QUOTES:
            hints["mask"] = True
    if st == "org":
        kind = hints.get("org_kind", "party")
        hints["org_kind"] = kind if kind in ORG_KINDS else "party"
        if hints["org_kind"] == "official":
            hints["mask"] = True
    if st in ALWAYS_MASKED:
        hints["mask"] = True
    hints["mask"] = hints.get("mask") is True
    return st, hints


def link_derived(text: str, phs: list[dict], info_sub: dict[int, tuple[str, dict]]) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
    """Which birth date every derived date («до досягнення повноліття, тобто до …») is computed from, read from the text
    before it is filled: the birth date named in the clause of the derived date («до досягнення ОСОБА_3, ІНФОРМАЦІЯ_2,
    повноліття»), or the one that follows another mention of the person named there; else the nearest birth date of a
    child in the 200 characters before it; else the child's birth date that follows another mention of the person
    named last before it. Outside the clause a birth date counts as a child's by its own mention, not by the vote
    over all mentions of the placeholder.
    Returns derived index -> birth-date index, and birth-date index -> (lowest, highest) age of the person: a derived
    date in the future needs a child of that age."""
    derived = {idx: hints.get("until", "majority") for idx, (st, hints) in info_sub.items() if st == "derived"}
    dob = {idx for idx, (st, _) in info_sub.items() if st == "dob"}
    if not derived or not dob:
        return {}, {}
    # birth-date placeholders whose own context has a child cue: (position, index), and the person named right before
    child_at: list[tuple[int, int]] = []
    of_person: dict[int, int] = {}   # person index -> the birth date of a child that follows a mention of the person
    of_anybody: dict[int, int] = {}  # … -> the birth date that follows a mention of the person, child or not
    for p in phs:
        if p["type"] == "ІНФОРМАЦІЯ" and p["idx"] in dob:
            left = text[max(0, p["start"] - INFO_LEFT): p["start"]]
            st, hints = info_subtype(left, text[p["end"]: p["end"] + INFO_RIGHT])
            person = re.search(r"ОСОБА_(\d+)\s*,?\s*\(?\s*$", left)
            if person and st == "dob":
                of_anybody.setdefault(int(person.group(1)), p["idx"])
            if st == "dob" and hints.get("child"):
                child_at.append((p["start"], p["idx"]))
                if person:
                    of_person.setdefault(int(person.group(1)), p["idx"])
    source: dict[int, int] = {}
    ages: dict[int, tuple[int, int]] = {}
    for p in phs:
        idx = p["idx"]
        if p["type"] != "ІНФОРМАЦІЯ" or idx not in derived or idx in source:
            continue
        left = text[max(0, p["start"] - DERIVED_LOOK_BACK): p["start"]]
        clause, named = _DERIVED_CLAUSE_DOB.search(left), _DERIVED_CLAUSE_PERSON.search(left)
        near = [i for pos, i in child_at if p["start"] - DERIVED_LOOK_BACK <= pos < p["start"]]
        persons = _PERSON_REF.findall(left)
        if clause and int(clause.group(1)) in dob:
            src = int(clause.group(1))
        elif named and int(named.group(1)) in of_anybody:
            src = of_anybody[int(named.group(1))]
        else:
            src = near[-1] if near else of_person.get(int(persons[-1])) if persons else None
        # two children of one decision do not come of age on one day: a birth date gives one derived date of a kind
        if src is None or any(source[d] == src and derived[d] == derived[idx] for d in source):
            continue
        source[idx] = src
        low, high = DERIVED_CHILD_AGE.get(derived[idx], DERIVED_CHILD_AGE["majority"])
        ages[src] = (max(low, ages[src][0]), min(high, ages[src][1])) if src in ages else (low, high)
    # «до повноліття» and «до трьох років» of one child can leave no age: the narrower demand wins
    return source, {i: (min(a), min(a)) if a[0] > a[1] else a for i, a in ages.items()}


def has_org_info(doc: dict) -> bool:
    """The decision has a labelled institution-like placeholder: a body, a unit's name or a name in quotes that is
    not masked. Only such decisions are filled again (INFO_ORG_FILLS)."""
    resolved = (resolve_subtype(st, hints) for st, hints in info_subtypes(doc).values())
    return any(st in INFO_ORG_SUBTYPES and not hints["mask"] for st, hints in resolved)


class Filler:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.persons = PersonGenerator(self.rng)
        self.addresses = AddressGenerator(self.rng)
        self.numbers = NumberGenerator(self.rng)
        self.info = InfoGenerator(self.rng, self.persons)
        self.stats: Counter = Counter()

    def fill(self, doc: dict, fill: int = 1) -> dict:
        """`fill` numbers the versions of one document: the same decision with other synthetic values."""
        # every document has its own random stream: a change in one fill (or in the set of documents) leaves the
        # other texts, and the NER cache keyed by them, as they were
        self.rng.seed(f"{self.seed}:{doc['id']}" + (f":{fill}" if fill > 1 else ""))
        text = doc["text"]
        phs = sorted(doc["placeholders"], key=lambda p: p["start"])
        # per-document registries: one identity per placeholder index
        people: dict[int, object] = {}
        addrs: dict[int, dict] = {}
        nums: dict[int, str] = {}
        infos: dict[int, tuple[str, str | Org, dict]] = {}
        seen_person: set[int] = set()

        # decide НОМЕР / ІНФОРМАЦІЯ sub-types from the most informative occurrence
        num_sub: dict[int, str] = {}
        for p in phs:
            if p["type"] == "НОМЕР":
                st = number_subtype(text[max(0, p["start"] - 60): p["start"]], text[p["end"]: p["end"] + 40])
                if st != "generic" or p["idx"] not in num_sub:
                    num_sub[p["idx"]] = st
        info_sub = {idx: resolve_subtype(st, hints) for idx, (st, hints) in info_subtypes(doc).items()}
        # the names the decision has in the open and the birth date of a child, for the values that depend on them
        state = DocState(Seen(text))
        state.derived_from, state.child_age = link_derived(text, phs, info_sub)
        # what the decision is about decides which institutions it anonymises
        org_ctx = {"military": bool(_MILITARY.search(text)),
                   "commercial": doc.get("kind") == "ГПК" or bool(_COMMERCIAL.search(text[:3000])),
                   "social": bool(_SOCIAL.search(text))}

        # one address area / style and one date format per document
        addr_ctx = None
        date_style = self.info.doc_date_style()
        # a person to tie e-mails / nicknames to: the one mentioned most recently before the placeholder
        out, labels, masks, pos = [], [], [], 0
        last_person = None
        tail = ""  # the end of the filled text so far
        for p in phs:
            out.append(text[pos: p["start"]])
            tail = (tail + out[-1])[-TAIL:]
            start = sum(len(s) for s in out)
            t, idx = p["type"], p["idx"]
            left = text[max(0, p["start"] - 90): p["start"]]
            right = text[p["end"]: p["end"] + INFO_RIGHT]
            mask_reason = None
            if t == "ОСОБА":
                if idx not in people:
                    people[idx] = self.persons.new_person(guess_gender(left))
                person = people[idx]
                case = self.persons.choose_case(guess_case(left))
                value = self.persons.render(person, case, first_mention=idx not in seen_person)
                seen_person.add(idx)
                last_person = person
                self.stats[f"case_{case}"] += 1
            elif t == "АДРЕСА":
                if idx not in addrs:
                    addr_ctx = addr_ctx or self.addresses.doc_context(text)
                    a = addrs[idx] = self.addresses.new_address(addr_ctx)
                    self.stats.update([f"addr_kind_{a['kind']}", f"addr_order_{a['style']['order']}", f"addr_src_{a['src']}"]
                                      + [f"addr_{k}" for k in ("from_doc", "with_zip", "apt", "no_street", "city_district") if a[k]])
                    value = self.addresses.render(a)
                else:
                    value = self.addresses.render(addrs[idx], variant=self.rng.random() < 0.3)
            elif t == "НОМЕР":
                if idx not in nums:
                    nums[idx] = self.numbers.generate(num_sub.get(idx, "generic"), text[max(0, p["start"] - 20): p["start"]])
                    self.stats[f"num_{num_sub.get(idx, 'generic')}"] += 1
                value = nums[idx]
            else:  # ІНФОРМАЦІЯ
                st, hints = info_sub.get(idx, ("other", {"mask": True}))
                if idx not in infos:
                    if st == "place":  # «уродженця ІНФОРМАЦІЯ_2»: a settlement, in the area of the decision
                        addr_ctx = addr_ctx or self.addresses.doc_context(text)
                        v = self.addresses.settlement_phrase(addr_ctx)
                    elif st == "org" and not hints["mask"] and hints["org_kind"] in ("short", "party") and _SHORT_SLOT.search(left):
                        # «військової частини А5885 (далі - 【X】)»: the short name of the body before the bracket, which
                        # may itself be a value filled a moment ago; a body with no usual short name gets a random one,
                        # masked when the slot is of the «short» kind (it would contradict the name next to it)
                        v = self.info.org_gen.short_for(tail[: tail.rfind("(")])
                        if v is not None:
                            state.seen.add(v)
                            self.stats["info_short_from_antecedent"] += 1
                        else:
                            if hints["org_kind"] == "short":
                                st, hints = info_sub[idx] = (st, {**hints, "mask": True})
                            v = self.info.generate(st, hints, last_person, date_style, org_ctx, state, left, right, idx)
                    else:
                        v = self.info.generate(st, hints, last_person, date_style, org_ctx, state, left, right, idx)
                    infos[idx] = (st, v, hints)
                    self.stats[f"info_{st}"] += 1
                    if hints["mask"]:
                        self.stats[f"info_masked_{st}"] += 1
                    if st == "org":
                        self.stats[f"info_orgkind_{hints['org_kind']}"] += 1
                    if st == "quoted":
                        self.stats[f"info_quotekind_{hints['quote_kind']}"] += 1
                    if isinstance(v, Org):
                        self.stats[f"info_org_{'military' if org_ctx['military'] else 'commercial' if org_ctx['commercial'] else 'other'}"] += 1
                    if st in ("dob", "death", "derived", "date_event"):
                        self.stats[f"date_doc_style_{date_style}"] += 1
                v = infos[idx][1]
                if hints["mask"]:  # every mention of the placeholder: the value is one per index, and so is its doubt
                    mask_reason = MASK_REASON
                # an institution is declined by the case its context asks for, like a person's name
                value = v.render(guess_org_case(left)) if isinstance(v, Org) else v
                if isinstance(v, Org) and inside_quotes(left, right):
                    # «з КНЗ ЦТЗ «【X】»»: the text has the quotes, so only the name that a body carries in its own
                    value = inner_name(v.render())
            out.append(value)
            tail = (tail + value)[-TAIL:]
            end = start + len(value)
            if mask_reason:
                masks.append({"start": start, "end": end, "reason": mask_reason})
            else:
                labels.append({"start": start, "end": end, "type": t, "pseudonym": f"{t}_{idx}", "source": "placeholder"})
            pos = p["end"]
        out.append(text[pos:])
        new_text = "".join(out)
        return {
            "id": doc["id"],
            "fill": fill,
            "corpus": doc.get("corpus"),
            "kind": doc.get("kind"),
            "case": doc.get("case"),
            "text": new_text,
            "text_pseudonymized": text,
            "labels": labels,
            "masks": masks,
            "is_negative": doc.get("is_negative", False),
            "n_persons": len(people),
        }


def run_fill(seed: int = 42, limit: int | None = None) -> None:
    filler = Filler(seed)
    multi: Counter = Counter()

    def gen():
        for i, doc in enumerate(read_jsonl(step_path("00_filtered"))):
            if limit and i >= limit:
                break
            n = VS_FILLS if doc.get("kind") == "ВС" else 1
            if has_org_info(doc):
                multi["org_info_docs"] += 1
                multi["org_info_refilled"] += INFO_ORG_FILLS > n  # filled more often than its kind of decision alone asks for
                n = max(n, INFO_ORG_FILLS)
            for k in range(1, n + 1):
                yield filler.fill(doc, fill=k)

    n = write_jsonl(step_path("01_filled"), gen())
    # where the institution / company names came from: real names of the org_names table or the generator's word lists
    filler.stats.update({f"org_name_{k}": v for k, v in filler.info.org_gen.used.items()})
    # what was drawn: kinds of body by kind of slot («party_military», «holder_civil»), names in quotes, redraws of a
    # name that already stood in the decision
    filler.stats.update({f"org_drawn_{k}": v for k, v in filler.info.org_gen.drawn.items()})
    stats = dict(sorted(filler.stats.items()))
    (SILVER / "fill_stats.json").write_text(json.dumps({"documents": n, **multi, **stats}, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("filled %d documents (%s); %s", n, dict(multi), stats)
