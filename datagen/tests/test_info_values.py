"""Data v6: values of ІНФОРМАЦІЯ placeholders by sub-type and hints, masks of the fill step, the repeat rule.

The sub-types are given to the generators directly or forced on the fill step (silver.fill.run.info_subtypes is
replaced), so these tests do not depend on the context rules."""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter
import random
import re

import pytest

import silver.fill.run as run
from silver.excerpt import excerpt
from silver.fill.info import DERIVED_CHILD_AGE, NEWS_DOMAINS, REF_YEAR, SITE_DOMAINS, WAR_START, DocState, InfoGenerator
from silver.fill.orgs import (AIR_COMMANDS, BANKS, FORMATION_NAMES, KIND_MIX, MAX_REDRAWS, MOBILE_OPERATORS, OPERATIONAL_COMMANDS, Org,
                              OrgGenerator, Seen, inner_name, usable_name)
from silver.fill.persons import Person
from silver.label import label_doc
from silver.normalize import find_placeholders, normalize

NAMES = {("court", "company"): ["Судова Назва"], ("court", "body"): ["Головне управління Держпраці у Львівській області"],
         ("treasury", "social"): ["Управління соціального захисту населення Стрийської районної державної адміністрації"],
         ("treasury", "prosecutor"): ["Вінницька обласна прокуратура"], ("treasury", "court"): ["Херсонський апеляційний суд"]}
MASK = {"mask": True}


def _doc(raw: str, doc_id: str = "t", **extra) -> dict:
    text = normalize(raw)
    return {"id": doc_id, "text": text, "placeholders": find_placeholders(text), **extra}


def _force(monkeypatch, subtypes: dict[int, tuple[str, dict]]) -> None:
    monkeypatch.setattr(run, "info_subtypes", lambda doc: {i: v for i, v in subtypes.items()
                                                           if any(p["type"] == "ІНФОРМАЦІЯ" and p["idx"] == i for p in doc["placeholders"])})


def _info(out: dict, idx: int, where: str = "labels") -> list[str]:
    """Values of ІНФОРМАЦІЯ_idx among the labels; among the masks every masked value comes back (they have no pseudonym)."""
    if where == "labels":
        return [out["text"][l["start"]:l["end"]] for l in out["labels"] if l["pseudonym"] == f"ІНФОРМАЦІЯ_{idx}"]
    return [out["text"][m["start"]:m["end"]] for m in out["masks"]]


def _filled(out: dict) -> list[str]:
    """Every ІНФОРМАЦІЯ value of the text in its order, a label or a mask alike."""
    spans = [l for l in out["labels"] if l["type"] == "ІНФОРМАЦІЯ"] + out["masks"]
    return [out["text"][x["start"]:x["end"]] for x in sorted(spans, key=lambda x: x["start"])]


def _civil(out: dict) -> list[str]:
    return [out["text"][m["start"]:m["end"]] for m in out["masks"] if m["reason"] == "info_civil_org"]


# ---- masks ---------------------------------------------------------------------------------------------------------
def test_a_masked_value_is_a_mask_at_every_mention_and_never_a_label(monkeypatch):
    raw = ("ОСОБА_1 звернувся з позовом до ІНФОРМАЦІЯ_1 про стягнення коштів. Суддя ІНФОРМАЦІЯ_2 відкрив провадження. "
           "Зобов'язати ІНФОРМАЦІЯ_1 виплатити кошти ОСОБА_1 . Скаргу подало ІНФОРМАЦІЯ_2 повторно.")
    _force(monkeypatch, {1: ("org", {"org_kind": "party"}), 2: ("other", MASK)})
    out = run.Filler(seed=5).fill(_doc(raw))
    assert len(_info(out, 1)) == 2 and not _info(out, 2)
    assert [m["reason"] for m in out["masks"]] == ["info_unsure", "info_unsure"]
    masked = _info(out, 2, "masks")
    assert len(masked) == 2 and "ІНФОРМАЦІЯ" not in out["text"]
    assert len({m.split()[0][:4] for m in masked}) == 1  # one value per placeholder index, declined per mention
    spans = sorted([(l["start"], l["end"]) for l in out["labels"]] + [(m["start"], m["end"]) for m in out["masks"]])
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))  # a mask never overlaps a label
    assert out["n_persons"] == 1


@pytest.mark.parametrize("subtype, hints, masked", [
    ("other", {}, True), ("date_event", {}, True), ("ident", {}, True), ("place", {}, True), ("no_such_subtype", {}, True),
    ("date", {}, True),  # the sub-type of data v5 that the rules no longer give: unknown, so «other»
    ("death", {"weak": True, "mask": True}, True), ("death", {}, False), ("dob", {"child": True}, False), ("derived", {"until": "age3"}, False),
    ("email", {}, False), ("url", {"url_kind": "site"}, False), ("url", {"url_kind": "page", "mask": True}, True), ("military_name", {}, False),
    ("nick", {"nick_kind": "page"}, True), ("nick", {"nick_kind": "handle", "mask": True}, True),  # masked since the review of v6
    ("org", {}, False), ("org", {"org_kind": "unit"}, False), ("org", {"org_kind": "unit", "mask": True}, True),
    ("org", {"org_kind": "short"}, False), ("org", {"org_kind": "holder"}, False), ("org", {"org_kind": "official"}, True),
    # only a military formation is a label: a company, a bank, a hospital is not personal data (reason info_civil_org)
    ("quoted", {"quote_kind": "company"}, True), ("quoted", {"quote_kind": "bank"}, True), ("quoted", {"quote_kind": "formation"}, False),
    ("quoted", {"quote_kind": "institution"}, True), ("quoted", {"quote_kind": "person"}, True), ("quoted", {"quote_kind": "unknown"}, True),
    ("quoted", {"quote_kind": "no_such_kind"}, True), ("quoted", {}, True),
])
def test_every_subtype_of_the_contract_is_filled_and_masked_as_its_table_says(monkeypatch, subtype, hints, masked):
    _force(monkeypatch, {1: (subtype, hints)})
    out = run.Filler(seed=2).fill(_doc("ОСОБА_1 подав заяву. У справі зазначено «ІНФОРМАЦІЯ_1» як доказ."))
    assert "ІНФОРМАЦІЯ" not in out["text"]
    if subtype == "org" and not masked and _civil(out):  # the body that was drawn is a civil one or a company
        assert not _info(out, 1) and len(_civil(out)) == len(out["masks"]) == 1
    else:
        assert bool(_info(out, 1, "masks")) is masked and bool(_info(out, 1)) is not masked
    resolved = run.resolve_subtype(subtype, hints)[1]
    assert resolved["mask"] is masked
    civil_quote = subtype == "quoted" and hints.get("quote_kind", "company") in run.CIVIL_QUOTES
    assert (resolved.get("reason") == "info_civil_org") is civil_quote
    if subtype != "org":
        assert bool(_civil(out)) is civil_quote
    assert out["n_persons"] == 1  # a masked surname in quotes is not a person of the document


def test_an_unknown_subtype_is_filled_like_other(monkeypatch):
    assert run.resolve_subtype("no_such_subtype", {"child": True}) == ("other", MASK)
    g = InfoGenerator(random.Random(1))
    assert all(isinstance(g.generate("no_such_subtype", {}, None), Org) for _ in range(50))


def test_the_later_layers_keep_the_masks_of_the_fill_step(monkeypatch):
    raw = ("ВСТАНОВИВ:\n" + "Позивач ОСОБА_1 , РНОКПП НОМЕР_1 , звернувся до суду. " * 12 + "Довідку видано ІНФОРМАЦІЯ_1 , прізвище змінено на «ІНФОРМАЦІЯ_2», "
           "подію зафіксовано ІНФОРМАЦІЯ_3 , логін ІНФОРМАЦІЯ_4 . " + "Суд дослідив докази у справі та заслухав пояснення сторін. " * 12)
    _force(monkeypatch, {1: ("other", MASK), 2: ("quoted", {"quote_kind": "person"}), 3: ("date_event", MASK), 4: ("ident", MASK)})
    for seed in range(8):
        out = run.Filler(seed=seed).fill(_doc(raw))
        unsure = [(m["start"], m["end"]) for m in out["masks"]]
        assert len(unsure) == 4 and not [l for l in out["labels"] if l["type"] == "ІНФОРМАЦІЯ"]
        labelled = label_doc(out)
        assert [(m["start"], m["end"]) for m in labelled["masks"] if m["reason"] == "info_unsure"] == unsure
        # no layer puts a label on a masked value: not the date layer on the masked date, not the person layer on the surname
        assert not [l for l in labelled["labels"] for s, e in unsure if l["start"] < e and s < l["end"]]
        cut = excerpt(labelled)
        assert [cut["text"][m["start"]:m["end"]] for m in cut["masks"] if m["reason"] == "info_unsure"] == [out["text"][s:e] for s, e in unsure]


# ---- only a military body is a label ---------------------------------------------------------------------------------
def test_a_civil_body_or_a_company_is_masked_at_every_mention_and_a_military_body_is_labelled(monkeypatch):
    raw = ("ОСОБА_1 звернувся з позовом до ІНФОРМАЦІЯ_1 про стягнення коштів. Суд зобов'язав ІНФОРМАЦІЯ_1 надати довідку, "
           "а документи зберігаються в ІНФОРМАЦІЯ_2 .")
    _force(monkeypatch, {1: ("org", {"org_kind": "party"}), 2: ("org", {"org_kind": "holder"})})
    kinds = Counter()
    for seed in range(200):
        f = run.Filler(seed=seed)
        out = f.fill(_doc(raw, doc_id=str(seed)))
        labelled, masked = [v for i in (1, 2) for v in _info(out, i)], _civil(out)
        assert len(labelled) + len(masked) == 3 and len(_info(out, 1)) in (0, 2)  # one decision per placeholder, not per mention
        assert {m["reason"] for m in out["masks"]} <= {"info_civil_org"}
        assert not any(re.search(r"Пенсійного фонду|поліції|соціального захисту населення|ТОВ|АТ |ради$", v) for v in labelled), labelled
        kinds.update(k for k in f.stats if k.startswith("info_civil_"))
    assert kinds["info_civil_civil"] > 20 and kinds["info_civil_company"] > 5 and kinds["info_civil_masked_org"] > 30


def test_no_military_draw_is_a_civil_body():
    g = OrgGenerator(random.Random(8), names=NAMES)
    names = [g.military_body().render() for _ in range(3000)]
    assert not [n for n in names if re.search(r"Пенсійного фонду|Національної поліції|соціального захисту населення", n)]
    assert not g.used["body_treasury_social"]


# ---- «other» and the repeat rule -------------------------------------------------------------------------------------
def test_other_is_always_an_institution():
    g = InfoGenerator(random.Random(4))
    assert all(isinstance(g.generate("other", MASK, None, "iso", {"military": True}), Org) for _ in range(300))
    assert not hasattr(__import__("silver.fill.info", fromlist=["x"]), "P_OTHER_ORG")


def test_only_a_labelled_institution_asks_for_another_fill(monkeypatch):
    assert run.INFO_ORG_FILLS == {"base": 1, "large": 2}[run.VARIANT]
    doc = _doc("Позов ОСОБА_1 до ІНФОРМАЦІЯ_1 .")
    for value, expected in [(("other", MASK), False), (("other", {}), False), (("org", {"org_kind": "official", "mask": True}), False),
                            (("org", {"org_kind": "unit", "mask": True}), False), (("quoted", {"quote_kind": "person", "mask": True}), False),
                            (("quoted", {"quote_kind": "unknown", "mask": True}), False), (("dob", {}), False), (("date_event", MASK), False),
                            (("quoted", {"quote_kind": "company"}), False), (("quoted", {"quote_kind": "bank"}), False),
                            (("org", {}), True), (("org", {"org_kind": "holder"}), True), (("military_name", {}), True),
                            (("quoted", {"quote_kind": "formation"}), True)]:
        _force(monkeypatch, {1: value})
        assert run.has_org_info(doc) is expected, value


def test_run_fill_repeats_only_decisions_with_a_labelled_institution(monkeypatch, tmp_path):
    docs = [_doc("Позов ОСОБА_1 до ІНФОРМАЦІЯ_1 про стягнення.", "1"), _doc("Позов ОСОБА_1 , згадано ІНФОРМАЦІЯ_2 у справі.", "2"),
            _doc("Позов ОСОБА_1 до ОСОБА_2 .", "3")]
    (tmp_path / "00_filtered.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
    monkeypatch.setattr(run, "step_path", lambda step: tmp_path / f"{step}.jsonl")
    monkeypatch.setattr(run, "SILVER", tmp_path)
    monkeypatch.setattr(run, "INFO_ORG_FILLS", 2)  # the «large» variant
    _force(monkeypatch, {1: ("org", {"org_kind": "party"}), 2: ("other", MASK)})
    run.run_fill(seed=3)
    filled = [json.loads(l) for l in (tmp_path / "01_filled.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(d["id"], d["fill"]) for d in filled] == [("1", 1), ("1", 2), ("2", 1), ("3", 1)]
    assert filled[0]["text"] != filled[1]["text"]
    stats = json.loads((tmp_path / "fill_stats.json").read_text(encoding="utf-8"))
    assert stats["documents"] == 4 and stats["org_info_docs"] == 1 and stats["org_info_refilled"] == 1
    assert stats["info_org"] == 2 and stats["info_orgkind_party"] == 2 and stats["info_other"] == 1 and stats["info_masked_other"] == 1
    # the bodies that were kept, by kind of slot; the masked «other» placeholder is counted apart
    assert "info_masked_org" not in stats and sum(v for k, v in stats.items() if k.startswith("org_drawn_party_")) == 2
    assert sum(v for k, v in stats.items() if k.startswith("org_drawn_other_")) == 1


def test_fill_statistics_count_masked_values_and_kinds(monkeypatch):
    raw = "Позов до ІНФОРМАЦІЯ_1 , ТОВ «ІНФОРМАЦІЯ_2», прізвище «ІНФОРМАЦІЯ_3», суддя ІНФОРМАЦІЯ_4 , дата ІНФОРМАЦІЯ_5 ."
    _force(monkeypatch, {1: ("org", {"org_kind": "holder"}), 2: ("quoted", {"quote_kind": "bank"}), 3: ("quoted", {"quote_kind": "person", "mask": True}),
                         4: ("org", {"org_kind": "official", "mask": True}), 5: ("date_event", MASK)})
    f = run.Filler(seed=1)
    f.fill(_doc(raw))
    want = {"info_org": 2, "info_quoted": 2, "info_date_event": 1, "info_masked_org": 1, "info_masked_quoted": 1, "info_masked_date_event": 1,
            "info_orgkind_holder": 1, "info_orgkind_official": 1, "info_quotekind_bank": 1, "info_quotekind_person": 1,
            "info_civil_masked_quoted": 1}  # the bank; a holder that is not a military body is counted the same way
    got = {k: v for k, v in f.stats.items() if k.startswith("info_") and not k.startswith("info_org_")}
    civil_holder = {k: got.pop(k) for k in ("info_civil_masked_org", "info_civil_company", "info_civil_civil") if k in got}
    assert got == want and sum(civil_holder.values()) in (0, 2)


def test_report_tells_masked_placeholders_and_kinds(monkeypatch, tmp_path, capsys):
    import silver.report as report

    (tmp_path / "00_filtered.jsonl").write_text("", encoding="utf-8")
    stats = {"documents": 3, "info_org": 6, "info_other": 3, "info_dob": 1, "info_masked_other": 3, "info_masked_org": 1, "info_orgkind_party": 5,
             "info_orgkind_official": 1, "info_quotekind_bank": 2, "info_org_military": 9, "org_drawn_party_military": 19, "org_drawn_party_civil": 1,
             "org_drawn_redraw": 4, "org_info_docs": 2, "org_info_refilled": 1}
    (tmp_path / "fill_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    monkeypatch.setattr(report, "SILVER", tmp_path)
    monkeypatch.setattr(report, "step_path", lambda step: tmp_path / f"{step}.jsonl")
    report.run_report()
    capsys.readouterr()
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "by sub-type (labelled and masked together): org 60.0%, other 30.0%, dob 10.0%" in text
    assert "4 of 10 (40.0%); by sub-type: other 75.0%, org 25.0%" in text
    assert "by kind of slot: party 83.3%, official 16.7%" in text and "military 95.0%, civil 5.0%, company 0.0%" in text
    assert "drawn again 4 times" in text and "filled once more because of it: 1" in text
    assert "75 %" not in text and "info_unsure" in text


# ---- dates -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("until, years", [("majority", 18), ("age23", 23), ("age3", 3)])
def test_a_derived_date_follows_the_birth_date_of_the_child(until, years):
    g = InfoGenerator(random.Random(8))
    for style in ("iso", "numeric", "words", "words_roku", "words_narodzh"):
        # the fill step linked the derived date (placeholder 3) to the birth date in placeholder 1
        doc = DocState(derived_from={3: 1}, child_age={1: DERIVED_CHILD_AGE[until]})
        born = dt.date.fromisoformat(g.generate("dob", {}, None, "iso", doc=doc, idx=1))
        g.generate("dob", {}, None, "iso", doc=doc, idx=2)  # another birth date in between changes nothing
        low, high = DERIVED_CHILD_AGE[until]
        assert doc.child_dob is not None and REF_YEAR - high <= born.year <= REF_YEAR - low
        value = g.generate("derived", {"until": until}, None, style, doc=doc, idx=3)
        assert value == g.render_date(born.replace(year=born.year + years), "words_roku" if style == "words_narodzh" else style)
        assert "народження" not in value and born.year + years >= REF_YEAR
        assert "року" not in g.generate("derived", {"until": until, "no_roku": True}, None, style, doc=doc, idx=3)


def test_a_derived_date_fixes_a_birth_date_that_comes_later_in_the_text():
    g = InfoGenerator(random.Random(4))
    for _ in range(50):
        doc = DocState(derived_from={1: 2}, child_age={2: DERIVED_CHILD_AGE["age3"]})
        until = dt.date.fromisoformat(g.generate("derived", {"until": "age3"}, None, "iso", doc=doc, idx=1))
        born = dt.date.fromisoformat(g.generate("dob", {}, None, "iso", doc=doc, idx=2))
        assert until == born.replace(year=born.year + 3) and until.year >= REF_YEAR


def test_unlinked_derived_dates_take_different_children_and_never_a_past_date():
    g = InfoGenerator(random.Random(6))
    for _ in range(100):
        doc = DocState()
        first = dt.date.fromisoformat(g.generate("dob", {"child": True}, None, "iso", doc=doc))
        second = dt.date.fromisoformat(g.generate("dob", {"child": True}, None, "iso", doc=doc))
        g.generate("dob", {}, None, "iso", doc=doc, left="стягнути з відповідача ОСОБА_1, ")  # the payer
        a = dt.date.fromisoformat(g.generate("derived", {"until": "majority"}, None, "iso", doc=doc))
        b = dt.date.fromisoformat(g.generate("derived", {"until": "majority"}, None, "iso", doc=doc))
        assert a == second.replace(year=second.year + 18) and b == first.replace(year=first.year + 18)  # the nearest first
        assert dt.date.fromisoformat(g.generate("derived", {"until": "age3"}, None, "iso", doc=doc)).year >= REF_YEAR


@pytest.mark.parametrize("until, first, last", [("majority", 1, 17), ("age23", 1, 5), ("age3", 0, 3), ("no_such_cue", 1, 17)])
def test_a_derived_date_without_a_child_is_a_date_ahead(until, first, last):
    g = InfoGenerator(random.Random(9))
    years = {int(g.generate("derived", {"until": until}, None, "iso")[:4]) for _ in range(400)}
    assert min(years) == REF_YEAR + first and max(years) == REF_YEAR + last


def test_the_fill_step_ties_a_derived_date_to_the_last_child_before_it(monkeypatch):
    raw = ("Стягувати аліменти на сина ОСОБА_2 , ІНФОРМАЦІЯ_1 , до повноліття, тобто до ІНФОРМАЦІЯ_2 , та на доньку ОСОБА_3 , "
           "ІНФОРМАЦІЯ_3 , до досягнення трьох років, тобто до ІНФОРМАЦІЯ_4 року. Позивач ОСОБА_1 , ІНФОРМАЦІЯ_5 .")
    _force(monkeypatch, {1: ("dob", {"child": True}), 2: ("derived", {"until": "majority"}), 3: ("dob", {"child": True}),
                         4: ("derived", {"until": "age3", "no_roku": True}), 5: ("dob", {})})
    styles = set()
    for seed in range(40):
        out = run.Filler(seed=seed).fill(_doc(raw))
        son, until18, daughter, until3, _ = (_info(out, i)[0] for i in range(1, 6))
        year = [int(re.search(r"\d{4}", v).group()) for v in (son, until18, daughter, until3)]
        assert year[1] == year[0] + 18 and year[3] == year[2] + 3
        assert son.replace(str(year[0]), "Y").replace(" народження", "") == until18.replace(str(year[1]), "Y")  # the same day and month
        assert "народження" not in until18 + until3 and "року" not in until3
        styles.add("iso" if "-" in son else "numeric" if "." in son else "words")
    assert styles == {"iso", "numeric", "words"}


def test_an_event_date_is_recent_and_never_a_birth_date():
    g = InfoGenerator(random.Random(10))
    for style in ("iso", "numeric", "words", "words_roku", "words_narodzh"):
        values = [g.generate("date_event", {}, None, style) for _ in range(100)]
        assert all(REF_YEAR - 6 <= int(re.search(r"\d{4}", v).group()) <= REF_YEAR and "народження" not in v for v in values)
        assert not any("року" in g.generate("date_event", {"no_roku": True}, None, style) for _ in range(30))
    weak = g.generate("death", {"weak": True, "mask": True}, None, "words_narodzh")
    assert weak.endswith("року") and REF_YEAR - 21 <= int(re.search(r"\d{4}", weak).group()) <= REF_YEAR


# ---- web addresses, accounts, identifiers ------------------------------------------------------------------------------
def test_web_addresses_and_accounts_follow_their_kind():
    g = InfoGenerator(random.Random(11))
    person = Person("masc", "Коваленко", "Іван", "Петрович")
    sites = [g.generate("url", {"url_kind": "site"}, person) for _ in range(200)]
    # «https://www.olx.ua», «court.gov.ua», «http://pravo-info.com.ua»: a host, no path
    assert all(re.fullmatch(r"(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+", s) for s in sites) and len(set(sites)) > 60
    pages = [g.generate("url", {"url_kind": "page"}, person) for _ in range(200)]
    # «https://tsn.ua/news/sud-sprava-123456.html», «https://www.youtube.com/watch?v=dQw4w9WgXcQ», «https://youtu.be/dQw4w9WgXcQ»
    assert all(re.fullmatch(r"https://[a-z0-9.-]+/\S+", p) for p in pages)
    assert any("youtube.com/watch?v=" in p for p in pages) and any(p.endswith(".html") for p in pages)
    profiles = [g.generate("url", {"url_kind": "profile"}, person) for _ in range(100)] + [g.generate("url", {}, person) for _ in range(100)]
    assert all(re.match(r"https://(?:www\.facebook\.com/profile|t\.me/|instagram\.com/|www\.olx\.ua/uk/list/user/)", p) for p in profiles)
    handles = [g.generate("nick", {"nick_kind": "handle"}, person) for _ in range(100)] + [g.generate("nick", {}, person) for _ in range(50)]
    assert all(re.fullmatch(r"@?kovalenko[\w.]*", h) for h in handles)
    channels = [g.generate("nick", {"nick_kind": "page"}, person) for _ in range(200)]
    # «https://t.me/kovalenko12», «t.me/novyny_misto», «https://www.youtube.com/@kovalenko.ua», «https://www.facebook.com/groups/123»
    assert all(re.fullmatch(r"(?:https://)?(?:www\.)?(?:t\.me|facebook\.com|instagram\.com|youtube\.com|tiktok\.com)/\S+", c) for c in channels)
    idents = [g.generate("ident", MASK, person) for _ in range(200)] + [g.generate("ident", MASK, None) for _ in range(100)]
    assert all(re.fullmatch(r"[A-Za-z0-9._]{4,24}", i) for i in idents) and len(set(idents)) > 150


# ---- institutions by kind of slot --------------------------------------------------------------------------------------
@pytest.mark.parametrize("org_kind, about, ctx", [
    ("party", "military", {"military": True}), ("party", "commercial", {"commercial": True}), ("party", "other", {}),
    ("party", "military", {"military": True, "commercial": True}),  # a serviceman's case in a commercial court is still military
    ("holder", "military", {"military": True}), ("holder", "other", {}), ("holder", "commercial", {"commercial": True}),
    ("short", "military", {"military": True}),  # a short name is an abbreviation of a party
])
def test_kinds_of_body_follow_the_slot_and_the_decision(org_kind, about, ctx):
    g = OrgGenerator(random.Random(21), names={})
    n = 6000
    for _ in range(n):
        g.institution(**ctx, org_kind=org_kind)
    want = KIND_MIX["holder" if org_kind == "holder" else "party"][about]
    got = [100 * g.drawn[f"{org_kind}_{k}"] / n for k in ("military", "civil", "company")]
    assert all(abs(a - b) < 2.5 for a, b in zip(got, want)), (got, want)


def test_the_mixes_are_those_of_the_audit():
    assert KIND_MIX == {"party": {"military": (95, 4, 1), "commercial": (5, 15, 80), "other": (70, 25, 5)},
                        "holder": {"military": (60, 25, 15), "commercial": (10, 45, 45), "other": (10, 45, 45)}}


def test_a_unit_slot_holds_the_units_own_name():
    g = OrgGenerator(random.Random(22), names={})
    units = [g.institution(commercial=True, org_kind="unit") for _ in range(400)]  # military whatever the decision is about
    names = [u.render() for u in units]
    assert not any(re.search(r"[Вв]ійськова частина|в/ч|ТЦК|територіальний центр|Міністерство|[АВТ]\d{4}", n) for n in names)
    for word in ("бригада", "полк", "прикордонний загін", "управління", "територіальної оборони", "навчальн"):
        assert any(word in n for n in names), word
    assert all(not u.fixed and u.render("gent") != u.render() for u in units)  # every one declines
    assert Org("31 прикордонний загін", "імені гетьмана Івана Виговського").render("gent") == "31 прикордонного загону імені гетьмана Івана Виговського"
    assert g.drawn["unit_military"] == 400


def test_a_short_name_never_declines():
    g = OrgGenerator(random.Random(23), names={})
    for ctx in ({"military": True}, {}, {"commercial": True}):
        shorts = [g.institution(**ctx, org_kind="short") for _ in range(300)]
        assert all(s.fixed and s.render("gent") == s.render("datv") == s.render() for s in shorts)
        assert max(len(s.render()) for s in shorts) < 60 and not any("{" in s.render() for s in shorts)
    military = {g.institution(military=True, org_kind="short").render() for _ in range(400)}
    assert {"ТЦК та СП", "РТЦК та СП", "Міноборони", "ВЛК"} <= military and any(re.fullmatch(r"в/ч [АВТ]\d{4}", m) for m in military)
    assert any(m.startswith("ГУ ПФУ в ") and m.endswith(" області") for m in military | {g.short_name("civil").render() for _ in range(200)})


def test_a_holder_is_a_bank_a_hospital_a_police_unit_or_an_operator():
    g = OrgGenerator(random.Random(24), names={})
    names = [g.institution(org_kind="holder").render() for _ in range(600)]
    for cue in ("Банк", "лікарня", "поліції", "Київстар"):
        assert any(cue in n for n in names), cue
    assert not any(re.search(r"(?<![\w-])суд(?![\w-])", n) for n in names)


def test_an_official_slot_holds_a_court_a_prosecutors_office_or_a_bank():
    g = OrgGenerator(random.Random(25), names=NAMES)
    names = [g.institution(org_kind="official").render() for _ in range(300)]
    assert all(re.search(r"суд|прокуратур|прокурора|«", n) for n in names)
    assert any("суд" in n for n in names) and any("прокуратур" in n for n in names) and any("«" in n for n in names)
    bank_names = {name for _, name in BANKS}
    assert all(g.official("перерахувала кошти на банківську картку ").tail.strip("«»") in bank_names for _ in range(40))
    assert all("суд" in g.official("ухвалою слідчого судді ").render() for _ in range(40))
    assert all(re.search(r"прокуратур|прокурора", g.official("клопотання прокурора ").render()) for _ in range(40))


def test_a_court_is_never_a_civil_body():
    g = OrgGenerator(random.Random(26), names=NAMES)
    bodies = [g.civil_body(registers=True).render() for _ in range(600)] + [g.institution(social=True).render() for _ in range(600)]
    bodies += [g.institution(org_kind=k, military=m).render() for k in ("party", "holder", "short", "unit") for m in (True, False) for _ in range(300)]
    assert "Вінницька обласна прокуратура" in bodies and not any(re.search(r"(?<![\w-])суд(?![\w-])", b) for b in bodies)


# ---- a name that already stands in the decision ------------------------------------------------------------------------
def test_seen_finds_whole_names_in_any_case_and_spelling():
    seen = Seen("Позов до Міністерства оборони України; ТОВ «Комтех Логістік»; вжито заходів; ГУ ПФУ в Одеській області; Солом’янський суд")
    assert seen.has(Org("Міністерство", "оборони України"))  # the genitive stands in the text
    assert seen.has(Org("ТОВ", "«КОМТЕХ ЛОГІСТІК»", fixed=True))  # the bare name in its quotes, in any capitals
    assert seen.has("Комтех Логістік") and seen.has("Солом'янський суд") and not seen.has("Комтех Логіст")
    assert not seen.has("Захід")  # «заходів» does not hold the name «Захід»
    assert not seen.has(Org("Міністерство", "юстиції України"))
    assert Seen("відмовлено Головному управлінню Держпраці").has(Org("Головне управління", "Держпраці"))  # any case form counts
    seen.add(Org("Міністерство", "юстиції України"))
    assert seen.has(Org("Міністерство", "юстиції України"))  # a name drawn for another placeholder of the decision


def test_a_body_named_in_the_decision_is_drawn_again():
    text = ("Позов до Міністерства оборони України, Генеральний штаб Збройних Сил України, ТЦК та СП, Національної гвардії України, "
            "Державна прикордонна служба України, Служба безпеки України.")
    plain, checked = OrgGenerator(random.Random(31), names={}), OrgGenerator(random.Random(31), names={})
    named = ("Міністерство оборони України", "Генеральний штаб Збройних Сил України", "ТЦК та СП", "Національна гвардія України",
             "Державна прикордонна служба України", "Служба безпеки України")
    without = [plain.institution(military=True).render() for _ in range(500)]
    with_check = [checked.institution(military=True, seen=Seen(text)).render() for _ in range(500)]
    assert sum(n in named for n in without) > 25  # about one draw in fifteen is a body the decision names
    assert not any(n in named for n in with_check)
    assert checked.drawn["redraw"] >= sum(n in named for n in without) and not checked.drawn["collision_kept"]


def test_the_check_gives_up_after_its_redraws_and_stays_deterministic():
    seen = Seen("оперативне командування «Захід»")
    g = OrgGenerator(random.Random(32), names={})
    calls = []

    def draw() -> str:
        calls.append(g.rng.random())
        return "Захід"

    assert g.fresh(draw, seen) == "Захід" and len(calls) == 1 + MAX_REDRAWS
    assert g.drawn["redraw"] == MAX_REDRAWS and g.drawn["collision_kept"] == 1
    text = "Позов до Міністерства оборони України та ТЦК та СП."
    a = [OrgGenerator(random.Random(33), names={}).institution(military=True, seen=Seen(text)).render() for _ in range(3)]
    b = [OrgGenerator(random.Random(33), names={}).institution(military=True, seen=Seen(text)).render() for _ in range(3)]
    assert a == b


def test_the_fill_step_never_labels_a_name_the_decision_has_in_the_open(monkeypatch):
    raw = ("ОСОБА_1 звернувся з позовом до ІНФОРМАЦІЯ_1 , третя особа - Міністерство оборони України, про стягнення грошового забезпечення. "
           "Зобов'язати ІНФОРМАЦІЯ_2 та ІНФОРМАЦІЯ_3 надати довідку. Відповідь Генерального штабу Збройних Сил України, ТЦК та СП "
           "і Національної гвардії України долучено. Оперативне командування «ІНФОРМАЦІЯ_4» і ВПС «Захід», «Південь», «Схід», «Центр».")
    _force(monkeypatch, {1: ("org", {}), 2: ("org", {}), 3: ("org", {}), 4: ("quoted", {"quote_kind": "formation"})})
    open_names = ("Міністерств[оа] оборони України", "Генеральн(?:ий|ого) штабу? Збройних Сил України", "ТЦК та СП", "Національн(?:а|ої) гварді[яї] України")
    for seed in range(60):
        out = run.Filler(seed=seed).fill(_doc(raw, doc_id=str(seed)))
        *values, formation = _filled(out)
        assert not any(re.fullmatch(rx, v) for rx in open_names for v in values), values
        assert len(set(values)) == 3  # nor the same body for two placeholders
        assert _info(out, 4) == [formation] and formation not in ("Захід", "Південь", "Схід", "Центр")
        assert out == run.Filler(seed=seed).fill(_doc(raw, doc_id=str(seed)))


# ---- names in the quotes of the text -------------------------------------------------------------------------------------
def test_quoted_names_follow_their_kind():
    g = OrgGenerator(random.Random(41), names=NAMES)
    banks = {name.lower() for _, name in BANKS}
    drawn = [g.quoted_name("bank") for _ in range(300)]
    assert all(b.lower() in banks for b in drawn) and {"ПриватБанк", "Ощадбанк", "Універсал Банк", "Райффайзен Банк", "ПУМБ"} <= set(drawn)
    formations = [g.quoted_name("formation") for _ in range(300)]
    assert set(formations) <= set(FORMATION_NAMES) and {"Захід", "Південь", "Чоп"} <= set(formations)
    institutions = [g.quoted_name("institution") for _ in range(400)]
    assert all(re.search(r"лікарня|поліклініка|диспансер|допомоги|[Лл]іцей|Гімназія|школа|освіти|колонія|ізолятор|покарань|виправний центр", i)
               for i in institutions)
    for example in (r"Центральна районна лікарня", r"Міська клінічна лікарня № \d+", r"Ліцей № \d+", r"\w+ виправна колонія \(№ \d+\)"):
        assert any(re.fullmatch(example, i) for i in institutions), example
    assert not any("«" in n or '"' in n for n in drawn + formations + institutions)
    assert "Судова Назва" in {g.quoted_name("company") for _ in range(100)} and "Судова Назва" in {g.quoted_name("unknown") for _ in range(100)}
    assert {k: g.drawn[f"quoted_{k}"] for k in ("bank", "formation", "institution", "company")} == {"bank": 300, "formation": 300, "institution": 400, "company": 200}


@pytest.mark.parametrize("left, expected", [
    ("лікування в КНП «", r"лікарня|поліклініка|диспансер|допомоги"), ("відбуває покарання в ДУ «", r"колонія|ізолятор|покарань|виправний центр|лікарня|диспансер|поліклініка|допомоги"),
    ("навчається у закладі загальної середньої освіти «", r"[Лл]іцей|Гімназія|школа|освіти"),
])
def test_the_words_before_the_quote_choose_the_institution(left, expected):
    g = OrgGenerator(random.Random(42), names={})
    names = [g.quoted_name("institution", left=left) for _ in range(150)]
    assert all(re.search(expected, n) for n in names), [n for n in names if not re.search(expected, n)][:3]
    if "ДУ" in left:
        assert sum(bool(re.search(r"колонія|ізолятор|покарань|виправний центр", n)) for n in names) > 90


def test_the_fill_step_labels_known_quoted_kinds_and_masks_the_rest(monkeypatch):
    raw = ("ОСОБА_1 має рахунок в АТ КБ «ІНФОРМАЦІЯ_1», служив в оперативному командуванні «ІНФОРМАЦІЯ_2», лікувався в КНП «ІНФОРМАЦІЯ_3», "
           "працював у ТОВ «ІНФОРМАЦІЯ_4», змінив прізвище на «ІНФОРМАЦІЯ_5», згадано «ІНФОРМАЦІЯ_6».")
    _force(monkeypatch, {1: ("quoted", {"quote_kind": "bank"}), 2: ("quoted", {"quote_kind": "formation"}), 3: ("quoted", {"quote_kind": "institution"}),
                         4: ("quoted", {"quote_kind": "company"}), 5: ("quoted", {"quote_kind": "person", "mask": True}),
                         6: ("quoted", {"quote_kind": "unknown", "mask": True})})
    out = run.Filler(seed=6).fill(_doc(raw))
    bank, formation, hospital, company, surname, unknown = _filled(out)
    assert bank.lower() in {name.lower() for _, name in BANKS} and formation in FORMATION_NAMES
    assert re.search(r"лікарня|поліклініка|диспансер|допомоги", hospital) and company
    # the formation is the only label; the bank, the hospital and the company are not personal data
    assert _info(out, 2) == [formation] and _civil(out) == [bank, hospital, company] and not any(_info(out, i) for i in (1, 3, 4))
    assert re.fullmatch(r"[А-ЯІЇЄҐ][а-яіїєґ'’-]+(?:-[А-ЯІЇЄҐ][а-яіїєґ'’]+)?", surname) and unknown and not _info(out, 5) and not _info(out, 6)
    for span in out["labels"][1:] + out["masks"]:  # every value stands bare between the quotes of the text
        assert out["text"][span["start"] - 1] == "«" and out["text"][span["end"]] == "»"
    assert out["n_persons"] == 1


# ---- repairs after the review of v6 ----------------------------------------------------------------------------------
def test_the_fill_step_links_every_derived_date_to_its_own_child(monkeypatch):
    raw = ("Стягувати з ОСОБА_1 , ІНФОРМАЦІЯ_1 , аліменти на дітей: ОСОБА_2 , ІНФОРМАЦІЯ_2 , та ОСОБА_3 , ІНФОРМАЦІЯ_3 , "
           "до досягнення ОСОБА_2 повноліття, тобто до ІНФОРМАЦІЯ_4 , а потім до досягнення ОСОБА_3 повноліття, тобто до ІНФОРМАЦІЯ_5 . "
           "Також до досягнення ОСОБА_6 , ІНФОРМАЦІЯ_6 , трирічного віку, тобто до ІНФОРМАЦІЯ_7 .")
    doc = _doc(raw)
    # the birth dates of the children carry no child hint: the link comes from the clause of the derived date
    forced = {1: ("dob", {}), 2: ("dob", {}), 3: ("dob", {}), 4: ("derived", {"until": "majority"}), 5: ("derived", {"until": "majority"}),
              6: ("dob", {}), 7: ("derived", {"until": "age3"})}
    assert run.link_derived(doc["text"], sorted(doc["placeholders"], key=lambda p: p["start"]), forced)[0] == {4: 2, 5: 3, 7: 6}
    _force(monkeypatch, forced)
    for seed in range(30):
        out = run.Filler(seed=seed).fill(doc)
        year = {i: int(re.search(r"\d{4}", _info(out, i)[0]).group()) for i in range(1, 8)}
        assert year[4] == year[2] + 18 and year[5] == year[3] + 18 and year[7] == year[6] + 3
        assert min(year[4], year[5], year[7]) >= REF_YEAR and _info(out, 4) != _info(out, 5) or year[2] == year[3]


def test_a_death_date_comes_after_the_birth_date_next_to_it(monkeypatch):
    raw = "Спадщина після ОСОБА_1 , ІНФОРМАЦІЯ_1 , який помер ІНФОРМАЦІЯ_2 . Раніше ІНФОРМАЦІЯ_3 помер ОСОБА_2 , а ОСОБА_2 , ІНФОРМАЦІЯ_4 , який помер ІНФОРМАЦІЯ_3 ."
    _force(monkeypatch, {1: ("dob", {}), 2: ("death", {}), 3: ("death", {}), 4: ("dob", {})})
    for seed in range(80):
        out = run.Filler(seed=seed).fill(_doc(raw, doc_id=str(seed)))
        born, died, died_first, born_later = (_date(_info(out, i)[0]) for i in range(1, 5))
        assert born < died and born_later < died_first


def _date(value: str) -> dt.date:
    months = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня", "листопада", "грудня"]
    if m := re.match(r"(\d{4})-(\d\d)-(\d\d)", value):
        return dt.date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.match(r"(\d\d)\.(\d\d)\.(\d{4})", value):
        return dt.date(int(m[3]), int(m[2]), int(m[1]))
    m = re.match(r"(\d{1,2}) (\w+) (\d{4})", value)
    return dt.date(int(m[3]), months.index(m[2]) + 1, int(m[1]))


def test_dates_follow_the_words_around_them():
    g = InfoGenerator(random.Random(12))
    war = [g.generate("death", {}, None, "iso", left="дружині загиблого ", right=" у період дії воєнного стану солдата") for _ in range(200)]
    assert all(dt.date.fromisoformat(d) >= WAR_START for d in war)
    assert any(dt.date.fromisoformat(g.generate("death", {}, None, "iso", left="який помер ")) < WAR_START for _ in range(50))
    adults = [g.generate("dob", {}, None, "iso", left="за участю засудженого ОСОБА_1, ") for _ in range(400)]
    assert all(int(d[:4]) <= REF_YEAR - 18 for d in adults)  # no minor in an adult's role
    assert any(int(g.generate("dob", {}, None, "iso", left="ОСОБА_1, ")[:4]) > REF_YEAR - 18 for _ in range(400))  # … but one in ten elsewhere
    minors = [g.generate("dob", {}, None, "iso", left="усиновлювачем неповнолітньої дитини ОСОБА_5, ") for _ in range(200)]
    assert all(int(d[:4]) >= REF_YEAR - 17 for d in minors)  # the cue at the mention the value is drawn at, whatever the hints say


@pytest.mark.parametrize("left, host", [
    ("на своїй офіційній сторінці у facebook за посиланням ", "facebook.com"), ("у соціальній мережі «Фейсбук» на власній сторінці ", "facebook.com"),
    ("у Telegram-каналі за адресою ", "t.me"), ("на сторінці в Instagram ", "instagram.com"), ("на youtube каналі ", "youtube.com"),
])
def test_a_profile_link_is_of_the_network_the_text_names(left, host):
    g, person = InfoGenerator(random.Random(2)), Person("Іван", "Петрович", "Коваль", "masc")
    links = [g.generate("url", {"url_kind": "profile"}, person, left=left) for _ in range(40)]
    links += [g.generate("nick", {"nick_kind": "page"}, person, left=left) for _ in range(40)]
    assert all(host in link for link in links), [link for link in links if host not in link][:3]


def test_web_addresses_follow_the_words_before_them():
    g, person = InfoGenerator(random.Random(3)), Person("Іван", "Петрович", "Коваль", "masc")
    profiles = [g.generate("url", {"url_kind": "profile"}, person) for _ in range(200)]
    assert not any("olx" in p for p in profiles)  # a list of adverts is not a profile
    # no court or government domain as a labelled site: the decisions name them in the open
    assert not any(d.endswith(("gov.ua", "court.gov.ua")) for d in SITE_DOMAINS + NEWS_DOMAINS)
    assert not any("gov.ua" in g.generate("url", {"url_kind": "site"}, person, left="на сайті ") for _ in range(300))
    court = [g.generate("url", {"url_kind": "page"}, person, left="з використанням сервісу «Електронний суд», розміщеного за посиланням: ")
             for _ in range(60)]
    assert all(re.fullmatch(r"https://[a-z]{2}\.[a-z]{2}\.court\.gov\.ua/sud\d{4}/.*", c) for c in court)  # the site of one court, never the bare portal
    named = [g.generate("url", {"url_kind": kind}, person, left="в мережі Інтернет на веб-сайті ZN.UA за адресою: ") for kind in ("site", "page") for _ in range(30)]
    assert all(re.search(r"(?://|^|www\.)zn\.ua", n) for n in named)
    video = [g.generate("url", {"url_kind": "page"}, person, left="розміщено на відео хостингу Yоutube за посиланням ") for _ in range(60)]
    article = [g.generate("url", {"url_kind": "page"}, person, left="у статті, що розміщена за посиланням: ") for _ in range(60)]
    assert sum("youtu" in v for v in video) > 30 and not any("youtu" in a for a in article)
    posts = [g.generate("url", {"url_kind": "page"}, person, left="у публікації на сторінці Facebook за посиланням ") for _ in range(30)]
    assert all("facebook.com" in p for p in posts)


@pytest.mark.parametrize("antecedent, expected", [
    ("з позовом до військової частини А5885", r"(в/ч|ВЧ|військова частина) А5885"),
    ("до Шевченківського районного територіального центру комплектування та соціальної підтримки", r"РТЦК та СП|ТЦК та СП|ТЦК"),
    ("до Харківського обласного територіального центру комплектування та соціальної підтримки", r"ОТЦК та СП|ТЦК та СП|ТЦК"),
    ("до Головного управління Пенсійного фонду України у Хмельницькій області", r"(ГУ ПФУ|ГУПФУ) у Хмельницькій області"),
    ("до Міністерства оборони України", r"Міноборони|МОУ|Міноборони України|МО України"),
    ("до Військово-лікарської комісії Центрального госпіталю Міністерства оборони України", r"ВЛК"),
    ("позов Товариства з обмеженою відповідальністю «Комтех Логістік»", r"ТОВ «Комтех Логістік»"),
    ("до Квартирно-експлуатаційного відділу м. Ужгорода", r"КЕВ м\. Ужгорода"),
    ("до Департаменту кадрової політики Міністерства оборони України", None),  # a department is not the ministry
    ("до 24 окремої механізованої бригади", None), ("до Печерської міської ради", None),
])
def test_a_short_name_is_built_from_the_body_before_the_bracket(antecedent, expected):
    g = OrgGenerator(random.Random(5), names={})
    got = {str(g.short_for(antecedent)) for _ in range(20)}
    if expected is not None:  # the kind of the body decides whether its short name is a label
        civil = "Пенсійного" in antecedent or "Товариства" in antecedent
        assert g.last_kind == ("civil" if "Пенсійного" in antecedent else "company" if civil else "military")
    assert got == {"None"} if expected is None else all(re.fullmatch(expected, v) for v in got), got


def test_the_fill_step_writes_the_short_name_of_the_body_before_it(monkeypatch):
    raw = ("ОСОБА_1 звернувся з позовом до військової частини НОМЕР_1 (далі - ІНФОРМАЦІЯ_1 , відповідач), до ІНФОРМАЦІЯ_2 (далі - відповідач-2, "
           "ІНФОРМАЦІЯ_3 ) та до Печерської міської ради (далі - ІНФОРМАЦІЯ_4 ). Зобов'язати ІНФОРМАЦІЯ_4 та ІНФОРМАЦІЯ_3 вчинити дії.")
    _force(monkeypatch, {1: ("org", {"org_kind": "short"}), 2: ("org", {"org_kind": "party"}), 3: ("org", {"org_kind": "party"}),
                         4: ("org", {"org_kind": "short"})})
    built = 0
    for seed in range(40):
        f = run.Filler(seed=seed)
        out = f.fill(_doc(raw, doc_id=str(seed), kind="КАС"))
        code = re.search(r"військової частини (\S+) \(далі", out["text"]).group(1)
        assert re.fullmatch(rf"(в/ч|ВЧ|військова частина) {code}", _info(out, 1)[0])
        # a body with no usual short name: the random short name is a mask at every mention, a full name stays a label
        assert not _info(out, 4) and len([m for m in out["masks"] if m["reason"] == "info_unsure"]) == 2
        full, short = _info(out, 2), _info(out, 3)
        assert len(full) + len(short) + len(_civil(out)) == 3  # a body that is not military is masked, and so is its short name
        if f.stats["info_short_from_antecedent"] == 2:  # the body drawn for ІНФОРМАЦІЯ_2 has a usual short name
            assert len(short) == (2 if full else 0)
            if full:
                built += 1
                assert len(set(short)) == 1 and short[0] != full[0] and len(short[0]) < len(full[0])
    assert built > 5


def test_a_unit_before_its_parent_body_is_of_that_service_and_does_not_repeat_it():
    g = OrgGenerator(random.Random(21), names={})
    border = [g.institution(org_kind="unit", right=" Державної прикордонної служби України (військова частина НОМЕР_1)").render() for _ in range(200)]
    assert all(re.search("прикордонний загін|регіональне управління", b) and "України" not in b for b in border)
    after_code = [g.institution(org_kind="unit", right=" (військова частина НОМЕР_1) Державної прикордонної служби України").render() for _ in range(100)]
    assert all(re.search("прикордонний загін|регіональне управління", b) for b in after_code)
    guard = [g.institution(org_kind="unit", right=" Національної гвардії України про визнання").render() for _ in range(200)]
    assert not any(re.search("України|прикордонн|територіальної оборони", u) for u in guard)
    army = [g.institution(org_kind="unit", right=" Збройних Сил України").render() for _ in range(200)]
    assert not any(re.search("України|прикордонн|гварді", u) for u in army)
    free = [g.institution(org_kind="unit", right=" (військова частина НОМЕР_1) про визнання").render() for _ in range(300)]
    assert any("України" in u for u in free) and any("прикордонний" in u for u in free)  # no parent named: any unit, with or without one


@pytest.mark.parametrize("left, wanted, banned", [
    ("запобіжний захід тримати в умовах гауптвахти ", "[Вв]ійськова частина [АВТ]\\d{4}|Військової служби правопорядку", "Банк|ТОВ|Пенсійн|лікарн"),
    ("встановлено, що в провадженні ", "поліці|ГУНП|прокурат|прокурор", "Банк|ТОВ |ПрАТ|в/ч"),
    ("отримавши у приміщенні РТЦК та СП повістку, прибув до ", "ТЦК|територіальний центр", "Банк|ТОВ|податков"),
])
def test_a_holder_slot_follows_the_words_before_it(left, wanted, banned):
    g = OrgGenerator(random.Random(31), names={})
    for military in (True, False):
        names = [g.institution(military=military, org_kind="holder", left=left).render() for _ in range(200)]
        assert all(re.search(wanted, n) for n in names), [n for n in names if not re.search(wanted, n)][:3]
        assert not any(re.search(banned, n) for n in names)


def test_a_bank_or_an_operator_holds_records_only():
    g = OrgGenerator(random.Random(32), names={})
    banks = {b for _, b in BANKS} | {o for _, o in MOBILE_OPERATORS}
    work = [g.institution(org_kind="holder", left="Протягом цих років вона працювала в ").render() for _ in range(400)]
    records = [g.institution(org_kind="holder", left="документів, які перебувають у володінні ").render() for _ in range(400)]
    assert not any(inner_name(n) in banks for n in work) and sum(inner_name(n) in banks for n in records) > 60


def test_quoted_names_follow_the_words_before_the_quote():
    g = OrgGenerator(random.Random(41), names=NAMES)
    operators = {name for _, name in MOBILE_OPERATORS}
    got = [g.quoted_name("company", left="у володінні оператора мобільного зв`язку ПрАТ «") for _ in range(100)]
    assert set(got) <= {name for form, name in MOBILE_OPERATORS if form == "ПрАТ"} and len(set(got)) > 1
    assert set(g.quoted_name("company", left="операторів телекомунікацій: ПрАТ «ІНФОРМАЦІЯ_1», ТОВ «") for _ in range(60)) <= operators
    banks = {b for _, b in BANKS}
    # the rules gave «company» (the kind of the last mention); the mention the value is drawn at names the bank
    assert {g.quoted_name("company", left="зі своєї банківської картки АТ «").lower() for _ in range(60)} <= {b.lower() for b in banks}
    assert {g.quoted_name("formation", left="до Оперативного командування «") for _ in range(100)} == set(OPERATIONAL_COMMANDS)
    assert {g.quoted_name("formation", left="Повітряного командування «") for _ in range(100)} == set(AIR_COMMANDS)
    penal = [g.quoted_name("institution", left="відбувають покарання та перебувають в ДУ «") for _ in range(100)]
    assert not any(re.search("лікарн|поліклінік|диспансер|медик", p) for p in penal)
    # three of the four commands stand in the decision: the fourth is taken, not a redraw that may fail
    seen_text = "Оперативне командування «Захід», «Південь» та «Схід» повідомили"
    assert {g.quoted_name("formation", seen=Seen(seen_text), left="оперативного командування «") for _ in range(30)} == {"Північ"}


def test_names_that_are_persons_or_too_short_are_not_drawn():
    assert usable_name("Комтех Логістік") and usable_name("АВТ Баварія")
    assert not usable_name("ТЦ") and not usable_name("Балаклейский водоканал В.П. Трубчанинов") and not usable_name("Баліцького Миколи Васильовича")
    g = OrgGenerator(random.Random(51), names={})
    assert not any(g.tck().render() == "ТЦК та СП" for _ in range(500))  # the bare abbreviation names no office: a short name only


def test_an_institution_inside_the_quotes_of_the_text_is_its_bare_name(monkeypatch):
    assert inner_name("АТ «Укртелеком»") == "Укртелеком" and inner_name('ПП "Дельта"') == "Дельта"
    assert inner_name("Міністерство оборони України") == "Міністерство оборони України"
    raw = "Позов ІНФОРМАЦІЯ_1 про стягнення. Ліки отримано з КНЗ «ІНФОРМАЦІЯ_1», а директор ІНФОРМАЦІЯ_1 заперечив."
    _force(monkeypatch, {1: ("org", {"org_kind": "party"})})
    with_quotes = 0
    for seed in range(60):
        out = run.Filler(seed=seed).fill(_doc(raw, doc_id=str(seed), kind="ГПК"))
        first, quoted, third = _info(out, 1) or _civil(out)
        with_quotes += "«" in first or '"' in first
        assert not re.search(r"[«»\"]", quoted)
        if "«" in first or '"' in first:
            assert quoted == inner_name(first)
    assert with_quotes > 20


def test_the_kept_body_is_counted_once_and_other_apart():
    text = normalize("Позивач просить стягнути з Міністерства оборони України та Генерального штабу Збройних Сил України кошти.")
    g = OrgGenerator(random.Random(61), names={})
    for _ in range(300):
        g.institution(military=True, seen=Seen(text))
        g.institution(military=True, seen=Seen(text), count_as="other")
    kept = lambda slot: sum(v for k, v in g.drawn.items() if k.startswith(f"{slot}_"))  # noqa: E731
    assert kept("party") == 300 and kept("other") == 300 and g.drawn["redraw"] > 0


def test_report_names_the_unit_tail_and_the_other_draws(monkeypatch, tmp_path, capsys):
    import silver.report as report

    (tmp_path / "00_filtered.jsonl").write_text("", encoding="utf-8")
    stats = {"documents": 1, "info_org": 4, "info_other": 2, "info_short_from_antecedent": 3, "org_drawn_party_military": 3, "org_drawn_party_civil": 1,
             "org_drawn_other_military": 1, "org_drawn_other_company": 1}
    (tmp_path / "fill_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    monkeypatch.setattr(report, "SILVER", tmp_path)
    monkeypatch.setattr(report, "step_path", lambda step: tmp_path / f"{step}.jsonl")
    report.run_report()
    capsys.readouterr()
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "by sub-type (labelled and masked together): org 66.7%, other 33.3%" in text  # the short-name counter is no sub-type
    assert "military 75.0%, civil 25.0%, company 0.0%" in text and "no rule typed (masked): military 50.0%, civil 0.0%, company 50.0%" in text
    assert "built from the body before the bracket: 3" in text and "the tail of a unit's name" in text and "social-network account" in text
