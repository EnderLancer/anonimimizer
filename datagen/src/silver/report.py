"""Write data/silver/REPORT.md from the pipeline outputs."""
from __future__ import annotations

import json
from collections import Counter

import pyarrow.parquet as pq

from silver.io import SILVER, VARIANT, read_jsonl, step_path


def run_report() -> None:
    lines = ["# Silver dataset report", ""]
    n_src = sum(1 for _ in read_jsonl(step_path("00_filtered")))
    lines += [f"- variant: **{VARIANT}**", f"- documents after filter: **{n_src}**"]
    for name in ("train", "dev", "dev_judges"):
        p = SILVER / f"{name}.parquet"
        if not p.exists():
            continue
        t = pq.read_table(p)
        ann = [a for d in t.column("annotations").to_pylist() for a in d]
        msk = [m for d in t.column("masks").to_pylist() for m in d]
        by_type = Counter(a["type"] for a in ann)
        by_src = Counter(a["source"] for a in ann)
        by_kind = Counter(f"{c}: {k}" for c, k in zip(t.column("corpus").to_pylist(), t.column("kind").to_pylist()))
        by_reason = Counter(m["reason"] for m in msk)
        tot = sum(by_type.values()) or 1
        lines += ["", f"## {name}.parquet", "", f"- documents: {t.num_rows}", f"- annotations: {len(ann)}; masks: {len(msk)}",
                  "- by corpus and kind of decision: " + ", ".join(f"{k} {v}" for k, v in by_kind.most_common()),
                  "- by type: " + ", ".join(f"{k} {v} ({v / tot:.1%})" for k, v in by_type.most_common()),
                  "- by source: " + ", ".join(f"{k} {v}" for k, v in by_src.most_common()),
                  "- masks by reason: " + ", ".join(f"{k} {v}" for k, v in by_reason.most_common())]
    fs = SILVER / "fill_stats.json"
    if fs.exists():
        s = json.loads(fs.read_text(encoding="utf-8"))

        def shares(prefix: str, skip: str | tuple[str, ...] = "\0") -> str:
            sel = {k[len(prefix):]: v for k, v in s.items() if k.startswith(prefix) and not k.startswith(skip)}
            total = sum(sel.values()) or 1
            return ", ".join(f"{k} {v / total:.1%}" for k, v in sorted(sel.items(), key=lambda x: -x[1]))

        def drawn(slot: str) -> str:
            """Kinds of body drawn for one kind of slot: «military 95.1%, civil 3.9%, company 1.0%»."""
            sel = {k: s.get(f"org_drawn_{slot}_{k}", 0) for k in ("military", "civil", "company")}
            total = sum(sel.values())
            return ", ".join(f"{k} {v / total:.1%}" for k, v in sel.items()) if total else "none"

        # the sub-type counters only: info_<subtype>, without the breakdowns that share the prefix
        info_other = ("info_org_", "info_masked_", "info_orgkind_", "info_quotekind_", "info_short_", "info_civil_")
        n_info = sum(v for k, v in s.items() if k.startswith("info_") and not k.startswith(info_other)) or 1
        n_masked = sum(v for k, v in s.items() if k.startswith("info_masked_"))
        n_civil = sum(v for k, v in s.items() if k.startswith("info_civil_masked_"))

        n_addr = sum(v for k, v in s.items() if k.startswith("addr_kind_")) or 1
        lines += ["", "## Synthetic values (fill step, before excerpts and the split)", "",
                  f"- addresses: {n_addr}; settlement: {shares('addr_kind_')}",
                  f"- address table: {shares('addr_src_')}",
                  f"- address begins with: {shares('addr_order_')}",
                  "- address parts: " + ", ".join(f"{name} {s.get(key, 0) / n_addr:.1%}" for name, key in (
                      ("settlement named in the decision", "addr_from_doc"), ("index", "addr_with_zip"),
                      ("apartment", "addr_apt"), ("no street", "addr_no_street"), ("district of a city", "addr_city_district"))),
                  f"- НОМЕР placeholders: {shares('num_')}",
                  f"- ІНФОРМАЦІЯ placeholders by sub-type (labelled and masked together): {shares('info_', skip=info_other)}",
                  f"- ІНФОРМАЦІЯ placeholders whose value is masked, not labelled (reason info_unsure): {n_masked} of {n_info} "
                  f"({n_masked / n_info:.1%}); by sub-type: {shares('info_masked_') or 'none'}",
                  f"- ІНФОРМАЦІЯ placeholders whose value is a civil body, a company, a bank or an institution: masked, not labelled "
                  f"(reason info_civil_org): {n_civil} of {n_info} ({n_civil / n_info:.1%}); by sub-type: "
                  f"{shares('info_civil_masked_') or 'none'}",
                  f"- institution slots (sub-type org) by kind of slot: {shares('info_orgkind_') or 'none'}",
                  f"- bodies kept for a party / short-name slot: {drawn('party')} / {drawn('short')}; for a holder slot: {drawn('holder')}; "
                  f"for a placeholder no rule typed (masked): {drawn('other')}; "
                  f"a name that already stood in the decision was drawn again {s.get('org_drawn_redraw', 0)} times "
                  f"and kept {s.get('org_drawn_collision_kept', 0)} times",
                  f"- short names after «далі -» built from the body before the bracket: {s.get('info_short_from_antecedent', 0)}",
                  f"- names in the quotes of the text by kind: {shares('info_quotekind_') or 'none'}",
                  f"- institutions / companies (org, other) by kind of decision: {shares('info_org_')}; "
                  f"documents with a labelled institution: {s.get('org_info_docs', 0)}, of them filled once more because of it: "
                  f"{s.get('org_info_refilled', 0)}",
                  f"- source of civil-body and company names (military bodies are always composed): {shares('org_name_') or 'word lists only'}",
                  f"- date format of a document: {shares('date_doc_style_')}"]
    m = SILVER / "audit" / "metrics.json"
    if m.exists():
        a = json.loads(m.read_text(encoding="utf-8"))
        lines += ["", "## NER audit", "", f"- docs: {a['docs']}, PERS total: {a['pers_total']}",
                  f"- holes_rate: {a['holes_rate']:.4f}, masked_rate: {a['masked_rate']:.4f}, labeled_rate: {a['labeled_rate']:.4f}",
                  "- unconfirmed by source: " + ", ".join(f"{k} {v:.3f}" for k, v in a["unconfirmed_rate_by_source"].items())]
    lines += ["", "## Known limitations", "",
              "- grammatical case of synthetic names is guessed from the left context (heuristic); mismatches remain in a minority of mentions",
              "- address values come from the Ukrposhta directory and, when its table was built, from OpenStreetMap (settlement, "
              "district, street, index and house number of one row); apartments and the order / wording of the parts are generated; "
              "districts of cities exist for OpenStreetMap rows only",
              "- an ІНФОРМАЦІЯ placeholder in the position of a party («позовом до …», «касаційна скарга …», «зобов'язати …») is filled "
              "with an institution - a military body in 95 % of the slots of a serviceman's case, a company in 80 % of a commercial "
              "one - declined by the case its context asks for; the unit's own name where its code stands next to the slot (of the "
              "service the text names after it), a short undeclined name after «далі -», built from the body before the bracket "
              "where that body has a usual short name; names of military bodies are composed from word lists (no public register names "
              "them); civil bodies and companies are partly real names - from court decisions, the Treasury's register of budget "
              "holders and the register of legal entities - when scripts/fetch_orgs.py has built their table; courts are never "
              "drawn as a labelled value, and a name that already stands in the decision is drawn again (up to 5 times)",
              "- a placeholder whose context does not tell the kind of the hidden value (no rule matched, a date not tied to birth or "
              "death, a login, a social-network account, a settlement, a surname or an unrecognised name in quotes, the body of a "
              "judge / prosecutor / bank card, the tail of a unit's name («прикордонного загону …»), a short name after «далі -» "
              "of a body that has no usual one, a link after a video-conference cue, a death date after a weak cue) still gets "
              "a value so that the text reads naturally - an institution where no "
              "rule matched - but the value is a mask (reason info_unsure), not a label: nothing is learnt from it, in either "
              "direction; the share of such placeholders is in the fill statistics above",
              "- only a military body is a label (the task statement lists «назва військової частини» under ІНФОРМАЦІЯ): a civil "
              "body, a company, a bank, a hospital or a school that stands in an anonymised slot identifies no natural person, and "
              "the same names stand open in most decisions, so its value is a mask (reason info_civil_org) at every mention",
              "- a value is drawn at the first mention of its placeholder, with the hints of the last one that has the winning "
              "sub-type: where the mentions disagree (an institution that is also mentioned inside quotes, a bank named as a "
              "company elsewhere) the value fits one of them only; a birth date and a death date agree only when they stand next "
              "to each other, and a derived date follows a child's birth date only when the text links the two",
              "- documents are excerpts: the heading is kept in about 15 % and the signature block in about 13 % of them",
              "- open IBAN/phone/plate values without a clear personal context are masked, not labelled",
              "- this module does not read data/test; the 250 decisions the test set was cut from are used only as an exclusion list "
              "(their ids, their cases and near-copies are dropped)",
              "- an open enforcement-proceeding number («виконавче провадження № …», «ВП № …», АСВП, ЗВП; every number of a "
              "list) is labelled НОМЕР only when a participant who is a natural person is visible - a filled ОСОБА placeholder "
              "within 300 characters or in the heading, or an open name after «ФОП / боржник / стягувач»; otherwise it is masked "
              "(reason vp_no_person). In the source corpus Supreme Court clerks hide this number in 93 % of its occurrences, "
              "local courts in 16 %, and about one open number in six belongs to a proceeding with no natural person in sight",
              "- two choices rest on knowledge about the test set: the shares of date formats and the cutting of excerpts follow "
              "an outside report that measured the test set"]
    (SILVER / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
