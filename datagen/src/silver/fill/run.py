"""Step 2: replace placeholders with synthetic values, producing text + labelled spans."""
from __future__ import annotations

import json
import logging
import random
from collections import Counter

from silver.fill.addresses import AddressGenerator
from silver.fill.context import guess_case, guess_gender, info_subtype, number_subtype
from silver.fill.info import InfoGenerator
from silver.fill.numbers import NumberGenerator
from silver.fill.persons import PersonGenerator
from silver.io import SILVER, VARIANT, read_jsonl, step_path, write_jsonl

log = logging.getLogger("silver")
# how many times a Supreme Court decision is filled, each time with other synthetic values
VS_FILLS = {"base": 1, "large": 2}[VARIANT]


class Filler:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.persons = PersonGenerator(self.rng)
        self.addresses = AddressGenerator(self.rng)
        self.numbers = NumberGenerator(self.rng)
        self.info = InfoGenerator(self.rng)
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
        infos: dict[int, tuple[str, str, dict]] = {}
        seen_person: set[int] = set()

        # decide НОМЕР / ІНФОРМАЦІЯ sub-types from the most informative occurrence
        num_sub: dict[int, str] = {}
        info_sub: dict[int, tuple[str, dict]] = {}
        for p in phs:
            left, right = text[max(0, p["start"] - 60): p["start"]], text[p["end"]: p["end"] + 40]
            if p["type"] == "НОМЕР":
                st = number_subtype(left, right)
                if st != "generic" or p["idx"] not in num_sub:
                    num_sub[p["idx"]] = st
            elif p["type"] == "ІНФОРМАЦІЯ":
                st, hints = info_subtype(left, right)
                if st != "other" or p["idx"] not in info_sub:
                    info_sub[p["idx"]] = (st, hints)

        # one address area / style and one date format per document
        addr_ctx = None
        date_style = self.info.doc_date_style()
        # a person to tie e-mails / nicknames to: the one mentioned most recently before the placeholder
        out, labels, masks, pos = [], [], [], 0
        last_person = None
        for p in phs:
            out.append(text[pos: p["start"]])
            start = sum(len(s) for s in out)
            t, idx = p["type"], p["idx"]
            mask_reason = None
            if t == "ОСОБА":
                if idx not in people:
                    people[idx] = self.persons.new_person(guess_gender(text[max(0, p["start"] - 60): p["start"]]))
                person = people[idx]
                case = self.persons.choose_case(guess_case(text[max(0, p["start"] - 60): p["start"]]))
                value = self.persons.render(person, case, first_mention=idx not in seen_person)
                seen_person.add(idx)
                last_person = person
                self.stats[f"case_{case}"] += 1
            elif t == "АДРЕСА":
                if idx not in addrs:
                    addr_ctx = addr_ctx or self.addresses.doc_context(text)
                    a = addrs[idx] = self.addresses.new_address(addr_ctx)
                    self.stats.update([f"addr_kind_{a['kind']}", f"addr_order_{a['style']['order']}"]
                                      + [f"addr_{k}" for k in ("from_doc", "with_zip", "apt", "no_street") if a[k]])
                    value = self.addresses.render(a)
                else:
                    value = self.addresses.render(addrs[idx], variant=self.rng.random() < 0.3)
            elif t == "НОМЕР":
                if idx not in nums:
                    nums[idx] = self.numbers.generate(num_sub.get(idx, "generic"), text[max(0, p["start"] - 20): p["start"]])
                    self.stats[f"num_{num_sub.get(idx, 'generic')}"] += 1
                value = nums[idx]
            else:  # ІНФОРМАЦІЯ
                st, hints = info_sub.get(idx, ("other", {}))
                if idx not in infos:
                    infos[idx] = (st, self.info.generate(st, hints, last_person, date_style), hints)
                    self.stats[f"info_{st}"] += 1
                    if st in ("dob", "death"):
                        self.stats[f"date_doc_style_{date_style}"] += 1
                value = infos[idx][1]
            out.append(value)
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

    def gen():
        for i, doc in enumerate(read_jsonl(step_path("00_filtered"))):
            if limit and i >= limit:
                break
            for k in range(1, (VS_FILLS if doc.get("kind") == "ВС" else 1) + 1):
                yield filler.fill(doc, fill=k)

    n = write_jsonl(step_path("01_filled"), gen())
    stats = dict(sorted(filler.stats.items()))
    (SILVER / "fill_stats.json").write_text(json.dumps({"documents": n, **stats}, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("filled %d documents; %s", n, stats)
