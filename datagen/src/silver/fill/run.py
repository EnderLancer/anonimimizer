"""Step 2: replace placeholders with synthetic values, producing text + labelled spans."""
from __future__ import annotations

import logging
import random
from collections import Counter

from silver.fill.addresses import AddressGenerator
from silver.fill.context import guess_case, guess_gender, info_subtype, number_subtype
from silver.fill.info import InfoGenerator
from silver.fill.numbers import NumberGenerator
from silver.fill.persons import PersonGenerator
from silver.io import read_jsonl, step_path, write_jsonl

log = logging.getLogger("silver")


class Filler:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.persons = PersonGenerator(self.rng)
        self.addresses = AddressGenerator(self.rng)
        self.numbers = NumberGenerator(self.rng)
        self.info = InfoGenerator(self.rng)
        self.stats: Counter = Counter()

    def fill(self, doc: dict) -> dict:
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
                    addrs[idx] = self.addresses.new_address()
                    value = self.addresses.render(addrs[idx])
                else:
                    value = self.addresses.render(addrs[idx], variant=self.rng.random() < 0.3)
            elif t == "НОМЕР":
                if idx not in nums:
                    nums[idx] = self.numbers.generate(num_sub.get(idx, "generic"))
                    self.stats[f"num_{num_sub.get(idx, 'generic')}"] += 1
                value = nums[idx]
            else:  # ІНФОРМАЦІЯ
                st, hints = info_sub.get(idx, ("other", {}))
                if idx not in infos:
                    infos[idx] = (st, self.info.generate(st, hints, last_person), hints)
                    self.stats[f"info_{st}"] += 1
                value = infos[idx][1]
                if st == "other":
                    mask_reason = "info_unknown"
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
            yield filler.fill(doc)

    n = write_jsonl(step_path("01_filled"), gen())
    log.info("filled %d documents; %s", n, dict(sorted(filler.stats.items())))
