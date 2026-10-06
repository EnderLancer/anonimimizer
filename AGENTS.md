# AGENTS.md

Guidance for coding agents working in this repository. Read README.md first for the flow and formats.

## What this project is

A pseudonymizer for Ukrainian court decisions. Two independent stages (RegEx, Model) annotate the
original text; `solution/link.py` merges their annotations and numbers the pseudonyms;
`scripts/to_submission.py` strips everything except `id`/`start`/`end`/`pseudonym`; `data/evaluate.py`
scores the result. The score that matters is `final_score` (NER macro F1, IoU ≥ 0.5) from `evaluate.py`.

## Rules

- **Do not modify `data/`.** `data/evaluate.py` is the official scorer and `data/test/` is the expected
  result. Changing either invalidates every score.
- **Stages never see the expected result.** They read `out/inputs.jsonl` (`id` + `original` only, made by
  `scripts/split_test_data.py`). Do not read `annotations`/`pseudonymized` from `data/test/` in stage code.
- **Stage outputs carry no pseudonym numbers.** RegEx and Model emit `start`, `end`, `type`, `text`,
  `metadata`. Numbering belongs to `anonymizer/mapping.py`, called only from `solution/link.py`.
- **Offsets are into `original`**, `end` exclusive. Never annotate a modified/masked copy of the text.
- **Record provenance in `metadata`.** RegEx annotations have `source: "regex"` and `rule: <rule name>`;
  Model annotations have `source: "model"`, `label`, `score`. Keep this when adding rules or models.
- Code comments and docstrings are in Ukrainian; match the surrounding style.
- The RegEx stage and everything after it use only the standard library. Heavy dependencies
  (torch, transformers) belong to the Model stage and `solution/requirements.txt` only.

## How to check a change

RegEx-only loop (about a second):

```bash
mkdir -p out
python scripts/split_test_data.py data/test/court_cases.jsonl out/inputs.jsonl
python solution/regex_annotate.py out/inputs.jsonl out/regex.jsonl
python solution/link.py out/linked.jsonl out/regex.jsonl
python scripts/to_submission.py out/linked.jsonl out/submission.jsonl
python data/evaluate.py data/test/court_cases.jsonl out/submission.jsonl
```

Add the Model stage (`solution/model_annotate.py`, ~4 min on CPU, needs `solution/requirements.txt`)
when the change touches the model, the merge or the mapping. Report the before/after score and the
per-type table in your summary; a change that lowers `final_score` needs a stated reason.

To see what a stage misses, compare `out/linked.jsonl` against `data/test/court_cases.jsonl` by
`id`: each linked annotation's `metadata` tells you which rule or model label produced it.

## Where to make changes

| Goal | File |
|---|---|
| New or better pattern | `anonymizer/regex_stage.py` — add a named entry to `PERSON`, `NUMBER` or `INFO`; addresses are built in `_addresses` |
| Different model / label mapping / threshold | `anonymizer/model_stage.py` (`DEFAULT_LABELS`, `min_score`) or `--model` / `--min-score` |
| Overlap resolution between stages | `anonymizer/pipeline.py` (`merge`, priority = order of files given to `link.py`) |
| Which mentions share a pseudonym | `anonymizer/mapping.py` (`person_key`, `value_key`) |
| CI steps | `.github/workflows/evaluate.yml` |

When adding a new stage, write the same annotation format (see `anonymizer/annotations.py`) with
`metadata.stage` set on the document row, and pass its file to `solution/link.py` in priority order.
