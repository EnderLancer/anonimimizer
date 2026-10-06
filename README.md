# anonimimizer

Pseudonymization of personal data in Ukrainian court decisions: names, addresses, numbers and other
identifying information are replaced with pseudonyms such as `ОСОБА_1`, `АДРЕСА_2`, `НОМЕР_3`, `ІНФОРМАЦІЯ_4`.

## Flow

```
document ─┬─► RegEx ─┐
          └─► Model ─┴─► Hidden values at positions ─► Mapping values ─► Link results ─► Cleaned document
```

RegEx and Model read the original document independently. Each one writes a list of annotations
(hidden values at positions). `solution/link.py` merges the two lists, gives equal values the same
numbered pseudonym and builds the cleaned document.

| Step | Script | Output |
|---|---|---|
| RegEx | `solution/regex_annotate.py` | annotations with `type`, `text`, `metadata.rule` |
| Model | `solution/model_annotate.py` | annotations with `type`, `text`, `metadata.label/score` |
| Mapping values + Link results | `solution/link.py` | merged annotations with `pseudonym` + `pseudonymized` text |
| Strip for evaluation | `scripts/to_submission.py` | `{id, annotations: [{start, end, pseudonym}]}` |
| Check against expected result | `data/evaluate.py` | metrics JSON |
| Result % | `scripts/report.py` | Markdown report |

### Annotation format

Every stage writes JSON Lines, one document per line:

```json
{"id": 116258650,
 "original": "…",
 "metadata": {"stage": "regex"},
 "annotations": [
   {"start": 149, "end": 175, "type": "ОСОБА", "text": "Попова Сергія Олексійовича",
    "metadata": {"source": "regex", "rule": "person_full_name"}}
 ]}
```

`start`/`end` are character offsets in `original` (`end` exclusive). `solution/link.py` adds
`pseudonym` to each annotation, `pseudonymized` to each document and, when RegEx and Model found the
same span, `metadata.agreed_with`.

## Running locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r solution/requirements.txt          # only needed for the Model step

python scripts/split_test_data.py data/test/court_cases.jsonl out/inputs.jsonl   # id + original only
python solution/regex_annotate.py out/inputs.jsonl out/regex.jsonl
python solution/model_annotate.py out/inputs.jsonl out/model.jsonl              # ~4 min on CPU
python solution/link.py out/linked.jsonl out/regex.jsonl out/model.jsonl
python scripts/to_submission.py out/linked.jsonl out/submission.jsonl
python data/evaluate.py data/test/court_cases.jsonl out/submission.jsonl
```

Leave out the Model line and `out/model.jsonl` for a RegEx-only run (about a second, no dependencies).
The order of the files passed to `link.py` is the priority: when spans overlap, the earlier file wins.

## CI

`.github/workflows/evaluate.yml` runs the whole flow on every pull request and push to `main` and
writes the result % to the job summary. Intermediate files (`regex.jsonl`, `model.jsonl`,
`linked.jsonl`, `submission.jsonl`, `result.json`) are uploaded as the `evaluation-results` artifact.

Manual runs (Actions → Evaluation → Run workflow) accept:

- `use_model` — off for a RegEx-only run;
- `ner_model` — any Hugging Face token-classification model (default `ukr-models/uk-ner`);
- `min_score` — fail the run below this score in %. A `MIN_SCORE` repository variable does the same
  for every run.

## Current results

| | Score (NER macro F1) |
|---|---|
| RegEx | 78.71% |
| RegEx + Model | 78.75% |

Per type (RegEx + Model): ОСОБА 98.9%, АДРЕСА 90.3%, НОМЕР 84.3%, ІНФОРМАЦІЯ 41.4%.
The off-the-shelf NER model only contributes names (`PER`); its `ORG` label mostly marks courts and
state bodies, which are not personal data, so it is turned off in `anonymizer/model_stage.py`.
A model fine-tuned on `data/train/2024-court-decisions.parquet` is the obvious next step.

## Repository layout

```
anonymizer/          library code shared by the scripts
  annotations.py     annotation format, JSONL I/O
  regex_stage.py     RegEx rules (named; the name goes to metadata.rule)
  model_stage.py     Hugging Face NER wrapper and label → type mapping
  mapping.py         Mapping values: same value → same pseudonym number
  pipeline.py        merge of stage outputs, Link results
solution/            command-line entry points of the flow
scripts/             test-data split, strip for evaluation, report
data/                challenge data and the official evaluate.py (see data/README.md)
```
