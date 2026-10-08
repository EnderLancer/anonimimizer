# datagen — silver training data for the pseudonymizer

Turns pseudonymized ЄДРСР decisions (`data/train/2024-court-decisions.parquet` and the Supreme Court file
`2024-5K-supreme-court-decisions-deduplicated.parquet` of the same Hugging Face dataset) into a
fine-tuning dataset in the test-set schema: text with visible personal data + character spans.

## Setup

```bash
cd datagen
uv sync --extra ner --extra train      # python 3.11; ner = spaCy auditor, train = smoke training
uv run scripts/fetch_resources.py      # name dictionary, Ukrposhta address directory -> data/resources/;
                                       # Supreme Court decisions and the test-pool exclusion list -> data/train/
```

The NER auditor model is unzipped from its wheel into `data/resources/models/` (the wheel's
metadata is not pip-installable):

```bash
mkdir -p ../data/resources/wheels ../data/resources/models
curl -L -o ../data/resources/wheels/uk_ner_web_trf_13class-1.0.0-py3-none-any.whl \
  https://huggingface.co/dchaplinsky/uk_ner_web_trf_13class/resolve/main/uk_ner_web_trf_13class-any-py3-none-any.whl
(cd ../data/resources/models && unzip -q -o ../wheels/uk_ner_web_trf_13class-1.0.0-py3-none-any.whl 'uk_ner_web_trf_13class/*')
```

## Pipeline

```bash
uv run --no-sync scripts/make_silver.py all --skip-ner   # filter -> fill -> label -> excerpt -> audit -> samples -> split -> export -> report
uv run --no-sync scripts/make_silver.py ner --limit 1200  # optional NER auditor (~1 doc/s on CPU)
uv run --no-sync scripts/make_silver.py audit             # holes_rate etc. -> data/silver/audit/
uv run --no-sync scripts/make_silver.py report            # data/silver/REPORT.md
uv run --no-sync scripts/make_silver.py smoke             # end-to-end check: parquet -> xlm-roberta-base -> evaluate.py
uv run --no-sync pytest
```

Steps and intermediate files: `filter` → `00_filtered.jsonl`, `fill` → `01_filled.jsonl`,
`label` → `02_labeled.jsonl`, `ner` → `03_ner.jsonl`, `excerpt` → `04_excerpt.jsonl`, `split` → `split_*.jsonl`, `export` →
`train.parquet`, `dev.parquet`, `dev_judges.parquet`, `dev.jsonl`, `dev_judges.jsonl`.

## Variants

`--variant` chooses the corpus mix and the output folder; every step accepts it.

| variant | folder | general ЄДРСР corpus | Supreme Court decisions | train documents |
|---|---|---|---|---|
| `base` (default) | `data/silver/` | cut down by kind of decision (`KEEP` in `filter.py`) | filled once | 10 955 |
| `large` | `data/silver_large/` | every decision with a rare entity type, a fifth of the petty-offence rulings, 30 % of the decisions with persons only | filled twice, each time with other synthetic values | 23 363 |

```bash
uv run --no-sync scripts/make_silver.py all --skip-ner --variant large
```

Both variants share the same dev set (500 Supreme Court decisions) and the same exclusions. A second fill of a decision has the
id `<ЄДРСР id>-v2`; dev and dev_judges take the first fill only, and no fill of a held-out decision is trained on.

## Output schema (`data/silver/*.parquet`)

| column | meaning |
|---|---|
| `id`, `src_id` | `<ЄДРСР id>-v<fill number>`, ЄДРСР id |
| `corpus`, `kind` | source file (`vs5k`, `edrsr`) and kind of decision (ВС, військові, КАС, ЦПК, ГПК, КПК, КУпАП, інше) |
| `text` | decision with visible PD (synthetic placeholders fills + real public names) — train on this |
| `annotations` | `start, end, type, pseudonym, source` — character spans in `text`, `end` exclusive |
| `masks` | `start, end, reason` — spans whose tokens get label `-100` (PD-like, uncertain) |
| `text_pseudonymized` | `text` with every annotation replaced by its pseudonym |
| `text_source` | the ЄДРСР text as published (normalized, the whole decision, not the excerpt) |
| `n_persons` | distinct ОСОБА pseudonyms |

`dev.jsonl` / `dev_judges.jsonl` use the test schema (`id, pseudonymized, original, annotations`)
so `python ../data/evaluate.py ../data/silver/dev.jsonl <submission>` scores a model on dev
exactly like on the test set. The test set itself is never read by this module; the 250 decisions it was cut
from (`250-deanonymized-court-cases.parquet`) serve as an exclusion list: their ids, other decisions of the same
cases and near-copies are dropped by `filter`, and `export` fails if one of them reaches an output file.

## Data versions

- **v1** (2026-10-06): 6 address templates, 195 embedded streets, ІНФОРМАЦІЯ with unknown context masked.
  Test score of the model trained on it: 0.824 (АДРЕСА recall 0.62, ІНФОРМАЦІЯ recall 0.53).
- **v2** (2026-10-07): 14 address templates incl. «по вулиці …» with inflection, 1 471 real streets and 1 068 settlements
  mined from court addresses in the ЄДРСР corpus, combinatorial institution / ТЦК / military-unit names, ІНФОРМАЦІЯ with
  unknown context filled with an institution (60 %) or a date (40 %) and **labelled**, more НОМЕР formats
  (passport «серія … №», IBAN grouped by 4, phone variants, УНЗР).

- **v3** (2026-10-08): addresses from the Ukrposhta directory (25 615 settlements, 330 713 street rows with real indexes
  and house numbers), placed in the region and settlements the decision names, one address style per document; VIN, IMEI,
  enforcement-proceeding numbers, plates in both alphabets; open «ВП № …» numbers labelled as НОМЕР; one date format per
  document with more ISO dates; Supreme Court decisions (5K file) as the main corpus with the test pool excluded, the
  general corpus cut down by kind of decision; documents cut to excerpts; split by court case, dev from Supreme Court decisions.

## Layout

```
scripts/make_silver.py      CLI (steps above)
scripts/fetch_resources.py  external resources
src/silver/filter.py        document filter, corpus mix, text normalization
src/silver/exclude.py       test-pool exclusion list (ids, cases, near-copies)
src/silver/excerpt.py       cut decisions to the reasoning part
src/silver/fill/            placeholder filling: persons (pymorphy3 inflection), addresses (Ukrposhta directory, region of
                            the document), numbers, info
src/silver/persons.py       open names (judges, lawyers, …) -> ОСОБА
src/silver/numbers.py       РНОКПП / passports / IBAN -> НОМЕР
src/silver/masks.py         e-mails, birth dates, residential addresses, masks
src/silver/ner.py           spaCy auditor layer (cached per document)
src/silver/audit.py         hole analysis, manual-QA samples
src/silver/split.py, export.py, report.py, smoke.py
docs/silver-data-plan.md    rules in detail
tests/                      pytest
```
