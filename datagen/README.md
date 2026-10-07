# datagen — silver training data for the pseudonymizer

Turns the pseudonymized ЄДРСР dump (`data/train/2024-court-decisions.parquet`) into a
fine-tuning dataset in the test-set schema: text with visible personal data + character spans.

## Setup

```bash
cd datagen
uv sync --extra ner --extra train      # python 3.11; ner = spaCy auditor, train = smoke training
uv run scripts/fetch_resources.py      # name dictionary, address lists -> data/resources/
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
uv run --no-sync scripts/make_silver.py all --skip-ner   # filter -> fill -> label -> audit -> samples -> split -> export (~3 min)
uv run --no-sync scripts/make_silver.py ner --limit 1200  # optional NER auditor (~1 doc/s on CPU)
uv run --no-sync scripts/make_silver.py audit             # holes_rate etc. -> data/silver/audit/
uv run --no-sync scripts/make_silver.py report            # data/silver/REPORT.md
uv run --no-sync scripts/make_silver.py smoke             # end-to-end check: parquet -> xlm-roberta-base -> evaluate.py
uv run --no-sync pytest
```

Steps and intermediate files: `filter` → `00_filtered.jsonl`, `fill` → `01_filled.jsonl`,
`label` → `02_labeled.jsonl`, `ner` → `03_ner.jsonl`, `split` → `split_*.jsonl`, `export` →
`train.parquet`, `dev.parquet`, `dev_judges.parquet`, `dev.jsonl`, `dev_judges.jsonl`.

## Output schema (`data/silver/*.parquet`)

| column | meaning |
|---|---|
| `id`, `src_id` | `<ЄДРСР id>-v1`, ЄДРСР id |
| `text` | decision with visible PD (synthetic placeholders fills + real public names) — train on this |
| `annotations` | `start, end, type, pseudonym, source` — character spans in `text`, `end` exclusive |
| `masks` | `start, end, reason` — spans whose tokens get label `-100` (PD-like, uncertain) |
| `text_pseudonymized` | `text` with every annotation replaced by its pseudonym |
| `text_source` | the ЄДРСР text as published (normalized) |
| `n_persons` | distinct ОСОБА pseudonyms |

`dev.jsonl` / `dev_judges.jsonl` use the test schema (`id, pseudonymized, original, annotations`)
so `python ../data/evaluate.py ../data/silver/dev.jsonl <submission>` scores a model on dev
exactly like on the test set. The test set itself is never read by this module.

## Data versions

- **v1** (2026-10-06): 6 address templates, 195 embedded streets, ІНФОРМАЦІЯ with unknown context masked.
  Test score of the model trained on it: 0.824 (АДРЕСА recall 0.62, ІНФОРМАЦІЯ recall 0.53).
- **v2** (2026-10-07): 14 address templates incl. «по вулиці …» with inflection, 1 471 real streets and 1 068 settlements
  mined from court addresses in the ЄДРСР corpus, combinatorial institution / ТЦК / military-unit names, ІНФОРМАЦІЯ with
  unknown context filled with an institution (60 %) or a date (40 %) and **labelled**, more НОМЕР formats
  (passport «серія … №», IBAN grouped by 4, phone variants, УНЗР).

## Layout

```
scripts/make_silver.py      CLI (steps above)
scripts/fetch_resources.py  external resources
src/silver/filter.py        document filter + text normalization
src/silver/fill/            placeholder filling: persons (pymorphy3 inflection), addresses, numbers, info
src/silver/persons.py       open names (judges, lawyers, …) -> ОСОБА
src/silver/numbers.py       РНОКПП / passports / IBAN -> НОМЕР
src/silver/masks.py         e-mails, birth dates, residential addresses, masks
src/silver/ner.py           spaCy auditor layer (cached per document)
src/silver/audit.py         hole analysis, manual-QA samples
src/silver/split.py, export.py, report.py, smoke.py
docs/silver-data-plan.md    rules in detail
tests/                      pytest
```
