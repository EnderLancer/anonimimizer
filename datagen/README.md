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
uv pip install osmium shapely          # optional (the «osm» extra): OpenStreetMap as a second address table
uv run --no-sync scripts/fetch_osm.py  # 880 MB extract of Ukraine -> data/resources/osm_addresses.parquet (~2 min after the download)
```

More Supreme Court decisions (optional, read by the pipeline as the corpus `vs_edrsr` when the file exists):
download «Єдиний державний реєстр судових рішень за 2024 рік» from data.gov.ua (`edrsr_data_2024.zip`, 345 MB,
metadata + links to the RTF texts), unzip it, then

```bash
uv run --no-sync scripts/fetch_edrsr.py --dump /path/to/edrsr_data_2024 --dry-run   # what would be fetched
uv run --no-sync scripts/fetch_edrsr.py --dump /path/to/edrsr_data_2024             # ~35k RTF files (~25 min) -> data/train/2024-vs-edrsr.parquet
```

It takes every decision of the administrative chamber in the categories of public / military service, pensions,
social protection and defence (where military bodies are anonymised as ІНФОРМАЦІЯ) and 2 000 random decisions of
each other chamber, drops decisions whose only placeholders are ОСОБА (keeps a quarter) and near-copies, and caches
the RTF files in `data/train/edrsr_rtf/`.

Real organisation names (optional, used by the generator when the table exists):

```bash
uv pip install openpyxl                  # the «orgs» extra
uv run --no-sync scripts/fetch_orgs.py   # 6 MB + 330 MB download, ~3 min -> data/resources/org_names.parquet
```

See «Organisation names» below.

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

## Address tables

The address generator reads every table it finds in `data/resources/`:

| table | script | rows | what it gives |
|---|---|---|---|
| `ukrposhta.parquet` (required) | `fetch_resources.py` | 330 713 | settlement, both districts, street, index, house numbers for 25 615 settlements |
| `osm_addresses.parquet` (optional) | `fetch_osm.py` | 71 948 | the same for 8 580 settlements, plus districts of 39 cities and buildings the post office does not list |

A settlement that is in both tables (94 % of the OpenStreetMap rows) keeps the Ukrposhta spelling, type and districts; there an
address comes from an OpenStreetMap row with probability 0.3 (`P_OSM` in `fill/addresses.py`), because the generator of the test set
used Ukrposhta data. `make_silver.py --addresses ukrposhta` leaves OpenStreetMap out and rebuilds the data exactly as it was
without it. OpenStreetMap data is © OpenStreetMap contributors under the Open Database License: name them when the table or a
dataset built with it is published.

## Organisation names

Institutions and companies that fill ІНФОРМАЦІЯ placeholders are composed from the word lists of `fill/orgs.py`. When
`data/resources/org_names.parquet` exists, a share of them are real names instead:

| source | script reads | names | used for |
|---|---|---|---|
| court decisions | the corpora of `data/train/` without the test pool | 13 157 company names (seen at least twice), 368 state bodies (in at least two decisions) | any decision: 60 % of company names, 50 % of central / regional bodies |
| Treasury register of budget holders | `rrk_*.xlsx`, data.gov.ua | 454 social-protection bodies, 26 prosecutor's offices, 122 courts | servicemen's and social cases only; the courts are in the table but are not drawn since data v6 |
| register of legal entities (ЄДР) | `UO.zip`, data.gov.ua | 33 989 registered companies (one in 25), capitals turned into the spelling of court decisions | commercial cases only: half of the real company names |

Military bodies (the ministry, ТЦК, units, medical commissions) always come from the word lists: no public register names
them, and they are what the test set anonymises most. The kinds of institution and their shares do not change with the
table - only the names do. A real body is declined like a composed one (`real_org`: the words up to the head noun).
Both registers are published under Creative Commons Attribution: name the Treasury and the Ministry of Justice when the
table or a dataset built with it is published. `REPORT.md` shows the share of names from each source.

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
  enforcement-proceeding numbers, plates in both alphabets; open «ВП № …» numbers labelled as НОМЕР when a natural person is a visible participant, masked otherwise; one date format per
  document with more ISO dates; Supreme Court decisions (5K file) as the main corpus with the test pool excluded, the
  general corpus cut down by kind of decision; documents cut to excerpts; split by court case, dev from Supreme Court decisions.
  Since the evening of 2026-10-08 the data is built with the OpenStreetMap table as well (29 % of the addresses, districts of
  cities in 2 %); `--addresses ukrposhta` gives the earlier files.

## Data v6: ІНФОРМАЦІЯ values after the audit

Four readers hand-labelled about 2,300 ІНФОРМАЦІЯ placeholders of the corpus and four verifiers checked every proposed rule
on fresh samples. The fill step chooses the kind of value from the text around a placeholder and labels what it generates;
where the kind was wrong, data v5 trained on a wrong positive example. What changed in the values and the fill step
(`fill/info.py`, `fill/orgs.py`, `fill/run.py`; the context rules of `fill/context.py` changed with them):

- **Masks instead of guesses.** A sub-type whose hints say `mask: True` still gets a natural value, but the value is recorded in
  `masks` with the reason `info_unsure`, not as a label: the loss ignores it. Always masked: a placeholder no rule matched
  («other»), a date not tied to birth or death, a login, a social-network account, a settlement, a surname or an unrecognised
  name in quotes, the body of a judge / prosecutor / bank card; masked on the rule's word: a death date after a weak cue, the
  tail of a unit's name, a link after a video-conference cue or after a link the text already gave; masked by the fill step: a
  short name after «далі -» of a body that has no usual one. Why: with
  the 75 % institution / 25 % date fill about 46 % of the «other» values were of the wrong kind. «other» is now always an
  institution. The label and excerpt steps keep these masks. Two paths of the label step can still put a label on a masked
  value and drop its mask (`label_persons` and `label_numbers` do not look at masks): a value shaped like a person's name, or a
  login shaped like a document number next to a passport cue. The first is closed at the source - person-shaped and
  two-letter names are left out of the name table (`usable_name`) - the second is open (0 cases in 2,250 fills of the
  decisions that have a login). `dev.jsonl` / `dev_judges.jsonl` carry no masks: a masked value is open text there.
- **Repeated fills.** A decision is filled once more only when it has a *labelled* institution-like placeholder (a body, a
  unit's name, a company / bank / formation / institution name in quotes), and one time fewer than before
  (`INFO_ORG_FILLS`: base 1, large 2; was 2 / 3). Why: with the repeats, decisions with a labelled institution were 29-31 % of
  those that mention an organisation (18 % on first fills).
- **Kinds of institution by kind of slot** (`org_kind` of the rules, `KIND_MIX` in `fill/orgs.py`; military / civil / company):
  a party of a serviceman's case 95 / 4 / 1 (was 78 / 14 / 8; the audit found 99 % military among the hidden bodies of known
  kind), of a commercial case 5 / 15 / 80, of any other 70 / 25 / 5; a holder of documents, a place of work or custody (banks,
  hospitals, police units, mobile operators) 60 / 25 / 15 in a serviceman's case and 10 / 45 / 45 elsewhere; the unit's own name
  - a numbered brigade or regiment, a border detachment, a regional directorate, a territorial-defence or training unit - where
  the unit's code stands in the open next to the slot; a short name that does not decline («ТЦК та СП», «в/ч А1234», «ГУ ПФУ в
  Одеській області») after «далі -».
- **No labelled courts, no labelled open names.** Courts are no longer drawn from the Treasury register (data v5 labelled them
  next to 435,854 open court mentions). A body one of whose case forms - or a quoted name whose text - already stands in the
  decision, or was drawn for another placeholder of it, is drawn again, up to 5 times (`Seen`).
- **Names in quotes by `quote_kind`:** a company name (as before), one of 38 real bank names after a bank cue, a direction or a
  place after a military formation («Захід», «Чоп»), a hospital / school / penal-institution name after «КНП» / «КЗ» / «ДУ»
  (the words before the quote choose which). Why: 21 % of the quoted placeholders are formations and institutions, where a
  company name read «оперативного командування «Комтех Логістік»».
- **Dates.** A date computed from a child's birth date («до повноліття, тобто до …») is that child's generated birth date plus
  18, 23 or 3 years, or a date ahead when the decision has none - data v5 wrote a past date there. A birth date keeps the only
  right to «… року народження».
- **Web addresses and accounts by kind:** a site, a page (an article or a video) or a profile; an account name or a page /
  channel link; a login-like identifier (masked).

After the review of v6 (three reviewers read the generated text; `tests/test_info_values.py`, «repairs after the review»):

- **Values follow the words around them.** The first mention of a placeholder gives its value, so the fill step passes the
  text on both sides to the generators. A short name after «далі -» is built from the body before the bracket («військової
  частини А5885 (далі - в/ч А5885)», «… територіального центру комплектування … (далі - ТЦК та СП)»), also when that body is a
  value filled a moment ago. A unit before its parent body («【X】 Державної прикордонної служби України») is a unit of that
  service and does not repeat it. A holder slot after «гауптвахти» is a unit or a body of the Military Law Enforcement Service,
  after «у провадженні» a police unit or a prosecutor's office; a bank or a mobile operator stands only where the text speaks
  of records. A company in quotes after an operator cue is a mobile operator, after a card / account cue a bank; «оперативного
  командування «…»» is one of the four commands. A profile or channel link is of the network the text names; a page after a
  court cue is a page of one court's site; «веб-сайті ZN.UA» gives zn.ua. An institution mentioned inside the quotes of the
  text is written by its bare name.
- **Dates that agree.** A derived date is linked to its birth date before filling (`link_derived`): the date in its own clause,
  the date of the person named there, else the nearest child's date; two children never share a date and none is in the past
  (39 of 43 derived dates of the corpus now equal a generated birth date plus 18 / 23 / 3 years). A death date next to a birth
  date comes after it, and from 24.02.2022 on after a cue of the war. A birth date after «малолітнього / неповнолітньої ОСОБА_n,»
  is a minor's whatever the other mentions say; one after an adult's role is never a minor's.
- **Fewer generic names.** A bare «ТЦК та СП» is no longer drawn as a party (a short name only); court and government domains
  are no longer drawn as a site; the olx user list is no longer a profile.
- **Rules.** «nick» is masked (11 of 18 hand-labelled accounts were right). A joint-stock company in quotes after a card, an
  account or a bank word is `quote_kind` bank (45 of 45 read). The curly pair “ … ” counts as quotes. `url_kind` looks at the
  words after the cue («інтернет-ресурс» is a site), «канал» inside «телеканалу» is no profile cue.
- **Left as it is:** the hints are those of the last mention with the winning sub-type (the contract of the specification), so
  a value fits one mention where the mentions disagree; ISO dates stand before «року» of the text, as the generator of the test
  set writes them; bare national bodies («Міністерства оборони України») are still drawn as parties.

Enforcement-proceeding numbers, after an audit of 319 hand-read occurrences. Supreme Court clerks hide the number of an
enforcement proceeding in 93 % of its occurrences whoever the debtor is, local courts in 16 %; about one open number in six
(mostly commercial cases) belongs to a proceeding with no natural person in sight. The task counts as НОМЕР the identifiers
«що стосуються конкретної фізичної особи», so the label layer (`silver/numbers.py`) labels an open number only when a
participant who is a natural person is visible - a filled ОСОБА placeholder within 300 characters of the number or in the
first 1,500 characters of the decision, or an open name after «ФОП / боржник / стягувач» - and masks it otherwise (reason
`vp_no_person`). Names of judges and other officials in the heading do not count. The pattern now also finds every number of
a list, the cue without «№», «за №», АСВП and ЗВП (737 open numbers in the corpus instead of 606) and skips «ЄДРПОУ ВП
44118658», the code of a company unit. A НОМЕР placeholder right after a bare «ВП №» is filled as a proceeding number.
A value that the fill step masked is never labelled by a later layer (`label_doc` drops such labels).

`fill_stats.json` and `REPORT.md` count the sub-types (`info_<subtype>`), the masked ones (`info_masked_<subtype>`), the kinds
of slot (`info_orgkind_<kind>`), the kinds of quoted name (`info_quotekind_<kind>`), the kinds of body drawn
(`org_drawn_<slot>_<military|civil|company>`: the body that was kept; `org_drawn_other_…` for a placeholder no rule typed),
the redraws (`org_drawn_redraw`, `org_drawn_collision_kept`) and the short names built from the body before the bracket
(`info_short_from_antecedent`).

## Layout

```
scripts/make_silver.py      CLI (steps above)
scripts/fetch_resources.py  external resources
scripts/fetch_osm.py        OpenStreetMap address table (optional)
scripts/fetch_orgs.py       real organisation names: court decisions, Treasury register, register of legal entities (optional)
src/silver/filter.py        document filter, corpus mix, text normalization
src/silver/exclude.py       test-pool exclusion list (ids, cases, near-copies)
src/silver/osm.py           OpenStreetMap extract -> address table (used by scripts/fetch_osm.py)
src/silver/orgnames.py      harvesting and cleaning of organisation names (used by scripts/fetch_orgs.py), the table loader
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
