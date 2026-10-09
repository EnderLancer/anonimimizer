# Error analysis: where the 8.27 points go

Configuration: mode 3 (`results/full.jsonl`), official metric = NER macro F1, relaxed (IoU ≥ 0.5) = **0.9173**, so **8.27 points are lost**.

```bash
python analysis/error_analysis.py data/test/court_cases.jsonl results/full.jsonl   # writes analysis/out/
```

| File | Contents |
|---|---|
| `out/cases.md`, `out/cases.csv` | **all 180 error cases** with category, root cause, whose fault, cost in F1 points, corpus-consistency check and context |
| `out/strict.csv` | 62 boundary-only mismatches (count for strict F1, not for the final score) |
| `out/numbering.json` | 15 pseudonym-numbering errors (B³ consistency) |
| `out/leaks.json` | 4 persons whose name stays readable (deanonymization risk) |
| `out/oracle_gains.csv` | F1 if a whole category were fixed |
| `out/summary.txt` | console output |

The script reproduces `evaluate.py` exactly (asserted per type: TP/FP/FN). Every FP and FN belongs to exactly one case.

## How a case is counted

Unmatched gold and predicted spans that overlap each other (or overlap an already matched neighbour) form one **case**. Its category:

| category | meaning |
|---|---|
| `missed` | gold span, no prediction touches it (FN) |
| `spurious` | prediction, no gold span touches it (FP) |
| `wrong_type` | boundaries match (IoU ≥ .5), type is different (FN + FP) |
| `wrong_type+bounds` | different type and IoU < .5 |
| `too_narrow` / `too_wide` / `shifted` | same type, IoU < .5 |
| `split` / `merged` | 1 gold ↔ several preds / 1 pred ↔ several golds |
| `absorbed` / `extra_fragment` | FN inside a pred already matched to another gold / FP next to an already matched gold |
| `complex` | anything else |

**Exact cost per error.** For each type `1 − F1 = (FP + FN) / (2·TP + FP + FN)`. So every FP or FN of type *t* costs the same `1 / (4·(2TP+FP+FN))` of macro F1, and the costs add up to exactly 8.27:

| type | TP / FP / FN | F1 | **one error costs** | lost |
|---|---|---|---|---|
| ОСОБА | 2825 / 58 / 2 | .989 | 0.0044 pts | 0.26 |
| АДРЕСА | 227 / 13 / 6 | .960 | 0.0529 pts | 1.00 |
| НОМЕР | 432 / 10 / 15 | .972 | 0.0281 pts | 0.70 |
| ІНФОРМАЦІЯ | 147 / 66 / 33 | .748 | **0.0636 pts** | **6.30** |
| **total** | | **.917** | | **8.27** |

One ІНФОРМАЦІЯ error costs as much as 14.5 ОСОБА errors. 76% of the loss is ІНФОРМАЦІЯ.

## Whose fault

Each case is labelled `model` (pipeline error), `gold` (annotation error), or `ambiguous` (the annotation contradicts itself, so no consistent prediction can be correct). The `у розмітці` column in `cases.md` backs up `ambiguous`. It shows how many times the same string appears in the test set and how many of those times the gold masks it.

| fault | cases | FN | FP | **points** |
|---|---|---|---|---|
| ambiguous (inconsistent annotation) | 86 | 17 | 73 | **5.49** |
| model | 79 | 24 | 58 | **1.67** |
| gold (annotation error) | 15 | 15 | 16 | **1.11** |
| **total** | **180** | **56** | **147** | **8.27** |

So only about **1.7 of the 8.27 points are real pipeline errors**. About 6.6 points come from the annotation.

## All root causes, sorted by cost

| # | fault | category | type | root cause | cases | pts |
|---|---|---|---|---|---|---|
| 1 | ambiguous | spurious | ІНФ | state body masked by `party_orgs` but not in this document's gold. The same body is masked elsewhere: «Міністерства оборони України» masked 25 of 33 times, ГУ Нацполіції / ДПС masked in some documents and not in others. Overall in the test set: 54 state-body TP vs 64 FP | 64 | 4.07 |
| 2 | ambiguous | missed | ІНФ | private company name: «ТОВ "Комтех Логістік"» ×9, «ПрАТ "Дельта-Пром"» ×2, «Експрес-Сервіс». All in 2 documents; companies are left open everywhere else | 13 | 0.83 |
| 3 | gold | wrong_type | ІНФ→ОСОБА | gold marks a full name as ІНФОРМАЦІЯ (116186753 «Соловйова Катерина Олександрівна» ×6, 116158320 «Кушнір Інги Вікторівни» ×4) | 10 | 0.68 |
| 4 | ambiguous | wrong_type+bounds | ІНФ/НОМЕР | «Військова частина А1234» as one ІНФОРМАЦІЯ span (116211933, 116133531); elsewhere gold marks only «А1234» as НОМЕР | 4 | 0.37 |
| 5 | gold | wrong_type | ІНФ→АДРЕСА | gold marks addresses as ІНФОРМАЦІЯ (116234614) | 2 | 0.23 |
| 6 | model | spurious | АДРЕСА | public address: court (`Верховного Суду за адресою: м. Київ` ×4) or institution (Верховна Рада, ДПС, Київстар) | 7 | 0.37 |
| 7 | model | missed | НОМЕР | 7 patterns: enforcement decision number after «від <дата> №» (×3), registration number of real estate (12 digits, ×2), certificate «ПІБ № 81234567» (×2), ЄДРПОУ after «код» / «ідентифікаційний код юридичної особи» (×2), series with hyphen «СА-452581», «НВ-121545» (×2), ID-card passport (9 digits), a private IBAN after «рахунок №» dropped by the treasury-IBAN filter | 13 | 0.36 |
| 8 | model | missed | АДРЕСА | address without the «за адресою / проживає» marker: after «по вулиці…», «на … адресу», «що розташована за адресою: вул. … земельна ділянка» | 4 | 0.21 |
| 9 | model | too_narrow | АДРЕСА | CRF stops at «, м.»: «вул. Миру, буд. 34, кв. 56» without «м. Дніпро, Дніпропетровська область» (also 116258503) | 2 | 0.21 |
| 10 | model | missed | ІНФ | date of birth/death without the «народження» marker: «народилась 27 квітня 1987 року», «12 серпня 2023 року … померла», «(30.03.1990., паспорт…» | 3 | 0.19 |
| 11 | model | spurious | ОСОБА | ECHR case names: «Пономарьов проти України», «Brualla Gomez de la Torre v. Spain», «Zubac», «Юніон Аліментаріа Сандерс С.А.», «PONOMARYOV v. UKRAINE» … | 37 | 0.16 |
| 12 | gold | complex | ІНФ | 116234786: gold cut «15» from «15 листопада 2020 року» and glued «листопада 2020 року» onto the name | 1 | 0.13 |
| 13 | ambiguous | spurious | АДРЕСА | crime-scene address that gold left open (116258697 «м. Київ, вул. Притисько-Микільська, 6», «вул. Гмирі, 3») | 2 | 0.11 |
| 14 | model | spurious | НОМЕР | short number: «28» (day in «військової частини 28 квітня»), «машино-місце № 213» ×2 | 3 | 0.08 |
| 15 | gold | complex | НОМЕР/ОСОБА | 116234813: gold boundary «Петренко» \| «Аліси Володимирівни DF987654» | 1 | 0.07 |
| 16 | ambiguous | spurious | ІНФ | date «19 червня 2023 року» (event date, not personal) | 1 | 0.06 |
| 17 | ambiguous | spurious | НОМЕР | treasury IBAN (masked 1 of 15 times in gold); construction permit number «ХК051210602617» | 2 | 0.06 |
| 18 | model | absorbed | НОМЕР | 116234786: CRF address span swallowed «ділянка № 3845» and cut the plot number in the middle | 1 | 0.03 |
| 19 | model | spurious | ОСОБА | «Верховна Рада» as a person (×4); «імені … полковника Євгенія Пікуса» inside a unit name (×4) | 8 | 0.04 |
| 20 | model | too_narrow | ОСОБА | **bug**, see «Leaks» (В. Пєсков) | 1 | 0.01 |
| 21 | gold | spurious | ОСОБА | «Туйчієвій Одінахон (код 1324567890)», a natural person missing from gold | 1 | 0.004 |

## Not in the 8.27, but still real errors

### Leaked names: 4 of 862 persons

| doc | gold | prediction | cause | fault |
|---|---|---|---|---|
| 116231432 | «Головуючий **В. Пєсков**» (end of doc) | «В» | `regex_persons` pattern «Прізвище І.» matches «Головуючий В.». The ROLE stripper removes «Головуючий» and keeps the lone initial «В.» as a person. The model has nothing, so the surname is never masked. Then `clean_person` drops the dot. **Bug.** A one-letter span should be discarded, and «І. Прізвище» (single initial) needs a pattern | model |
| 116234636 | «Шевченко О. В.,» | «Шевченко О» | The model says «Шевченко О.» and the regex says «Шевченко О. В.». `merge_persons` refuses to extend to the right because `INI_END` matches «Шевченко О.», which was meant for «Іванов І. П. Суд». Then the new dot-trimming rule (b43c0df) treats « В» as the start of a new sentence and cuts the dot. **Bug from the last two changes.** «В.» is left open | model |
| 116149856 | «Бічуй Марії Євгенії» | «Бічуй Марії» | the model drops «Євгенії» (a name, not a patronymic, so `PATR` doesn't catch it either) | model |
| 116234574 | «**адвокат** Мастістий І. А.» | «Мастістий І. А.» | gold includes the role word «адвокат», so the uncovered word counts as a leak | gold |

### Pseudonym numbering: 15 conflicts (B³ F1 .995)

`predict.cluster_persons` merges a mention into a group when it shares at least 2 word stems (prefix match ≥ 4 letters), and a group's stem set **grows by union**. So relatives chain together: the father's name «Ігор» matches the children's patronymic «Ігорівна», and «Кравченк-» matches the shared surname.

| kind | docs | fault |
|---|---|---|
| relatives merged into one pseudonym (same surname + name↔patronymic stem) | 116234748 (Кравченко ×4 people), 116128692 (Щербак), 116258594 (Мороз), 116258619 (Кравець), 116234574 (Борисенко), 116177376 (Олійник) | model |
| same name + patronymic, different surname | 116234645 (Мартинюк / Зінчук Сергій Олександрович) | model |
| different initials merged | 116149927 (Михайлова Н. О. / Н. М.), 116149911 (Олійник А. С. / Святослав Сергійович), 116177386 (Разумовська М. А. / Апатенко М. А.) | model |
| one person split: «Кравець Р. Ю.» vs «Кравця Р. Ю.» (inflected surname + initials) | 116177310 | model |
| gold gives one person two labels | 116234778 «Гварамадзе Левана / Леван Зурабович», 116205945 «Підлубний Е. С.» / «Підлубного Едуарда Станіславовича» | gold |
| gold gives two people one label | 116206025 «Тимченко Олександр Миколайович» = «Тимошенко Максим Олександрович» (116234813 «Шевченко» under Петренко's label is correct: it is her maiden name, «дівоче прізвище Шевченко») | gold |

### Boundary-only errors (strict F1): 62

The full list is in `out/strict.csv` and at the end of `out/cases.md`. Most are gold conventions that the gold itself does not follow consistently:

- АДРЕСА 17×: gold includes the preposition «по / на» («по вулиці …»), which it leaves out elsewhere;
- НОМЕР 12×: gold includes «в/ч» or «№»;
- ОСОБА 8×: trailing «.» or «,». In 4 cases it is the **dot-trimming rule b43c0df misfiring** before a line break or the end of the text: «Фінагеєва В. О.\n», «Литвиненко І. В.\n», «Огородніка К.М.\n», «Могил С.К.»⟨EOF⟩, where gold keeps the dot. In 3 more, gold includes a sentence-final «.» or «,» after a full name;
- ОСОБА 3×: pipe-table artifacts «|І. В. Литвиненко|» (116149911), from the NER model span; it is not stripped;
- ОСОБА 2×: capitalised sentence-start word glued onto the regex name: «Також Опанас Андрійович Савенко», «Оголошено Бабич …»;
- gold noise: «листопада 2020 року Шевченко Тамари Миколаївни», «адвокат Мастістий І. А.», «Шидерової» (without initials), «Підлубного Е. С» (without dot).

## What is worth fixing (model errors only, 1.67 pts available)

| fix | pts | where |
|---|---|---|
| more НОМЕР contexts: «від <дата> №», «номер об'єкта нерухомого майна», «ПІБ №», «код» + 8 digits, «XX-123456», 9-digit passport; do not drop a non-treasury IBAN after «рахунок №» | ≤ .36 | `predict_trf.py` rules |
| drop АДРЕСА after «суду за адресою», «(адреса: … ЄДРПОУ», «Верховної Ради України (…)» | ≤ .37 | `predict_trf.py` |
| CRF address must not stop at «, м.»; extend to city / oblast / postal code | ≤ .21 | `crf_training/` (README item 4) |
| АДРЕСА after «по вулиці», «на адресу» without «за адресою» | ≤ .21 | rules |
| dates «народився/народилась <дата>», «<дата> … помер(ла)», «(<дата>., паспорт» | ≤ .19 | rules |
| ECHR filter («у справі «X проти …», «X v. …») | .16 (+ precision for ОСОБА) | README item 5 |
| short numbers ≤ 3 digits after «машино-місце №» or before a month name | .08 | rules |
| **leak bugs**: discard one-letter person spans; let `INI_END` allow extending «Прізвище І.» → «Прізвище І. П.»; dot-trim only before `\n` when the next line starts a new sentence (or skip it when followed by «[А-Я]\.») | leaks 4 → 2 | `regex_persons.py`, `run.py`, `predict_trf.clean_person` |
| clustering: no union-chaining; a mention with full name + patronymic must not join a group whose full mention has a different first name | B³ | `predict.cluster_persons` |

The other 6.6 points (state bodies, companies, «Військова частина», full names marked ІНФОРМАЦІЯ) cannot be won by any consistent predictor: the same string is masked in one document and left open in another. If only those annotation-side cases were corrected, the **current predictions would score .983** (ІНФОРМАЦІЯ .748 → .992).

## Corrected gold: `data/test/court_cases_fixed.jsonl`

Two data files, edited by hand-reviewed corrections (no code in the project patches the gold):

- `data/test/court_cases_fixed.jsonl` — all 240 test documents in the original schema, with the **30 documents whose gold annotation is wrong** replaced by corrected copies (`annotations`, `annotations_by_pseudonym` and `pseudonymized` consistent with each other). It is a drop-in replacement for `court_cases.jsonl`: same ids, same texts.
- `data/test/court_cases_fixed_keep_state_bodies.jsonl` — the same corrections **except** the `unmask` row below: the 68 state-body mentions stay masked exactly as in the official gold (spans and labels identical to `court_cases.jsonl`). Comparing it with the two other files separates the gain from fixing annotation errors from the gain from the state-body decision.
- `data/test/fix_reasons_for_court_cases.jsonl` — only those 30 corrected documents (one JSON object per line), each with an extra `corrections` list: `action`, `before`, `after`, `description` (why it is wrong) and `legal_reference` (ст. 7 Закону «Про доступ до судових рішень» and/or the ДСА codifier № 59; texts in `legal/`).

| action | fixes | mentions | what |
|---|---|---|---|
| relabel | 3 | 12 | full names marked ІНФОРМАЦІЯ → ОСОБА (2 docs); addresses marked ІНФОРМАЦІЯ → АДРЕСА |
| reboundary_relabel | 2 | 4 | «Військова частина А1234» as ІНФОРМАЦІЯ → «А1234» as НОМЕР (dataset convention) |
| resplit | 3 | 6 | boundary in the wrong place: «15» \| «листопада 2020 року Шевченко…», «Петренко» \| «Аліси Володимирівни DF987654», «Шидерової» \| «Н. С. e-mail» |
| trim / extend | 7 | 7 | role word «адвокат», «, розташованого за адресою:», sentence comma or dot inside the span; an initial's dot dropped mid-sentence |
| merge / split | 3 | 4 | one person with two labels (Гварамадзе, Підлубний); two names under one label (Тимченко / Тимошенко) |
| add | 2 | 5 | unmasked natural person «Туйчієвій Одінахон»; unmasked real-estate registration number (×4) |
| unmask | 13 | 68 | state bodies and military formations marked ІНФОРМАЦІЯ (Міністерство оборони, ТЦК та СП, ГУ Нацполіції / ДПС, ВЛК, військові навчальні центри, …) → not masked |

The first 18 documents also had 1–8 labels with stray whitespace («ОСОБА_1 »), stripped and noted as `normalize_label` (108 such labels in the whole test set; scoring ignores them).

**State bodies.** Article 7 and codifier п. 5 protect information that identifies a *natural person*; a state body is not one, and the official gold itself leaves the same bodies open in other documents (54 masked vs 64 open in our run). So in the corrected gold they are not masked. The trade name of a sole proprietor («ФОП Бондар І. Л. «Експрес-Сервіс»») stays masked: it points to a natural person.

The pipeline follows the same rule for any dataset: it never masks state bodies (rules `ORG_GOV` / `ORG_MIL` in `predict_trf.py` only find them, so ІНФОРМАЦІЯ candidates from rules, the model or the CRF that hit one are dropped). CI scores the same run against all three gold files.

| mode | official gold | corrected, state bodies masked | corrected |
|---|---|---|---|
| model | .2459 | .2465 | .2465 |
| regex | .8736 | .8876 | .9498 |
| model_regex | .8760 | .8902 | .9525 |
| full | .8816 (ІНФОРМАЦІЯ F1 .605) | .8959 (.653) | **.9586** (.904) |

For `full`, fixing annotation errors is worth +.014; not masking state bodies in the gold is worth another +.063.

Before this change (state bodies masked when a party, as in the official gold) `full` scored .9173 on the official gold and .8620 on the corrected one.

Integrity: the official `data/evaluate.py` accepts the file; entity = text slice, no overlaps, `pseudonymized` rebuilds exactly.

**Not changed on purpose**, because these are systematic dataset conventions rather than one-off errors (see `LABELING_RULES.md`):
- judges and enforcement officers are masked, although the codifier exempts officials acting in office;
- «А1234» is НОМЕР;
- e-mail is ІНФОРМАЦІЯ, although the codifier says НОМЕР;
- private companies are masked in some documents only (legal entities, so legally they need no masking either: ТОВ «Комтех Логістік», ПрАТ «Дельта-Пром»);
- an initial's dot is dropped at sentence end;
- ЄСПЛ case names are left open;
- cadastral and bank-contract numbers are left open.
