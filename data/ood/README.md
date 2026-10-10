# OOD-набір: тексти поза судовим доменом

Невеликий gold-набір для перевірки NER-моделі, навченої на реєстрових рішеннях, на інших жанрах. Файл
`ood_cases.jsonl` має схему тесту (`id`, `original`, `pseudonymized`, `annotations` з `start`, `end`, `entity`,
`pseudonym`), тож `data/evaluate.py` і `run.py` читають його як є; додатково кожен рядок несе `source`
(news / wiki / web), `subset` (підвибірка), `origin`, `subsource`, `url`, `title`, `license`, а кожна анотація — `kind`
(підвид сутності, перелік у `GUIDELINES.md`).

## Джерела

| source | Звідки | Ліцензія | Підвибірки (`subset`) |
|---|---|---|---|
| news | `Goader/kobza`, `source = ukrainian-news` (передрук збірки `zeusfsx/ukrainian-news`) | CC0-1.0 за картою Kobza | war — новини з 24.02.2022 про частини, бригади, ТЦК; money — реквізити, телефони, email; random |
| wiki | `HuggingFaceFW/finewiki`, config `uk` | CC-BY-SA 4.0 (посилання на статтю в `url`) | bio — біографії з датами; military — військові формування; random |
| web | `Goader/kobza`, `source` ∈ {hplt-2.0, fineweb-2, cultura-x, ubertext2.0} | CC0-1.0 за картою Kobza | contacts — адреси, телефони, email; ubertext — UberText 2.0 без новин і Вікіпедії (соцмережі, блоги); random |

Один parquet-шард кожного набору (`000_ukr_corpus_00000.parquet`, `data/ukwiki/000_00000.parquet`) читається
локально; `lang-uk/malyuk` не використано — ті самі тексти є в kobza і finewiki з чіткішими ліцензіями.

Відбір (`build_ood.py sample`): текст без HTML, 400–3 000 символів з обрізанням на межі абзацу, частка кирилиці
≥ 0,7, без російських літер, без ознак судового рішення, без дублів (точних і за 5-грамами). Квоти підвибірок
добираються регулярними виразами за ключовими словами, а не моделлю: відбір моделлю завищив би recall.
Резервуарна вибірка з фіксованим seed, тож повторний запуск на тих самих шардах дає той самий набір.

## Розмітка

Правила — у `GUIDELINES.md`: ті самі чотири типи, що в челенджі, за реєстровою конвенцією з адаптаціями до
інших жанрів (публічні й вигадані особи — ОСОБА; усі військові органи, частини й формування — ІНФОРМАЦІЯ;
телефони компаній — НОМЕР; дати лише народження і смерті).

Розмітку зроблено агентами Claude Code без API (як `person_links.json`):

1. `build_ood.py export` пише `tasks/INSTRUCTIONS.md` і `tasks/<id>.md`.
2. Агенти читають документ і пишуть `answers/<id>.json`: рядки сутностей з типом, підвидом і `id` особи, без
   позицій. Перший прохід — лише текст, без підказок моделі.
3. `build_ood.py import` знаходить входження рядків цілими словами (довший рядок виграє у місці збігу),
   перевіряє типи, призначає псевдоніми за порядком першої появи і пише `ood_cases.jsonl` + `REPORT.md`.
4. `build_ood.py diff gold submission tasks_adj/` зводить розбіжності між gold і виходом пайплайна; другий агент
   вирішує кожну за інструкцією, правлячи `answers/<id>.json`; повторний import.
5. `build_ood.py check --sample N` друкує документи з позначками `[текст|ОСОБА_1/kind]` для ручної перевірки.

Назви файлів задач і відповідей — id документа з `_` замість двокрапки (`news_010a7fca4c4c.json`), щоб репозиторій
клонувався на Windows. `answers/` лишається в git, щоб gold можна було перебудувати; `tasks*/` і `01_candidates.jsonl` регенеруються.

## Оцінка

```bash
python run.py data/ood/ood_cases.jsonl results/ood_model_regex.jsonl --mode model_regex --ner hf:davidmelash/wechsel_base
python data/evaluate_ood.py data/ood/ood_cases.jsonl results/ood_model_regex.jsonl --bootstrap 500
python data/evaluate_ood.py data/ood/ood_cases.jsonl results/ood_model.jsonl results/ood_model_regex.jsonl --names model,model_regex --by subset
```

`evaluate_ood.py` рахує метрики `evaluate.py` на всьому наборі й на кожній групі (`--by source`, `subset`,
`origin`), а також хибні спрацювання на 1 000 символів у документах без сутностей типу, recall за підвидом
(`kind`) і бутстреп-інтервал середнього F1. CI — окремий workflow `.github/workflows/evaluate-ood.yml` (основний `evaluate.yml` не змінено): перед запуском `data/ood/merge_ner_cache.py` дописує кеш spaCy-моделі `data/ood/spacy_ner.json` у `ner_cache.json` робочої копії, далі режими model / regex /
model_regex і пише таблицю в summary; `ner-models.yml` приймає `data=data/ood/ood_cases.jsonl`.

Результати — у розділі «Поза доменом» головного README.
