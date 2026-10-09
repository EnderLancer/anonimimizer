# Аудит опублікованих рішень ЄДРСР

Команди нижче запускаються з кореня репозиторію.


`additional_tasks/audit/audit_registry.py` вибирає 500 випадкових непорожніх рішень із
`data/train/2024-court-decisions.parquet` (відтворювана reservoir-вибірка,
`--seed 42`) та створює локальний HTML-звіт для ручного перегляду.
Імена суддів, адвокатів, нотаріусів та інших службових осіб не виключаються.
Аудит також повертає службові адреси та URL, які звичайний псевдонімізатор
відсіює. Уже знеособлені фрагменти `ОСОБА_n` тощо не показуються.

```bash
.venv/bin/python -m pip install -r additional_tasks/audit/requirements-audit.txt
.venv/bin/python additional_tasks/audit/audit_registry.py
open additional_tasks/audit/out/review.html
```

За замовчуванням працюють regex + локальна CRF, без завантаження NER-моделі.
Цей режим може пропускати імена без по батькові або ініціалів. Якщо NER-модель
встановлено локально, встановіть також transformer-компонент. Для моделі
`models/uk_ner_web_trf_13class` потрібна spaCy `>=3.7.4,<3.8.0` (за її
`meta.json`). Для цього середовища зафіксовано NumPy `<2` та PyArrow
`19.0.1`: встановлена раніше PyArrow `26.0.0` не імпортується з NumPy
`1.26.4`. Сумісні залежності описані у `requirements-full.txt`:

```bash
.venv/bin/python -m pip install -r additional_tasks/audit/requirements-full.txt
.venv/bin/python additional_tasks/audit/audit_registry.py --mode full --limit 2 --out additional_tasks/audit/out-smoke
```

Після пробного запуску можна обробити всі 500 рішень:

```bash
.venv/bin/python additional_tasks/audit/audit_registry.py --mode full --ner spacy:/absolute/path/to/model --out additional_tasks/audit/out-full
# Лише правила, інший розмір вибірки та seed:
.venv/bin/python additional_tasks/audit/audit_registry.py --mode regex --limit 100 --seed 7 --out additional_tasks/audit/out-small
```

Для іншого parquet-файлу передайте його першим аргументом. Колонки за
замовчуванням — `id` і `text` (або `pseudonymized`); їх можна задати через
`--id-column` та `--text-column`. Скрипт читає parquet пакетами та тримає
у пам'яті лише вибрані рішення. NER-кеш окремий, у вихідній папці, з ключами
за моделлю, ID і хешем точного тексту; тестовий `ner_cache.json` не читається.

Fine-tuned Hugging Face token-classification модель з українськими BIO-мітками
`ОСОБА`, `АДРЕСА`, `НОМЕР`, `ІНФОРМАЦІЯ` також підтримується. Для WECHSEL:

```bash
.venv/bin/python -m pip install transformers torch
.venv/bin/python additional_tasks/audit/audit_registry.py --mode full --limit 2 \
  --ner "hf:davidmelash/wechsel_base_v7large_2026-10-09_r4?revision=5c6c4db32dd3def2f05c65ab8173b87b7798884f&aggregation=simple&stride=128" \
  --out additional_tasks/audit/out-wechsel-smoke
```

Для вибірки 500 рішень змініть `--limit 2` на `--limit 500` та вихідну папку
на `additional_tasks/audit/out-wechsel`. За замовчуванням інференс працює на CPU;
на першому запуску завантажуються ваги моделі (~496 МБ). У режимі `full`
кандидати моделі доповнюються правилами та локальною CRF. `revision` фіксує
версію ваг для відтворюваності аудиту.

Результати у `additional_tasks/audit/out/`:

- `review.html` — пошук за значенням/контекстом/ID, фільтри типів і статусів,
  повний текст, посилання на рішення, коментарі та оцінки. Перемикач «Усі
  рішення» дозволяє читати також документи без знайдених фрагментів.
- `candidates.jsonl` — один рядок на рішення, опублікований текст і всі
  кандидати з позиціями, значенням, контекстом та джерелами виявлення.
- `summary.json` — параметри запуску та кількість кандидатів за типами.

Оцінки у HTML зберігаються у локальному сховищі браузера, якщо воно доступне.
Кнопки експорту/імпорту JSON дозволяють зберегти їх у файл і перенести між
браузерами чи повторними прогонами. Ідентичні кандидати з кількох джерел
об'єднуються; фрагменти з різними межами або типами залишаються окремими.
Ці кількості — кандидати, а не підтверджені витоки або унікальні особи.
Звіт і JSON містять відкриті значення з корпусу; зберігайте їх локально,
публічні приклади маскуйте. Папки `out/`, `out-full/` та `out-small/` виключені з Git.

Дані для рендеру вбудовуються скриптом безпосередньо у `review.html` як
JSON у `<script id="data" type="application/json">`. Браузер читає цей JSON;
`candidates.jsonl` або parquet під час відкриття звіту не завантажуються.
Звіт є самодостатнім файлом. Ручні оцінки зберігаються окремо у
`localStorage` браузера; кнопка експорту завантажує `registry-reviews.json`
у папку, вибрану браузером, і не змінює файли результатів скрипта.
