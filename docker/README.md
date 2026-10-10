# Docker-образ для локального інференсу

Образ містить усе, що потрібно для псевдонімізації: правила, NER-модель з Hugging Face
(експортована в ONNX, ваги лежать усередині образу) і залежності. Інференс моделі
через ONNX Runtime (CPU), без torch і transformers в образі, без мережі, без GPU.

## Запуск готового образу

```bash
docker load -i pseudonymizer.tar                 # один раз, з архіву
docker run --rm -v "$PWD":/data pseudonymizer input.jsonl output.jsonl
```

Тека з даними монтується в `/data`; відносні шляхи рахуються від неї. Без аргументів
читається `input.jsonl`, пишеться `output.jsonl`. Формат входу — рядок JSON на документ
`{"id": ..., "original": "текст"}`, виходу — `{"id": ..., "annotations": [{"start", "end", "pseudonym"}]}`
(як у `run.py`).

Інші параметри `run.py` передаються далі, режим — змінною `MODE`:

```bash
docker run --rm -e MODE=model -v "$PWD":/data pseudonymizer input.jsonl output.jsonl         # лише модель, без правил
docker run --rm --network none --memory 8g -v "$PWD":/data pseudonymizer                      # як на перевірці
```

## Збірка

```bash
docker/build.sh                      # образ для цієї машини + архів dist/pseudonymizer.tar
docker/build.sh linux/amd64          # образ для x86-64 (на Apple Silicon — через емуляцію, повільніше)
NER_MODEL=org/model docker/build.sh  # інша модель з Hugging Face (ті самі мітки ОСОБА/АДРЕСА/НОМЕР/ІНФОРМАЦІЯ)
HF_TOKEN=hf_… docker/build.sh        # приватна модель: токен іде як секрет збірки, в образ не потрапляє
ONNX_QUANTIZE=0 docker/build.sh      # модель у fp32 замість int8 (вихід збігається з torch, але повільніше)
```

Те саме руками: `docker build -f docker/Dockerfile -t pseudonymizer .` і `docker save pseudonymizer -o pseudonymizer.tar`.
Мережа потрібна лише під час збірки (pip і ваги моделі). Збірка має три стадії: ваги з Hugging Face ──►
експорт у ONNX (`docker/export_onnx.py`: torch + transformers, logits ONNX звіряються з torch,
за замовчуванням динамічна int8-квантизація ваг) ──► кінцевий образ лише з `onnxruntime`, `tokenizers`
і `numpy`. Наприкінці збірки образ сам проганяє модель на одному реченні,
тож зламаний образ не збереться.

Модель за замовчуванням — `davidmelash/wechsel_base_v7_2026-10-09_r5_s43` (та сама, що в CI,
змінна `NER_MODEL` репозиторію). Образ ~0,7 GB (int8-модель 125 MB; з fp32-моделлю ~1,4 GB),
архів `docker save` ~0,2 GB.

Адаптер `onnx:` у `ner_adapters.py` працює і поза Docker: `python run.py cases.jsonl out.jsonl --ner onnx:тека`,
де тека — вихід `docker/export_onnx.py` (model.onnx, config.json, tokenizer.json). Вихід збігається
з `hf:` з `aggregation=simple`: групування токенів відтворено як у `transformers.pipeline`.

## Перевірка на тесті

`docker run --rm --network none --memory 8g --cpus 4 …` на 240 документах тесту з main
(`data/test/court_cases.jsonl`, лише `id` і `original`), оцінка `data/evaluate.py` з main;
Apple M-серія, VM colima на 4 ядрах arm64, режим `model_regex`, образ за замовчуванням (ONNX int8):

| Що | Значення |
|---|---|
| Час | 291 с (~1,2 с/документ; ліміт задачі — 2 години) |
| Пік пам'яті контейнера | 0,5 GB за `docker stats` (ліміт задачі — 8 GB) |
| Образ | 666 MB (архів `docker save` ~180 MB) |
| Бал, офіційний тест | 0,9011 (CI з тією ж моделлю в torch fp32: 0,9023) |
| … `court_cases_fixed.jsonl` | 0,8978 (CI: 0,8990) |
| … `court_cases_fixed_state_bodies_open.jsonl` | 0,9587 (CI: 0,9605) |
| Ризик деанонімізації (за іменем) | 0,007 (6 з 864) |

Різниця з CI (−0,001…−0,002) — ціна int8-квантизації. У fp32 (`ONNX_QUANTIZE=0`) вихід адаптера `onnx:`
байт у байт збігається з `hf:` (torch), але на arm64 ONNX Runtime у fp32 приблизно в 5 разів повільніший
за int8 (немає fused attention для графа transformers 5; на x86-64 очікується паритет fp32 з torch і
виграш int8 з VNNI). Вихід детермінований: два запуски дають однаковий файл. Образ для `linux/amd64`,
зібраний і запущений через емуляцію, на перших 10 документах дає той самий вихід, що й нативний.

## Файли

| Файл | Призначення |
|---|---|
| `docker/Dockerfile` | три стадії: ваги моделі (`huggingface_hub`) → експорт у ONNX (torch, transformers) → образ з onnxruntime і tokenizers |
| `docker/export_onnx.py` | експорт token-classification моделі у ONNX, звірка з torch, int8-квантизація (`--quantize`) |
| `docker/entrypoint.sh` | точка входу: шляхи відносно `/data`, чистить кеш моделі, викликає `run.py --ner onnx:/app/models/ner --mode $MODE` |
| `docker/requirements.txt` | залежності кінцевого образу |
| `docker/build.sh` | збірка + `docker save` в архів |
| `.dockerignore` | у контекст збірки йдуть лише модулі пайплайну |
