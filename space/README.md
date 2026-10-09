---
title: Псевдонімізація судових рішень
emoji: 🕶️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: "3.12"
short_description: Pseudonymization of Ukrainian court decisions
startup_duration_timeout: 1h
license: mit
models:
  - dchaplinsky/uk_ner_web_trf_13class
---

# Псевдонімізація українських судових рішень

Гібридне рішення: правила (regex) + NER-модель для ОСОБА + CRF для НОМЕР / ІНФОРМАЦІЯ / АДРЕСА.
Опис, метрики й код: [github.com/EnderLancer/anonimimizer](https://github.com/EnderLancer/anonimimizer).

Space оновлюється автоматично з гілки `main` (GitHub Actions → **Deploy Space**); редагувати файли тут немає сенсу.

NER-модель задається в інтерфейсі рядком `тип:repo[?опції]`:

- `spacy:dchaplinsky/uk_ner_web_trf_13class` — за замовчуванням;
- `hf:<repo>?aggregation=first&min_score=0.6` — будь-яка token-classification модель;
- `gliner:<repo>?labels=person,address&threshold=0.4` — GLiNER.

Мітки моделі переводяться в типи полем «Мітки моделі -> типи» (`LOC=АДРЕСА`, `ORG=`, `@pii`).

API: `gradio_client.Client("<space>").predict(text, "full", ner, labels, api_name="/pseudonymize")`.
