# Space: наступні кроки

Стан на 2026-10-09: Space створено і запущено: https://huggingface.co/spaces/orthodoxus-nihilist/anonimimizer (ZeroGPU, `zero-a10g`). Акаунт `EnderLancer` отримав `402 Payment Required`: ZeroGPU на безкоштовному акаунті доступний лише акаунтам, старшим за 30 днів, з підтвердженим email. Тому Space розгорнуто на акаунті `orthodoxus-nihilist` (створено 2025-05-28, email підтверджено, не PRO). Нижче — що зроблено, а що лишилося відкритим.

Як влаштовано демо й деплой, описано в розділі «Демо (Hugging Face Space)» головного `README.md`.

## 1. Акаунт для Space

Підійде акаунт Hugging Face, який:

- старший за 30 днів і має підтверджений email (безкоштовний: до 2 ZeroGPU-Space), **або**
- має PRO (тоді можна й `cpu-basic`: задайте `HF_SPACE_HARDWARE=cpu-basic`, див. нижче).

Перевірити: `hf auth whoami`, а потім на huggingface.co/settings/billing.

Якщо такого акаунта немає: створіть Static Space (безкоштовний) і подайте заявку на community grant у вкладці Community (заголовок `Apply for a GPU community grant: Academic project`). Після схвалення змініть у `space/README.md` `sdk: static` на `sdk: gradio`.

## 2. Токен і налаштування GitHub

1. Власник акаунта створює токен на huggingface.co/settings/tokens: **Write** (або fine-grained з правом запису в репозиторії свого акаунта).
2. У GitHub: Settings → Secrets and variables → Actions:
   - **Variables** → `HF_SPACE` = `<акаунт>/anonimimizer`;
   - **Secrets** → `HF_SPACE_TOKEN` = токен з кроку 1;
   - за потреби **Variables** → `HF_SPACE_HARDWARE` (за замовчуванням `zero-a10g`). Діє лише під час створення Space, далі залізо змінюють командою `hf spaces settings <space> --hardware <назва>`.

Секрет `HF_TOKEN` (read-токен для приватних моделей у `ner-models.yml`) не чіпаємо: це інший токен.

## 3. Перший деплой

Actions → **Deploy Space** → Run workflow (гілка `main`). Workflow створить Space, якщо його ще немає, і завантажить файли. Далі деплой іде сам після кожної успішної **Evaluation** на push у `main`.

Перший старт довгий: збирання залежностей і завантаження spaCy-моделі (~1,3 ГБ) під час старту. `startup_duration_timeout` у `space/README.md` дорівнює 1 год.

## 4. Перевірка

```bash
hf spaces info <акаунт>/anonimimizer --expand runtime     # стан: BUILDING → RUNNING
hf spaces logs <акаунт>/anonimimizer --build --tail 200   # помилки збирання
hf spaces logs <акаунт>/anonimimizer --tail 200           # старт застосунку, завантаження моделі
```

Виклик API (стан `RUNNING` ще не означає, що застосунок працює):

```python
from gradio_client import Client
c = Client("<акаунт>/anonimimizer")
hl, text, ann, rows = c.predict("Позивач Коваленко Марія Іванівна, РНОКПП 3112408765, проживає за адресою: м. Київ, вул. Січових Стрільців, буд. 21, кв. 14.",
                                "full", "spacy:dchaplinsky/uk_ner_web_trf_13class", "", api_name="/pseudonymize")
print(text)   # очікується [ОСОБА_1], [НОМЕР_1], [АДРЕСА_1]
```

Результати перевірки (через Gradio API, `/pseudonymize`):

| Модель | Результат |
|---|---|
| `spacy:dchaplinsky/uk_ner_web_trf_13class` (за замовчуванням) | ✅ `[ОСОБА_1]`, `[НОМЕР_1]`, `[АДРЕСА_1]` |
| `gliner:urchade/gliner_multi-v2.1?labels=person,address,phone number&threshold=0.4` | ✅ ті самі три мітки |
| `hf:dslim/bert-base-NER?aggregation=first&min_score=0.6` | ❌ до виправлення: `device=None` → CUDA поза `@spaces.GPU`; виправлено, див. п. 5 |
| `hf:Wismut/nym-pii-multilingual?aggregation=first` (`@pii`) | ❌ `Tokenizer class TokenizersBackend does not exist`: модель потребує `transformers` 5.x, а `spacy-transformers` 1.4 тримає `<4.53.3`, див. п. 5 |

Ще не зроблено: звірка розмітки spaCy-моделі з CLI на кількох документах (у логах Space є попередження `strict=False`: ваги завантажено з перевіркою версій вимкненою) і перевірка через `hf` CLI (замість неї використано Gradio API та сторінку логів).

## 5. Ймовірні проблеми

| Симптом | Причина | Що зробити |
|---|---|---|
| Збирання падає на `numpy` / `thinc` | не спостерігалося | — |
| **Збирання падає на `huggingface-hub`** (фактично сталося, `Build error` 2026-10-09) | `sdk_version: 6.30.0` тягне `gradio` з `huggingface-hub>=1.16`, а `transformers<4.53.3` (потрібен `spacy-transformers` 1.4) — `huggingface-hub<1.0`. Прибрати `gliner` це не вирішує | **виправлено**: `sdk_version: 5.50.0` (`gradio` 5.x сумісний з `huggingface-hub<2.0,>=0.33.5`). Перевірено: збірка і запуск успішні |
| Помилка CUDA на `hf:`-моделі (`Low-level CUDA init ... ZeroGPU`) | `pipeline(device=None)` в transformers ставить `device=0` (cuda:0), якщо `torch.cuda.is_available()`; на ZeroGPU це поза `@spaces.GPU` | **виправлено** в `ner_adapters.py`: за замовчуванням CPU (`device=-1`); GPU — явно `&device=0` |
| `hf:Wismut/nym-pii-multilingual` не завантажується (`TokenizersBackend`) | модель вимагає `transformers` 5.x, а `spacy-transformers` 1.4 тримає `<4.53.3`. Це не виправляється без відмови від `spacy-transformers` | відкрито: прибрати пресет або залишити з попередженням. Для `@pii`-моделей використовувати `gliner` або `dslim` |
| Застосунок не стартує: таймаут | довге завантаження моделі | збільшити `startup_duration_timeout` або винести завантаження моделі за замовчуванням з модульного рівня `app.py` |
| Out of memory після кількох моделей | у пам'яті до 2 адаптерів (`lru_cache(maxsize=2)` у `app.py`) | зменшити до 1 |
| Повільна відповідь на довгих рішеннях | інференс на CPU, roberta-large | див. п. 6 |

Помилки ZeroGPU / Gradio: `references/known-errors.md` у [huggingface/skills](https://github.com/huggingface/skills/tree/main/skills/huggingface-spaces).

## 6. Що можна покращити

- **GPU для моделей.** Зараз весь інференс іде на CPU, а `@spaces.GPU` — заглушка, тож квота відвідувачів не витрачається. Для `hf:` / `gliner:` моделей можна виконувати `adapter.predict` всередині `@spaces.GPU` з `device=0`; для spaCy-трансформера потрібен ще `spacy.require_gpu()` і `cupy`, тож простіше лишити CPU.
- **Docker не потрібен**: Gradio SDK збирає Space з `requirements.txt`. Docker-Space не підтримує ZeroGPU і на безкоштовному акаунті недоступний. Має сенс лише з PRO, якщо залежності так і не зберуться через pip.
- **Приклади.** У `app.py` два синтетичні приклади. Справжні документи з `data/test/` у публічний Space не кладемо: вони містять неприховані імена, адреси й номери.
- **Захист від зловживань.** Довжина тексту обмежена (`MAX_CHARS`), але будь-хто може завантажити довільну модель з HF. Якщо це стане проблемою, обмежити `ner` списком `PRESETS` (прибрати `allow_custom_value=True`).
- **Перевірка в CI.** Після `hf upload` можна дочекатися `RUNNING` і викликати `/pseudonymize` через `gradio_client`, щоб workflow падав на зламаному Space.
