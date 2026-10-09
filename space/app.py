"""Демо псевдонімізації українських судових рішень (Hugging Face Space).

Текст ──► NER-модель (будь-яка: spacy / hf / gliner, див. ner_adapters.py) ──► мітки -> типи ──► складання частин
     ──► regex-особи + правила ──► псевдоніми ОСОБА_1, АДРЕСА_1…

У Space модулі пайплайну лежать поруч (їх копіює .github/scripts/build_space.sh), локально — на рівень вище:
python space/app.py
"""
import importlib.util
import os
import re
import sys
from functools import lru_cache

try:
    import spaces  # ZeroGPU: імпорт до torch
except ImportError:
    spaces = None

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(HERE, "run.py")):
    sys.path.insert(0, os.path.dirname(HERE))

import gradio as gr
from huggingface_hub import HfApi, snapshot_download

OWNER = "davidmelash"  # моделі, натреновані в train/; лише їм дозволено trust_remote_code
OWNER_SKIP = {"davidmelash/wechsel_base"}
DEFAULT_NER = "hf:davidmelash/wechsel_base_v7_2026-10-09_r5_s43"
OTHER_PRESETS = [
    "spacy:dchaplinsky/uk_ner_web_trf_13class",
    "gliner:urchade/gliner_multi-v2.1?labels=person,address,phone number&threshold=0.4",
    "hf:dslim/bert-base-NER?aggregation=first&min_score=0.6",
]
MODES = {
    "model_regex": "2. Модель + правила (найкраще)",
    "regex": "1. Лише правила",
    "model": "0. Лише модель",
}
MAX_CHARS = 60_000

import ner_adapters as N  # noqa: E402
import ner_postprocess as NP  # noqa: E402
import run as R  # noqa: E402


def is_owner_model(name: str) -> bool:
    return re.fullmatch(rf"{OWNER}/[\w.-]+", name) is not None


def owner_models() -> list[str]:
    """Token-classification моделі акаунта OWNER (список — на старті Space); без мережі — лише DEFAULT_NER."""
    try:
        ids = sorted(m.id for m in HfApi().list_models(author=OWNER, pipeline_tag="token-classification",
                                                       token=N.hf_token()))
    except Exception as e:
        print(f"список моделей {OWNER}: {type(e).__name__}: {e}")
        ids = []
    specs = [f"hf:{i}" for i in ids if i not in OWNER_SKIP]
    return [DEFAULT_NER] + [s for s in specs if s != DEFAULT_NER]


PRESETS = owner_models() + OTHER_PRESETS


def resolve_spec(spec: str) -> str:
    """spaCy-модель з HF (spacy:org/repo) завантажується як тека (без .whl на 1,3 ГБ) і вантажиться за шляхом.
    Моделям OWNER вмикається trust_remote_code (токенізатор LiBERTa), будь-якій іншій він заборонений."""
    kind, name, opts = N.parse_spec(spec)
    if opts.get("trust_remote_code") and not is_owner_model(name):
        raise ValueError(f"trust_remote_code дозволено лише для моделей {OWNER}/…")
    if kind == "hf" and is_owner_model(name) and "trust_remote_code" not in opts:
        spec = N.make_spec(kind, name, **opts, trust_remote_code=1)
    if kind == "spacy" and not os.path.exists(name) and re.fullmatch(r"[\w.-]+/[\w.-]+", name):
        path = snapshot_download(name, ignore_patterns=["*.whl"], token=N.hf_token())
        spec = spec.replace(name, path, 1)
    return spec


@lru_cache(maxsize=2)
def load_adapter(spec: str) -> N.NerAdapter:
    return N.load_adapter(resolve_spec(spec))


if spaces is not None:
    @spaces.GPU(duration=1)
    def _gpu_noop():
        """ZeroGPU вимагає хоча б одну GPU-функцію; інференс іде на CPU, квоту не витрачаємо."""


def pseudonymize(text: str, mode: str = "model_regex", ner: str = DEFAULT_NER, labels: str = ""):
    """Псевдонімізує текст судового рішення.

    Args:
        text: текст документа українською.
        mode: model_regex | regex | model.
        ner: NER-модель: spacy:<repo або шлях> | hf:<repo>[?опції] | gliner:<repo>[?опції].
        labels: відповідність міток моделі типам через пробіл, напр. "LOC=АДРЕСА @pii"; "МІТКА=" вимикає мітку.

    Returns:
        розмітка для підсвічування, псевдонімізований текст, анотації [{start, end, pseudonym}], сутності моделі.
    """
    text = (text or "").strip()
    if not text:
        raise gr.Error("Введіть текст.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Текст задовгий: {len(text)} символів, максимум {MAX_CHARS}.")
    ner = (ner or DEFAULT_NER).strip()
    try:
        label_map = R.parse_label_map(labels.split())
    except SystemExit as e:  # parse_label_map — для CLI
        raise gr.Error(str(e))
    raw = []
    if mode != "regex":
        try:
            adapter = load_adapter(ner)
        except Exception as e:
            raise gr.Error(f"Не вдалося завантажити модель {ner}: {type(e).__name__}: {e}")
        raw = adapter.predict([text])[0]
    spans = NP.process(N.map_labels(raw, label_map), text)
    ann = R.annotate(text, spans, mode)

    highlighted, out, pos = [], [], 0
    for a in ann:
        highlighted += [(text[pos:a["start"]], None), (text[a["start"]:a["end"]], a["pseudonym"].rsplit("_", 1)[0])]
        out += [text[pos:a["start"]], f"[{a['pseudonym']}]"]
        pos = a["end"]
    highlighted.append((text[pos:], None))
    out.append(text[pos:])
    model_rows = [[x.text, x.label, (label_map.get(x.label.upper()) or "—"), round(x.score, 3)] for x in raw]
    ann_json = [{**a, "text": text[a["start"]:a["end"]]} for a in ann]
    return highlighted, "".join(out), ann_json, model_rows


EXAMPLES = [
    "ІМЕНЕМ УКРАЇНИ\n\nСуддя Шевченківського районного суду м. Києва, розглянувши матеріали за позовом "
    "Коваленко Марії Іванівни до Петренка Олега Васильовича про стягнення аліментів, встановив:\n\n"
    "Позивачка Коваленко М. І., 12.03.1985 року народження, зареєстрована за адресою: м. Київ, вул. Січових "
    "Стрільців, буд. 21, кв. 14, РНОКПП 3112408765, тел. +380671234567, звернулася до суду з позовом. "
    "Відповідач Петренко О. В. проживає за адресою: Київська область, м. Бровари, вул. Героїв Небесної Сотні, 5. "
    "Аліменти перераховувати на рахунок IBAN UA213223130000026007233566001.",
    "Обвинувачений Сидоренко Андрій Петрович, 04.11.1990 року народження, паспорт серії КВ 123456, виданий Подільським РВ "
    "ГУ ДМС України в м. Києві 15.05.2005, керуючи автомобілем «Toyota Camry», номерний знак АА 1234 ВС, "
    "VIN JTNBE46K073012345, скоїв наїзд на пішохода Мельник Ольгу Степанівну. Потерпіла Мельник О. С. "
    "проживає в с. Гора Бориспільського району, вул. Центральна, 7. Електронна пошта: olga.melnyk@example.com.",
]

with gr.Blocks(title="Псевдонімізація судових рішень") as demo:
    gr.Markdown(
        "## Псевдонімізація українських судових рішень\n"
        "NER-модель + правила (regex) для ОСОБА, НОМЕР / ІНФОРМАЦІЯ / АДРЕСА. "
        f"Моделі `{OWNER}/…` натреновані нами і знають усі чотири типи. "
        "Модель можна замінити на будь-яку з Hugging Face: `hf:` (token-classification), `gliner:` або `spacy:`. "
        "Код: [github.com/EnderLancer/anonimimizer](https://github.com/EnderLancer/anonimimizer)."
    )
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="Текст рішення", lines=14, max_lines=40)
            with gr.Row():
                mode = gr.Radio([(v, k) for k, v in MODES.items()], value="model_regex", label="Режим")
            ner = gr.Dropdown(PRESETS, value=DEFAULT_NER, allow_custom_value=True, label="NER-модель",
                              info="тип:repo[?опції], напр. hf:org/model?aggregation=first&min_score=0.6. "
                                   "Перше завантаження нової моделі триває до кількох хвилин.")
            labels = gr.Textbox(label="Мітки моделі -> типи", placeholder="LOC=АДРЕСА ORG= @pii",
                                info="МІТКА=ТИП через пробіл (ОСОБА, АДРЕСА, НОМЕР, ІНФОРМАЦІЯ); МІТКА= вимикає; "
                                     "@pii — набір міток PII-моделей. PER/PERS/PERSON → ОСОБА за замовчуванням.")
            btn = gr.Button("Псевдонімізувати", variant="primary")
        with gr.Column(scale=4):
            highlighted = gr.HighlightedText(label="Знайдені сутності", combine_adjacent=False,
                                             color_map={"ОСОБА": "red", "АДРЕСА": "green", "НОМЕР": "blue",
                                                        "ІНФОРМАЦІЯ": "orange"})
            result = gr.Textbox(label="Псевдонімізований текст", lines=10, max_lines=40)
    with gr.Accordion("Анотації та сирий вихід моделі", open=False):
        ann = gr.JSON(label="Анотації")
        model_rows = gr.Dataframe(headers=["фрагмент", "мітка моделі", "тип", "score"], label="Сутності моделі")
    gr.Examples([[e] for e in EXAMPLES], inputs=[text])

    btn.click(pseudonymize, [text, mode, ner, labels], [highlighted, result, ann, model_rows], api_name="pseudonymize")

load_adapter(DEFAULT_NER)  # модель за замовчуванням — під час старту, а не на першому запиті

if __name__ == "__main__":
    demo.launch(mcp_server=importlib.util.find_spec("mcp") is not None)
