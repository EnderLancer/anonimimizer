"""Нейромережа для імен: spaCy-модель uk_ner_web_trf_13class (xlm-roberta-large, NER-UK 2.0, ліцензія MIT).

Пошук моделі локально (за порядком):
  1. змінна оточення ANON_SPACY_MODEL (шлях до теки моделі);
  2. solution/models/spacy/  та  solution/models/ ;
  3. тека models/ у корені репозиторію та поруч із ним;
якщо не знайдено — завантаження з Hugging Face (~1,3 ГБ, один раз) у solution/models/spacy/.
"""
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MODEL_NAME = "uk_ner_web_trf_13class"
WHEEL_URL = ("https://huggingface.co/dchaplinsky/uk_ner_web_trf_13class/resolve/main/"
             "uk_ner_web_trf_13class-any-py3-none-any.whl")
SOLUTION_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DIR = SOLUTION_DIR / "models" / "spacy"


def _is_model_dir(p: Path) -> bool:
    return (p / "config.cfg").is_file() and (p / "meta.json").is_file() and (p / "ner").is_dir() and (p / "transformer").is_dir()


def _search(root: Path, depth: int = 4):
    if not root.is_dir():
        return None
    if _is_model_dir(root):
        return root
    if depth == 0:
        return None
    try:
        children = sorted(root.iterdir())
    except OSError:
        return None
    for c in children:
        if c.is_dir() and not c.name.startswith("."):
            found = _search(c, depth - 1)
            if found:
                return found
    return None


def find_model_dir():
    env = os.environ.get("ANON_SPACY_MODEL")
    if env and _is_model_dir(Path(env)):
        return Path(env)
    roots = [DEFAULT_DIR, SOLUTION_DIR / "models", SOLUTION_DIR.parent / "models", SOLUTION_DIR.parent.parent / "models"]
    for r in roots:
        found = _search(r)
        if found:
            return found
    return None


def download_model(dest=DEFAULT_DIR):
    """Завантажує колесо моделі з Hugging Face і розпаковує його (інтернет потрібен лише тут)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Завантаження {MODEL_NAME} (~1,3 ГБ) з Hugging Face…", file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        wheel = Path(tmp) / "model.whl"
        with urllib.request.urlopen(WHEEL_URL) as r, open(wheel, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done, last = 0, 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and done - last > 100 * (1 << 20):
                    print(f"  {done / 1e6:.0f}/{total / 1e6:.0f} MB", file=sys.stderr, flush=True)
                    last = done
        with zipfile.ZipFile(wheel) as z:
            z.extractall(dest)
    found = _search(dest)
    if not found:
        raise RuntimeError(f"Модель розпакована в {dest}, але теку з config.cfg не знайдено")
    return found


def ensure_model_dir(allow_download=True):
    found = find_model_dir()
    if found:
        return found
    if not allow_download:
        raise FileNotFoundError("Модель не знайдена локально. Запустіть: python solution/download_models.py")
    return download_model()


def load_nlp(model_dir=None):
    import spacy
    return spacy.load(str(model_dir or ensure_model_dir()))


def pers_spans(nlp, text):
    """Спани PERS (start, end) у тексті."""
    doc = nlp(text)
    return [(e.start_char, e.end_char) for e in doc.ents if e.label_ == "PERS"]


def pers_spans_chunked(nlp, text, max_chars):
    """Те саме, але довгий текст ділиться на шматки <= max_chars по межах абзаців (економія пам'яті)."""
    if not max_chars or len(text) <= max_chars:
        return pers_spans(nlp, text)
    out, start = [], 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        if end < n:
            cut = text.rfind("\n", start + max_chars // 2, end)
            if cut == -1:
                cut = text.rfind(". ", start + max_chars // 2, end)
            if cut != -1:
                end = cut + 1
        for s, e in pers_spans(nlp, text[start:end]):
            out.append((s + start, e + start))
        start = end
    return out
