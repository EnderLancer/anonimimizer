"""Налаштування запуску ner-models.yml: сценарій + overrides -> outputs для jobs.

Environment: SCENARIO, KIND, MODEL, DEFAULT_MODEL (for KIND=hf when MODEL is empty), NER_OPTIONS (key=value&…),
OVERRIDES (space-separated key=value), GITHUB_OUTPUT, GITHUB_STEP_SUMMARY (optional).
"""
import json
import os
import shlex
import sys
from urllib.parse import parse_qsl

sys.path.insert(0, os.getcwd())
import ner_adapters as N  # noqa: E402

DEFAULTS = {
    "modes": "model,model_regex,full",
    "shards": "8",
    "limit": "",
    "data": "data/test/court_cases.jsonl",
    "cosine": "0",
    "cosine_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "fail_below": "",
    "extra_pip": "",
    "ner_options": "",
}
SCENARIOS = {
    # три режими з моделлю на всьому тесті
    "standard": {},
    # перевірити, що модель завантажується й дає розумний вихід: 20 документів, хвилини
    "quick": {"modes": "model,full", "shards": "2", "limit": "20"},
    # те саме, що standard, плюс базова лінія без моделі (regex)
    "compare": {"modes": "regex,model,model_regex,full"},
    # швидший інференс: int8-квантизація (лише hf) і більше паралельних jobs
    "fast": {"shards": "16", "ner_options": "quantize=1"},
    # усі режими й cosine similarity
    "thorough": {"modes": "regex,model,model_regex,full", "cosine": "1"},
}
MODES = ("regex", "model", "model_regex", "full")


def fail(msg):
    print(f"::error::{msg}")
    sys.exit(1)


def resolve(env):
    scenario = env.get("SCENARIO", "standard")
    if scenario not in SCENARIOS:
        fail(f"невідомий сценарій {scenario!r}; доступні: {', '.join(SCENARIOS)}")
    cfg = {**DEFAULTS, **SCENARIOS[scenario]}
    for item in shlex.split(env.get("OVERRIDES", "")):
        k, sep, v = item.partition("=")
        if not sep or k not in DEFAULTS or k == "ner_options":
            fail(f"overrides: {item!r}; очікується key=value, ключі: {', '.join(k for k in DEFAULTS if k != 'ner_options')}")
        cfg[k] = v

    kind, model = env["KIND"], env["MODEL"].strip()
    if not model and kind == "hf":
        model = env.get("DEFAULT_MODEL", "").strip()
    if not model:
        fail("no model given (for type hf the repository variable NER_MODEL can supply it)")
    allowed = N.adapter_options(kind)
    opts = dict(parse_qsl(cfg["ner_options"]))                     # від сценарію: лише ті, що підходять типу моделі
    skipped = sorted(k for k in opts if k not in allowed)
    opts = {k: v for k, v in opts.items() if k in allowed}
    user = dict(parse_qsl(env.get("NER_OPTIONS", "")))              # від користувача: невідома опція — помилка
    bad = sorted(k for k in user if k not in allowed)
    if bad:
        fail(f"ner_options: тип {kind} не має опцій {bad}; доступні: {', '.join(allowed)}")
    opts.update(user)
    spec = N.make_spec(kind, model, **opts)

    modes = [m.strip() for m in cfg["modes"].split(",") if m.strip()]
    if not modes or any(m not in MODES for m in modes):
        fail(f"modes={cfg['modes']!r}; доступні: {', '.join(MODES)}")
    if not cfg["shards"].isdigit() or not 1 <= int(cfg["shards"]) <= 20:
        fail(f"shards={cfg['shards']!r}; потрібне число від 1 до 20")
    if cfg["limit"] and not cfg["limit"].isdigit():
        fail(f"limit={cfg['limit']!r}; потрібне число")
    if cfg["fail_below"]:
        try:
            float(cfg["fail_below"])
        except ValueError:
            fail(f"fail_below={cfg['fail_below']!r}; потрібне число (відсотки)")
    cosine = cfg["cosine"].lower() in ("1", "true", "yes")
    n = int(cfg["shards"])
    out = {
        "ner": spec,
        "modes": json.dumps(modes),
        "needs_model": str(any(m != "regex" for m in modes)).lower(),
        "shards": json.dumps(list(range(n))),
        "n_shards": str(n),
        "limit": cfg["limit"],
        "data": cfg["data"],
        "cosine": str(cosine).lower(),
        "cosine_model": cfg["cosine_model"],
        "fail_below": cfg["fail_below"],
        "extra_pip": cfg["extra_pip"],
        "weights_key": f"weights-{kind}-{model}-{opts.get('revision', 'default')}",
    }
    return scenario, out, skipped


def main():
    scenario, out, skipped = resolve(os.environ)
    for k in skipped:
        print(f"::notice::сценарій {scenario}: опція {k} не застосовується до {os.environ['KIND']}, пропущено")
    lines = "".join(f"{k}={v}\n" for k, v in out.items())
    print(lines, end="")
    if os.environ.get("GITHUB_OUTPUT"):
        open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf8").write(lines)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        rows = [("сценарій", scenario)] + [(k, v) for k, v in out.items() if k not in ("shards", "weights_key", "needs_model")]
        open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf8").write(
            "### Налаштування запуску\n\n| | |\n|---|---|\n" + "".join(f"| {k} | `{v}` |\n" for k, v in rows if v != ""))


if __name__ == "__main__":
    main()
