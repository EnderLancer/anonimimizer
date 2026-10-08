#!/usr/bin/env python3
"""Generate the Colab notebooks (train / compare / hypotheses / nym-pii) from code so they stay in sync.

    uv run scripts/make_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


ENV = ("import os; os.environ.update(TRANSFORMERS_VERBOSITY='error', PYTHONWARNINGS='ignore', "
       "TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1')\n")

CELL_CODE = code("""#@title 1. Code
REPO = "EnderLancer/anonimimizer"  #@param {type:"string"}
BRANCH = "dmelashych/datagen"  #@param {type:"string"}
import os
# private repo: add a GitHub token as Colab secret GH_TOKEN (left sidebar -> key icon -> enable notebook access)
try:
    from google.colab import userdata
    token = userdata.get('GH_TOKEN')
except Exception:
    token = None
url = f"https://{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"
if not os.path.exists('/content/ai-camp/train'):
    !rm -rf /content/ai-camp && git clone -q --branch "$BRANCH" "$url" /content/ai-camp
else:
    !cd /content/ai-camp && git pull -q
assert os.path.exists('/content/ai-camp/train'), 'clone failed: check REPO/BRANCH and the GH_TOKEN secret'
%cd /content/ai-camp/train
!pip install -q -e . 2>&1 | tail -1
""" + ENV + """import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'
print('GPU:', gpu)
if 'A100' not in gpu and 'L4' not in gpu:
    print('WARNING: not an A100 - Runtime -> Change runtime type -> A100 GPU (Colab Pro); on T4 halve every batch size')""")

CELL_DATA = code("""#@title 2. Data from Google Drive
from google.colab import drive
drive.mount('/content/drive')
import glob, os, shutil
from tqdm.auto import tqdm
DRIVE = '/content/drive/MyDrive/ai-camp'
os.makedirs('/content/ai-camp/data/silver', exist_ok=True); os.makedirs(f'{DRIVE}/runs', exist_ok=True)
src = f'{DRIVE}/silver' if os.path.exists(f'{DRIVE}/silver/train.parquet') else DRIVE
files = glob.glob(f'{src}/*.parquet') + glob.glob(f'{src}/*.jsonl')
assert files, f'no parquet/jsonl files found in {DRIVE} or {DRIVE}/silver'
for f in tqdm(files, desc='copy from Drive'):
    shutil.copy(f, '/content/ai-camp/data/silver/')
for need in ('train.parquet', 'dev.parquet', 'dev.jsonl'):
    assert os.path.exists(f'/content/ai-camp/data/silver/{need}'), f'{need} is missing in {src} - upload it to Drive first'
!ls -la /content/ai-camp/data/silver/
assert os.path.exists('/content/ai-camp/data/evaluate.py'), 'data/evaluate.py missing in the repo'
print('OK, data from', src)""")

CELL_HF = code("""#@title 3. Hugging Face - every trained model is pushed to a PRIVATE repo <HF_USER>/pseudo-<run>
HF_USER = ""  #@param {type:"string"}  # your HF username or org; empty = do not push
if HF_USER:
    import os
    from google.colab import userdata
    from huggingface_hub import login
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')  # Colab: left sidebar -> Secrets -> HF_TOKEN (write access), enable notebook access
    login(os.environ['HF_TOKEN'])
else:
    print('HF_USER is empty: models are NOT pushed to the Hub')""")

SETUP = [CELL_CODE, CELL_DATA, CELL_HF]


def train_notebook() -> dict:
    cells = [md("# Pseudonymizer - fine-tune on Colab\n\n"
                "Before running: **Runtime -> Change runtime type -> A100 GPU** (Colab Pro). Enable *Background execution* for long runs.\n\n"
                "Data (`train.parquet`, `dev.parquet`, `dev.jsonl`, optionally `dev_judges.*`) must be in Google Drive at "
                "`MyDrive/ai-camp/` (or `MyDrive/ai-camp/silver/`). Code is cloned from GitHub; every run is copied back to "
                "`MyDrive/ai-camp/runs/<name>/`.\n\nThe last cell scores the run on the organizers' test set and shows the test "
                "leaderboard over every run on Drive with the history of runs (`MyDrive/ai-camp/runs/leaderboard_all.md`).\n\n"
                "Timing on A100, 24k docs x 3 epochs: base models ~25 min, large ~1-1.5 h.")]
    cells += SETUP
    cells.append(code("""#@title 4. Train one model
MODEL = "benjamin/roberta-base-wechsel-ukrainian"  #@param ["benjamin/roberta-base-wechsel-ukrainian", "FacebookAI/xlm-roberta-base", "microsoft/mdeberta-v3-base", "benjamin/roberta-large-wechsel-ukrainian", "FacebookAI/xlm-roberta-large", "Goader/liberta-large-v2"] {allow-input: true}
RUN = "wechsel_base_full"  #@param {type:"string"}
EPOCHS = 3  #@param {type:"number"}
LR = 4e-5  #@param {type:"number"}  # base: 4e-5 at batch 64; large: 2e-5
BATCH = 64  #@param {type:"integer"}  # A100 40GB: base 64, large 32; T4 16GB: base 16, large 8 + GRAD_ACCUM 2
GRAD_ACCUM = 1  #@param {type:"integer"}
N_DOCS = 0  #@param {type:"integer"}
DEV_DOCS = 500  #@param {type:"integer"}
extra = (f' --n-docs {N_DOCS}' if N_DOCS else '') + (f' --push-to {HF_USER}/pseudo-{RUN}' if HF_USER else '')
# --skip-if-done: if the same model + hyper-parameters + data version is already on Drive, nothing is retrained
!python scripts/train.py --model $MODEL --out runs/$RUN --epochs $EPOCHS --lr $LR --batch $BATCH --grad-accum $GRAD_ACCUM --dev-docs $DEV_DOCS --skip-if-done $DRIVE/runs/$RUN $extra
!rm -rf runs/$RUN/checkpoints && mkdir -p $DRIVE/runs/$RUN && cp -r runs/$RUN/. $DRIVE/runs/$RUN/ && echo saved to $DRIVE/runs/$RUN - model, config.json with data version and fingerprint, train.log"""))
    cells.append(code("""#@title 5. (optional) CPU timing estimate - the organizers' limit is 2 h for 240 docs on CPU
!head -n 30 ../data/test/court_cases.jsonl > /tmp/test30.jsonl
import time; t0 = time.time()
!python scripts/predict.py --model runs/$RUN/model --input /tmp/test30.jsonl --output /tmp/cpu_sub.jsonl --device cpu --threads 4 --batch 8
dt = (time.time() - t0) / 30
import json, os
p = f'{DRIVE}/runs/cpu_timing_train.json'; timing = json.load(open(p)) if os.path.exists(p) else {}
timing[RUN] = dt; json.dump(timing, open(p, 'w'))  # the leaderboard reads runs/cpu_timing_*.json
print(f'{dt:.2f} s/doc on Colab CPU -> test (240 docs) ~ {dt*240/60:.1f} min  (a laptop CPU is usually 1.5-2x faster)')"""))
    cells.append(leaderboard("6. Leaderboard on the organizers' test set (this run + every run on Drive) + history of runs -> Drive",
                             "--name leaderboard_all --all --runs $RUN", "leaderboard_all"))
    return cells


def leaderboard(title: str = "Leaderboard on the organizers' test set + history of runs -> Drive",
                args: str = "--name $LEADERBOARD_NAME --runs $runs $baseline", name: str = "{LEADERBOARD_NAME}") -> dict:
    """The only results cell. scripts/leaderboard.py scores the runs on the test set (once per trained model), appends new
    evaluations to runs/test_history.jsonl and rewrites the Markdown file: leaderboard, progress, runs over time, history."""
    setup = ("runs = ','.join(dict.fromkeys(RUNS))\n" if "$runs" in args else "") \
        + ("baseline = f'--baseline {BASELINE}' if 'BASELINE' in globals() else ''\n" if "$baseline" in args else "")
    return code(f"""#@title {title}
from IPython.display import Markdown, display
{setup}!python scripts/leaderboard.py --drive $DRIVE {args}
display(Markdown(open(f'{{DRIVE}}/runs/{name}.md', encoding='utf-8').read()))""")


CPU_TIMING = code("""#@title CPU inference timing (Colab CPU is slower than a laptop; use it for ranking, not absolute numbers)
import json, os, subprocess, time
from tqdm.auto import tqdm
timing = {}
if CPU_TIMING_DOCS:
    !head -n $CPU_TIMING_DOCS ../data/test/court_cases.jsonl > /tmp/test_small.jsonl
    for run in tqdm([r for r in RUNS if os.path.exists(f'runs/{r}/model')], desc='cpu timing'):
        t0 = time.time()
        subprocess.run(['python','scripts/predict.py','--model',f'runs/{run}/model','--input','/tmp/test_small.jsonl','--output',f'/tmp/{run}_cpu.jsonl','--device','cpu','--threads','4','--batch','4'], capture_output=True)
        timing[run] = (time.time() - t0) / CPU_TIMING_DOCS
        print(f'{run:22s} {timing[run]:5.2f} s/doc -> test (240 docs) ~ {timing[run]*240/60:4.1f} min')
json.dump(timing, open(f'{DRIVE}/runs/cpu_timing_{LEADERBOARD_NAME}.json', 'w'))""")


def run_loop(var: str, has_extra: bool) -> dict:
    unpack = "(model, extra, lr, batch, accum, epochs)" if has_extra else "(model, lr, batch, accum, epochs)"
    extra_line = "" if has_extra else "    extra = ''\n"
    return code(f"""#@title Train all (skips a run whose model + hyper-parameters + data were already trained; saves to Drive right after training)
import os, shutil, time, json
items = list({var}.items()); n = len(items); t_all = time.time()
for i, (run, {unpack}) in enumerate(items, 1):
{extra_line}    pct = int(100 * (i - 1) / n)
    print(f'\\n[{{i}}/{{n}}  {{pct}}%] ===== {{run}}: {{model}} {{extra}} =====', flush=True)
    nd = f' --n-docs {{N_DOCS}}' if N_DOCS else ''
    nd += f' --push-to {{HF_USER}}/pseudo-{{run}}' if HF_USER else ''
    t0 = time.time()
    # --skip-if-done: train.py compares the fingerprint with the run on Drive and exits immediately if identical
    rc = os.system(f'python scripts/train.py --model {{model}} --out runs/{{run}} --epochs {{epochs}} --lr {{lr}} --batch {{batch}} '
                   f'--grad-accum {{accum}} --dev-docs 500 --skip-if-done {{DRIVE}}/runs/{{run}} {{extra}}{{nd}}')
    cfg = json.load(open(f'runs/{{run}}/config.json')) if os.path.exists(f'runs/{{run}}/config.json') else {{}}
    if rc != 0 or cfg.get('status') != 'done':
        print(f'\\n[{{i}}/{{n}}] {{run}} FAILED (exit {{rc}}, status {{cfg.get("status")}}) - see runs/{{run}}/train.log; stopping the loop', flush=True)
        break
    !rm -rf runs/$run/checkpoints && mkdir -p $DRIVE/runs/$run && cp -r runs/$run/. $DRIVE/runs/$run/
    print(f'[{{i}}/{{n}}] {{run}}: fingerprint={{cfg.get("fingerprint")}} '
          f'({{(time.time()-t0)/60:.1f}} min; total {{(time.time()-t_all)/60:.0f}} min) -> saved to Drive', flush=True)
print(f'\\n[{{n}}/{{n}}  100%] done in {{(time.time()-t_all)/60:.0f}} min')""")


def compare_notebook() -> dict:
    cells = [md("# Pseudonymizer - compare base models\n\n"
                "Trains every model in `MODELS` on the same data, measures CPU inference speed, scores every model on the organizers' "
                "test set with the official `evaluate.py` and writes the test leaderboard with the history of runs to "
                "`MyDrive/ai-camp/runs/leaderboard.md`.\n\n"
                "Runtime: **A100** (Colab Pro; L4 works too - ~2x slower). Base models ~15-25 min each, large ~1-1.5 h each. Enable *Background execution*.\n\n"
                "Every run saves `model/`, `config.json` (data version + fingerprint), `train.log`, `train_log.json` to "
                "`MyDrive/ai-camp/runs/<run>/` right after training. A run whose fingerprint is already on Drive is skipped.\n\n"
                "A run is predicted on the test set once: its `test_result.json` on Drive is reused until the run is retrained. "
                "Every new evaluation is appended to `runs/test_history.jsonl`, the source of the history tables.")]
    cells += SETUP
    cells.append(code("""#@title 4. Models to compare
# name: (hf id, lr, batch, grad_accum, epochs)
# A100 40 GB: base models use ~9 GB at batch 32 -> batch 64 is safe; large ~ batch 32. T4 16 GB: halve batches, grad_accum 2.
MODELS = {
    'wechsel_base':   ('benjamin/roberta-base-wechsel-ukrainian',  4e-5, 64, 1, 3),
    'xlmr_base':      ('FacebookAI/xlm-roberta-base',               4e-5, 64, 1, 3),
    'mdeberta_base':  ('microsoft/mdeberta-v3-base',                4e-5, 32, 2, 3),
    'wechsel_large':  ('benjamin/roberta-large-wechsel-ukrainian', 2e-5, 32, 1, 3),
    'xlmr_large':     ('FacebookAI/xlm-roberta-large',              2e-5, 32, 1, 3),
    'liberta_large':  ('Goader/liberta-large-v2',                   2e-5, 32, 1, 3),
}
RUNS = list(MODELS); LEADERBOARD_NAME = 'leaderboard'
SKIP_DONE = True  #@param {type:"boolean"}   # skip models that already have a run on Drive
N_DOCS = 0  #@param {type:"integer"}         # 0 = all; e.g. 5000 for a quick screening pass
CPU_TIMING_DOCS = 30  #@param {type:"integer"}  # test docs for the CPU speed estimate (0 = skip)
print(len(MODELS), 'models')"""))
    cells.append(run_loop("MODELS", has_extra=False))
    cells += [CPU_TIMING, leaderboard()]
    return cells


def hypotheses_notebook() -> dict:
    cells = [md("# Pseudonymizer - hypothesis tests\n\n"
                "Each hypothesis = one training run with a single change against the baseline, scored on the organizers' test set. "
                "Baseline = `BASELINE` (a run already on Drive, e.g. `wechsel_base`); the leaderboard (`runs/hypotheses.md`, with the "
                "history of runs) shows the difference to it.\n\n"
                "| id | hypothesis | change |\n|---|---|---|\n"
                "| H1 | long context helps | `markussagen/xlm-roberta-longformer-base-4096`, window 4096 / stride 512 (full attention over 4096 - quadratic) |\n"
                "| H2 | masks matter | baseline model, `--no-masks` (masked spans -> O) |\n"
                "| H3 | oversampling НОМЕР matters | baseline model, `--oversample-number 0` |\n"
                "| H4 | PII pre-training helps | `Wismut/nym-pii-multilingual` (mmBERT/ModernBERT, 40 PII types), window 1024 / stride 256, head replaced |\n\n"
                "Runtime: **A100**. H1 ~1 h, H2/H3 ~15 min each.")]
    cells += SETUP
    cells.append(code("""#@title 4. Hypotheses
BASELINE = "wechsel_base"  #@param {type:"string"}   # run name on Drive to compare against
# name: (model, extra CLI args, lr, batch, grad_accum, epochs)
HYPOTHESES = {
    'H1_longformer_4096': ('markussagen/xlm-roberta-longformer-base-4096', '--window 4096 --stride 512 --grad-checkpointing', 3e-5, 4, 8, 3),
    'H2_no_masks':        ('benjamin/roberta-base-wechsel-ukrainian', '--no-masks', 4e-5, 64, 1, 3),
    'H3_no_oversample':   ('benjamin/roberta-base-wechsel-ukrainian', '--oversample-number 0', 4e-5, 64, 1, 3),
    # PII-pretrained mmBERT (ModernBERT, 8k context, 40 PII types, 23 langs incl. uk): does PII pre-exposure beat a stronger Ukrainian encoder?
    'H4_nym_pii_mmbert':  ('Wismut/nym-pii-multilingual', '--window 1024 --stride 256', 3e-5, 32, 1, 3),
}
RUN_ONLY = ""  #@param {type:"string"}   # e.g. H1_longformer_4096 to run a single hypothesis
SKIP_DONE = True  #@param {type:"boolean"}
N_DOCS = 0  #@param {type:"integer"}
CPU_TIMING_DOCS = 20  #@param {type:"integer"}
TODO = {k: v for k, v in HYPOTHESES.items() if not RUN_ONLY or k == RUN_ONLY}
RUNS = [BASELINE] + list(HYPOTHESES); LEADERBOARD_NAME = 'hypotheses'
import os, shutil
if os.path.exists(f'{DRIVE}/runs/{BASELINE}') and not os.path.exists(f'runs/{BASELINE}'):
    shutil.copytree(f'{DRIVE}/runs/{BASELINE}', f'runs/{BASELINE}')
print(list(TODO))"""))
    cells.append(run_loop("TODO", has_extra=True))
    cells += [CPU_TIMING, leaderboard()]
    return cells


def nym_pii_notebook() -> dict:
    cells = [md("# Pseudonymizer - `Wismut/nym-pii-multilingual`\n\n"
                "Fine-tunes the PII-pretrained mmBERT (ModernBERT-base, MIT, 23 languages incl. Ukrainian, 40 PII types) on the silver data "
                "and compares it with `BASELINE` (a run already on Drive) on the organizers' test set. The 40-type head is dropped and "
                "replaced by our 4 types; only the encoder is reused.\n\n"
                "| run | question | change |\n|---|---|---|\n"
                "| `H4_nym_pii_mmbert` | does PII pre-training + long context beat the Ukrainian encoder? | window 1024 / stride 256 |\n"
                "| `nym_pii_w512` | how much of that is the longer window? | window 512 / stride 128, same as the baseline |\n\n"
                "`H4_nym_pii_mmbert` is the same run as H4 in `colab_hypotheses.ipynb`: whichever notebook trains it first, the other skips it "
                "by fingerprint. Do not train it in both at the same time.\n\n"
                "Runtime: **A100** (L4 works, ~2x slower). Each run saves to `MyDrive/ai-camp/runs/<run>/` right after training. "
                "The test leaderboard with the history of runs is written to `MyDrive/ai-camp/runs/nym_pii.md`.")]
    cells += SETUP
    cells.append(code("""#@title 4. Runs
BASELINE = "wechsel_base"  #@param {type:"string"}   # run name on Drive to compare against
MODEL = "Wismut/nym-pii-multilingual"  #@param {type:"string"}
# name: (model, extra CLI args, lr, batch, grad_accum, epochs)
NYM_RUNS = {
    'H4_nym_pii_mmbert': (MODEL, '--window 1024 --stride 256', 3e-5, 32, 1, 3),
    'nym_pii_w512':      (MODEL, '', 3e-5, 64, 1, 3),
}
RUN_ONLY = ""  #@param {type:"string"}   # e.g. nym_pii_w512 to run a single variant
N_DOCS = 0  #@param {type:"integer"}     # 0 = all; e.g. 2000 for a smoke run
CPU_TIMING_DOCS = 20  #@param {type:"integer"}
TODO = {k: v for k, v in NYM_RUNS.items() if not RUN_ONLY or k == RUN_ONLY}
RUNS = [BASELINE] + list(NYM_RUNS); LEADERBOARD_NAME = 'nym_pii'
import os, shutil
if os.path.exists(f'{DRIVE}/runs/{BASELINE}') and not os.path.exists(f'runs/{BASELINE}'):
    shutil.copytree(f'{DRIVE}/runs/{BASELINE}', f'runs/{BASELINE}')
print(list(TODO))"""))
    cells.append(run_loop("TODO", has_extra=True))
    cells += [CPU_TIMING, leaderboard()]
    return cells


def check_syntax(name: str, cells: list[dict]) -> None:
    """Every code cell must parse as Python once IPython magics / shell lines are replaced by `pass`."""
    import ast
    import re

    for c in cells:
        if c["cell_type"] != "code":
            continue
        lines = []
        for l in c["source"].splitlines():
            m = re.match(r"(\s*)[!%]", l)
            lines.append(m.group(1) + "pass" if m else l)
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as e:
            raise SystemExit(f"{name}: syntax error in cell '{c['source'].splitlines()[0][:60]}': {e}")


def main() -> None:
    meta = {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}}
    for name, cells in (("colab_train.ipynb", train_notebook()), ("colab_compare_models.ipynb", compare_notebook()),
                        ("colab_hypotheses.ipynb", hypotheses_notebook()), ("colab_nym_pii.ipynb", nym_pii_notebook())):
        check_syntax(name, cells)
        nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}
        (HERE / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name}: {len(cells)} cells")


if __name__ == "__main__":
    main()
