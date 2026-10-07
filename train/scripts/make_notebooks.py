#!/usr/bin/env python3
"""Generate the Colab notebooks (train / compare / hypotheses) from code so they stay in sync.

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
!ls -la /content/ai-camp/data/silver/
assert os.path.exists('/content/ai-camp/data/evaluate.py'), 'data/evaluate.py missing in the repo'
print('OK, data from', src)""")

CELL_HF = code("""#@title 3. (optional) Hugging Face login - needed only for --push-to
PUSH_TO = ""  #@param {type:"string"}
if PUSH_TO:
    from google.colab import userdata
    from huggingface_hub import login
    login(userdata.get('HF_TOKEN'))  # add HF_TOKEN in Colab: left sidebar -> Secrets""")

SETUP = [CELL_CODE, CELL_DATA, CELL_HF]

TYPES_LINE = "TYPES = ('ОСОБА','АДРЕСА','НОМЕР','ІНФОРМАЦІЯ')\n"


def train_notebook() -> dict:
    cells = [md("# Pseudonymizer - fine-tune on Colab\n\n"
                "Before running: **Runtime -> Change runtime type -> A100 GPU** (Colab Pro). Enable *Background execution* for long runs.\n\n"
                "Data (`train.parquet`, `dev.parquet`, `dev.jsonl`, optionally `dev_judges.*`) must be in Google Drive at "
                "`MyDrive/ai-camp/` (or `MyDrive/ai-camp/silver/`). Code is cloned from GitHub; every run is copied back to "
                "`MyDrive/ai-camp/runs/<name>/`.\n\nTiming on A100, 24k docs x 3 epochs: base models ~25 min, large ~1-1.5 h.")]
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
extra = (f' --n-docs {N_DOCS}' if N_DOCS else '') + (f' --push-to {PUSH_TO}' if PUSH_TO else '')
# --skip-if-done: if the same model + hyper-parameters + data version is already on Drive, nothing is retrained
!python scripts/train.py --model $MODEL --out runs/$RUN --epochs $EPOCHS --lr $LR --batch $BATCH --grad-accum $GRAD_ACCUM --dev-docs $DEV_DOCS --skip-if-done $DRIVE/runs/$RUN $extra
!rm -rf runs/$RUN/checkpoints && mkdir -p $DRIVE/runs/$RUN && cp -r runs/$RUN/. $DRIVE/runs/$RUN/ && echo saved to $DRIVE/runs/$RUN - model, config.json with data version and fingerprint, train.log"""))
    cells.append(code("""#@title 5. Score on full dev with the official script + copy the run to Drive
!python scripts/predict.py --model runs/$RUN/model --input ../data/silver/dev.jsonl --output runs/$RUN/dev_sub.jsonl --batch 32
!python ../data/evaluate.py ../data/silver/dev.jsonl runs/$RUN/dev_sub.jsonl
import os
if os.path.exists('../data/silver/dev_judges.jsonl'):
    !python scripts/predict.py --model runs/$RUN/model --input ../data/silver/dev_judges.jsonl --output runs/$RUN/dev_judges_sub.jsonl --batch 32
    !python ../data/evaluate.py ../data/silver/dev_judges.jsonl runs/$RUN/dev_judges_sub.jsonl | head -5
!rm -rf runs/$RUN/checkpoints && cp -r runs/$RUN $DRIVE/runs/ && echo copied to $DRIVE/runs/$RUN"""))
    cells.append(code("""#@title 6. (optional) CPU timing estimate - the organizers' limit is 2 h for 240 docs on CPU
!head -n 30 ../data/silver/dev.jsonl > /tmp/dev30.jsonl
import time; t0 = time.time()
!python scripts/predict.py --model runs/$RUN/model --input /tmp/dev30.jsonl --output /tmp/cpu_sub.jsonl --device cpu --threads 4 --batch 8
dt = (time.time() - t0) / 30
print(f'{dt:.2f} s/doc on Colab CPU -> test (240 docs) ~ {dt*240/60:.1f} min  (a laptop CPU is usually 1.5-2x faster)')"""))
    cells.append(REPORT)
    cells.append(code("""#@title 8. FINAL - score on the organizers' test set (run ONCE, for the chosen model only; never tune on this)
CONFIRM = False  #@param {type:"boolean"}
import datetime, subprocess
if CONFIRM:
    !python scripts/predict.py --model runs/$RUN/model --input ../data/test/court_cases.jsonl --output runs/$RUN/test_sub.jsonl --batch 32
    out = subprocess.run(['python', '../data/evaluate.py', '../data/test/court_cases.jsonl', f'runs/{RUN}/test_sub.jsonl'], capture_output=True, text=True).stdout
    print(out)
    with open(f'{DRIVE}/runs/test_runs.log', 'a') as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M} {RUN} {out.splitlines()[0]}\\n")
    !cp runs/$RUN/test_sub.jsonl $DRIVE/runs/$RUN/
else:
    print('set CONFIRM=True to run; this is the final, logged evaluation')"""))
    return cells


LEADERBOARD = code("""#@title Leaderboard (dev, official metric) -> Drive
import json, os
""" + TYPES_LINE + """def best(run):
    p = f'runs/{run}/dev_history.json'
    return max(json.load(open(p)), key=lambda h: h['final_score']) if os.path.exists(p) else None
rows = sorted(((r, best(r)) for r in RUNS), key=lambda x: -(x[1]['final_score'] if x[1] else 0))
lines = ['| run | model | best ep | macro F1 | ' + ' | '.join(t[:3] for t in TYPES) + ' | strict | risk | B3 | CPU s/doc | train min |', '|' + '---|'*12]
for run, h in rows:
    if not h: lines.append(f'| {run} | - | - | no result |'); continue
    m = h['metrics']; per = m['ner_relaxed']['per_type']; cfg = json.load(open(f'runs/{run}/config.json'))
    lines.append(f"| {run} | {cfg['model']} | {int(h['epoch'])} | **{m['final_score']:.4f}** | " + ' | '.join(f"{per[t]['f1']:.3f}" for t in TYPES) +
                 f" | {m['ner_strict']['macro']['f1']:.4f} | {m['deanonymization_risk_by_name']:.3f} | {m['consistency_b3']['f1']:.3f} | {timing.get(run, float('nan')):.2f} | {cfg['train_minutes']:.0f} |")
table = '\\n'.join(lines); print(table)
open(f'{DRIVE}/runs/{LEADERBOARD_NAME}.md', 'w').write(f'# {LEADERBOARD_NAME}\\n\\n' + table + '\\n'); print('\\nsaved to', f'{DRIVE}/runs/{LEADERBOARD_NAME}.md')""")

CPU_TIMING = code("""#@title CPU inference timing (Colab CPU is slower than a laptop; use it for ranking, not absolute numbers)
import json, os, subprocess, time
from tqdm.auto import tqdm
timing = {}
if CPU_TIMING_DOCS:
    !head -n $CPU_TIMING_DOCS ../data/silver/dev.jsonl > /tmp/dev_small.jsonl
    for run in tqdm([r for r in RUNS if os.path.exists(f'runs/{r}/model')], desc='cpu timing'):
        t0 = time.time()
        subprocess.run(['python','scripts/predict.py','--model',f'runs/{run}/model','--input','/tmp/dev_small.jsonl','--output',f'/tmp/{run}_cpu.jsonl','--device','cpu','--threads','4','--batch','4'], capture_output=True)
        timing[run] = (time.time() - t0) / CPU_TIMING_DOCS
        print(f'{run:22s} {timing[run]:5.2f} s/doc -> test (240 docs) ~ {timing[run]*240/60:4.1f} min')
json.dump(timing, open(f'{DRIVE}/runs/cpu_timing_{LEADERBOARD_NAME}.json', 'w'))""")

ERRORS = code("""#@title Per-type error dump for one run (what the model misses / over-predicts on dev)
RUN = "wechsel_base"  #@param {type:"string"}
import json, sys, collections
sys.path.insert(0, '../data'); import evaluate as ev
gold = {str(r['id']): r for r in ev.load_jsonl('../data/silver/dev.jsonl')}
h = best(RUN); pred = {str(r['id']): r for r in ev.load_jsonl(f"runs/{RUN}/dev_submission_epoch{int(h['epoch'])}.jsonl")}
fn, fp = collections.defaultdict(list), collections.defaultdict(list)
for k, g in gold.items():
    o = g['original']; G = ev.spans_from_annotations(g['annotations'], len(o), k); P = ev.spans_from_annotations(pred.get(k, {'annotations': []})['annotations'], len(o), k)
    pairs = ev.match(G, P, False); mg = {i for i, _ in pairs}; mp = {j for _, j in pairs}
    ctx = lambda s: (o[max(0, s.start-40):s.start] + '[[' + o[s.start:s.end] + ']]' + o[s.end:s.end+25]).replace('\\n', ' ')
    for i, s in enumerate(G):
        if i not in mg: fn[s.type].append(ctx(s))
    for j, s in enumerate(P):
        if j not in mp: fp[s.type].append(ctx(s))
for t in TYPES:
    print(f'\\n=== {t}: FN {len(fn[t])}, FP {len(fp[t])}')
    for x in fn[t][:8]: print('  FN', x)
    for x in fp[t][:8]: print('  FP', x)""")


def run_loop(var: str, has_extra: bool) -> dict:
    unpack = "(model, extra, lr, batch, accum, epochs)" if has_extra else "(model, lr, batch, accum, epochs)"
    extra_line = "" if has_extra else "    extra = ''\n"
    return code(f"""#@title Train all (skips a run whose model + hyper-parameters + data were already trained; saves to Drive right after training)
import os, shutil, time, json
items = list({var}.items()); n = len(items); t_all = time.time()
for i, (run, {unpack}) in enumerate(items, 1):
{extra_line}    pct = int(100 * (i - 1) / n)
    print(f'\n[{{i}}/{{n}}  {{pct}}%] ===== {{run}}: {{model}} {{extra}} =====', flush=True)
    nd = f' --n-docs {{N_DOCS}}' if N_DOCS else ''
    t0 = time.time()
    # --skip-if-done: train.py compares the fingerprint with the run on Drive and exits immediately if identical
    !python scripts/train.py --model $model --out runs/$run --epochs $epochs --lr $lr --batch $batch --grad-accum $accum --dev-docs 500 --skip-if-done $DRIVE/runs/$run $extra $nd
    !rm -rf runs/$run/checkpoints && mkdir -p $DRIVE/runs/$run && cp -r runs/$run/. $DRIVE/runs/$run/
    cfg = json.load(open(f'runs/{{run}}/config.json')) if os.path.exists(f'runs/{{run}}/config.json') else {{}}
    print(f'[{{i}}/{{n}}] {{run}}: status={{cfg.get("status")}} best dev F1={{cfg.get("best_dev_f1")}} fingerprint={{cfg.get("fingerprint")}} '
          f'({{(time.time()-t0)/60:.1f}} min; total {{(time.time()-t_all)/60:.0f}} min) -> saved to Drive', flush=True)
print(f'\n[{{n}}/{{n}}  100%] done in {{(time.time()-t_all)/60:.0f}} min')""")


REPORT = code("""#@title Report (Markdown + HTML) -> Drive, shown inline
import time, shutil
from IPython.display import HTML, display
stamp = time.strftime('%Y%m%d_%H%M')
ERRORS_RUN = ""  #@param {type:"string"}   # run to dump dev errors for; empty = best run
err = f' --errors-run {ERRORS_RUN}' if ERRORS_RUN else ''
!python scripts/report.py --runs runs --out runs/report.md $err > /dev/null
for ext in ('md', 'html'):
    shutil.copy(f'runs/report.{ext}', f'{DRIVE}/runs/report_{stamp}.{ext}')
    shutil.copy(f'runs/report.{ext}', f'{DRIVE}/runs/report_latest.{ext}')
print('saved:', f'{DRIVE}/runs/report_{stamp}.md', 'and .html (also report_latest.*)')
display(HTML(open('runs/report.html', encoding='utf-8').read()))""")


def compare_notebook() -> dict:
    cells = [md("# Pseudonymizer - compare base models\n\n"
                "Trains every model in `MODELS` on the same data, scores each epoch on the 500 dev documents with the official "
                "`evaluate.py`, measures CPU inference speed, and writes a leaderboard to `MyDrive/ai-camp/runs/leaderboard.md`.\n\n"
                "Runtime: **A100** (Colab Pro; L4 works too - ~2x slower). Base models ~15-25 min each, large ~1-1.5 h each. Enable *Background execution*.\n\n"
                "Every run saves `model/`, `config.json` (data version + fingerprint), `train.log`, `train_log.json`, dev submissions to "
                "`MyDrive/ai-camp/runs/<run>/` right after training. A run whose fingerprint is already on Drive is skipped.\n\n"
                "Decisions are made on dev only - the organizers' test set is not touched here.")]
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
CPU_TIMING_DOCS = 30  #@param {type:"integer"}  # dev docs for the CPU speed estimate (0 = skip)
print(len(MODELS), 'models')"""))
    cells.append(run_loop("MODELS", has_extra=False))
    cells += [CPU_TIMING, LEADERBOARD, REPORT]
    return cells


def hypotheses_notebook() -> dict:
    cells = [md("# Pseudonymizer - hypothesis tests\n\n"
                "Each hypothesis = one training run with a single change against the baseline, scored on the same 500 dev docs. "
                "Baseline = `BASELINE` (a run already on Drive, e.g. `wechsel_base`).\n\n"
                "| id | hypothesis | change |\n|---|---|---|\n"
                "| H1 | long context helps | `markussagen/xlm-roberta-longformer-base-4096`, window 4096 / stride 512 (full attention over 4096 - quadratic) |\n"
                "| H2 | masks matter | baseline model, `--no-masks` (masked spans -> O) |\n"
                "| H3 | oversampling НОМЕР matters | baseline model, `--oversample-number 0` |\n\n"
                "Runtime: **A100**. H1 ~1 h, H2/H3 ~15 min each.")]
    cells += SETUP
    cells.append(code("""#@title 4. Hypotheses
BASELINE = "wechsel_base"  #@param {type:"string"}   # run name on Drive to compare against
# name: (model, extra CLI args, lr, batch, grad_accum, epochs)
HYPOTHESES = {
    'H1_longformer_4096': ('markussagen/xlm-roberta-longformer-base-4096', '--window 4096 --stride 512 --grad-checkpointing', 3e-5, 4, 8, 3),
    'H2_no_masks':        ('benjamin/roberta-base-wechsel-ukrainian', '--no-masks', 4e-5, 64, 1, 3),
    'H3_no_oversample':   ('benjamin/roberta-base-wechsel-ukrainian', '--oversample-number 0', 4e-5, 64, 1, 3),
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
    cells.append(code("""#@title Results vs baseline
import json, os
""" + TYPES_LINE + """def best(run):
    p = f'runs/{run}/dev_history.json'
    return max(json.load(open(p)), key=lambda h: h['final_score']) if os.path.exists(p) else None
b = best(BASELINE)
print(f"{'run':22s} {'F1':>7} {'delta':>7}  " + '  '.join(f'{t[:3]:>5}' for t in TYPES) + '   strict   risk')
for run in RUNS:
    h = best(run)
    if not h: print(f'{run:22s}  no result'); continue
    m = h['metrics']; per = m['ner_relaxed']['per_type']
    d = m['final_score'] - b['final_score'] if b else float('nan')
    print(f"{run:22s} {m['final_score']:7.4f} {d:+7.4f}  " + '  '.join(f"{per[t]['f1']:5.3f}" for t in TYPES) + f"   {m['ner_strict']['macro']['f1']:.4f}   {m['deanonymization_risk_by_name']:.3f}")"""))
    cells.append(code("""#@title H1 diagnostics - do the baseline's misses sit far from the document start? (if the histogram is flat, long context cannot help)
import sys, collections
sys.path.insert(0, '../data'); import evaluate as ev
gold = {str(r['id']): r for r in ev.load_jsonl('../data/silver/dev.jsonl')}
def fn_positions(run):
    h = best(run); pred = {str(r['id']): r for r in ev.load_jsonl(f"runs/{run}/dev_submission_epoch{int(h['epoch'])}.jsonl")}
    pos = collections.Counter(); total = 0
    for k, g in gold.items():
        o = g['original']; G = ev.spans_from_annotations(g['annotations'], len(o), k); P = ev.spans_from_annotations(pred.get(k, {'annotations': []})['annotations'], len(o), k)
        mg = {i for i, _ in ev.match(G, P, False)}
        for i, s in enumerate(G):
            if i not in mg: pos[min(9, int(10 * s.start / len(o)))] += 1; total += 1
    return total, [pos[i] for i in range(10)]
for run in [r for r in RUNS if best(r)]:
    total, hist = fn_positions(run)
    print(f'{run:22s} FN total {total:4d}  by document decile: {hist}')"""))
    cells += [CPU_TIMING, LEADERBOARD, REPORT]
    return cells


def main() -> None:
    meta = {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}}
    for name, cells in (("colab_train.ipynb", train_notebook()), ("colab_compare_models.ipynb", compare_notebook()),
                        ("colab_hypotheses.ipynb", hypotheses_notebook())):
        nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}
        (HERE / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name}: {len(cells)} cells")


if __name__ == "__main__":
    main()
