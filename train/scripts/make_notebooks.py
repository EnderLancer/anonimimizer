#!/usr/bin/env python3
"""Generate the notebooks from code into train/notebooks/: colab_train_base.ipynb trains a fixed list of
models (MODELS; TOP_N adds the best of the past runs on Drive) on `silver`, colab_train_large.ipynb on `silver_large`.
vast_train_base.ipynb and vast_train_large.ipynb do the same on a rented machine with several GPUs, through scripts/launch.py.

    uv run scripts/make_notebooks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "notebooks"


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

CELL_DATA = code("""#@title 2. Google Drive: data folders and runs
from google.colab import drive
drive.mount('/content/drive')
import glob, os, shutil
from tqdm.auto import tqdm
DRIVE = '/content/drive/MyDrive/ai-camp'
LOCAL = '/content/ai-camp/data/silver'
os.makedirs(LOCAL, exist_ok=True); os.makedirs(f'{DRIVE}/runs/launches', exist_ok=True)

def use_data(folder):
    \"\"\"Replace the training data in the repo with the files of MyDrive/ai-camp/<folder>.\"\"\"
    src = f'{DRIVE}/{folder}'
    if folder == 'silver' and not os.path.exists(f'{src}/train.parquet'):
        src = DRIVE  # older layout: the files lie in MyDrive/ai-camp itself
    files = glob.glob(f'{src}/*.parquet') + glob.glob(f'{src}/*.jsonl')
    for need in ('train.parquet', 'dev.parquet', 'dev.jsonl'):
        assert os.path.exists(f'{src}/{need}'), f'{need} is missing in {src} - upload it to Drive first'
    for old in glob.glob(f'{LOCAL}/*.parquet') + glob.glob(f'{LOCAL}/*.jsonl'):
        os.remove(old)
    for f in tqdm(files, desc=f'copy from {src}'):
        shutil.copy(f, LOCAL)
    print('OK, data from', src)

def jobs(models, datasets, seeds, launch, n_docs=0):
    \"\"\"One training job per model x dataset x seed. Run name: <model>_<tag>_<launch>[_s<seed>], e.g.
    wechsel_base_v3_2026-10-08_r1_s43 (seed 42 has no suffix) - a new launch never reuses the folder of an older one.\"\"\"
    return [dict(run=f"{m['name']}_{tag}_{launch}" + (f'_s{s}' if s != 42 else ''), name=m['name'], model=m['model'], extra=m['extra'],
                 lr=m['lr'], batch=m['batch'], accum=m['accum'], epochs=m['epochs'], seed=s, data=folder, n_docs=n_docs)
            for tag, folder in datasets.items() for m in models for s in seeds]

assert os.path.exists('/content/ai-camp/data/evaluate.py'), 'data/evaluate.py missing in the repo'""")

SETUP = [CELL_CODE, CELL_DATA]


def cell_plan(tag: str, folder: str, seeds: str) -> dict:
    return code("""#@title 3. Models and the plan of this launch
TOP_N = 0  #@param {type:"integer"}        # besides MODELS, add this many of the best models of the past runs on Drive (0 = none)
LAUNCH = ""  #@param {type:"string"}       # empty = a new launch (today's date + next number); "2026-10-08_r1" continues that launch
SEEDS = "__SEEDS__"  #@param {type:"string"}   # comma-separated; every model is trained once per seed and the leaderboard averages them
N_DOCS = 0  #@param {type:"integer"}       # 0 = all; e.g. 2000 for a smoke run
CPU_TIMING_DOCS = 30  #@param {type:"integer"}  # test docs for the CPU speed estimate (0 = skip)
# this notebook trains on one dataset. tag: folder in MyDrive/ai-camp with train.parquet, dev.parquet, dev.jsonl
TAG, FOLDER = '__TAG__', '__FOLDER__'
DATASETS = {TAG: FOLDER}
# the models of every launch, chosen on 2026-10-09 (test macro F1 of 2026-10-08: wechsel_base 0.8530).
# Settings are the ones the past runs used.
MODELS = [
    dict(name='wechsel_base',         model='benjamin/roberta-base-wechsel-ukrainian', extra='', lr=4e-5, batch=64, accum=1, epochs=3),
    # windows of 2048 tokens (the model reads up to 8192): 8 of them hold as many tokens as 32 windows of 512. No result yet;
    # if the GPU runs out of memory, add --grad-checkpointing to extra
    dict(name='modern_liberta_large_w2048', model='Goader/modern-liberta-large', extra='--window 2048 --stride 256', lr=3e-5, batch=8, accum=1, epochs=3),
]
import datetime, json, re, sys, zoneinfo
from pathlib import Path
sys.path.insert(0, 'src')
from pseudo.data import DECODE_VERSION
from pseudo.leaderboard import load_history, test_version, top_models

launches = f'{DRIVE}/runs/launches'
plan_file = lambda launch: f'{launches}/{launch}_{TAG}.json'  # the other notebook keeps its own plan of the same launch name
if LAUNCH:
    assert os.path.exists(plan_file(LAUNCH)), f'no launch {LAUNCH} of {TAG} in {launches}: ' + ', '.join(sorted(os.listdir(launches)))
    plan = json.load(open(plan_file(LAUNCH)))  # the models of a launch stay the ones chosen when it started
    print(f'continuing launch {LAUNCH} (started {plan["created"]}); MODELS, TOP_N, SEEDS and N_DOCS of this cell are ignored')
else:
    assert os.path.exists(f'{DRIVE}/{FOLDER}/train.parquet'), f'no train.parquet in {DRIVE}/{FOLDER} - upload the dataset to Drive first'
    top = [dict(m) for m in MODELS]
    if TOP_N:
        # every run on Drive is scored on the current test file with the current decoder (once per trained model),
        # then the models are ranked by those scores only
        !python scripts/leaderboard.py --drive $DRIVE --name leaderboard --all
        best = top_models(load_history(Path(f'{DRIVE}/runs')), TOP_N + len(MODELS), test_version('../data/test/court_cases.jsonl'), DECODE_VERSION)
        top += [m for m in best if m['model'] not in {t['model'] for t in top}][:TOP_N]
    now = datetime.datetime.now(zoneinfo.ZoneInfo('Europe/Kyiv'))
    date = now.strftime('%Y-%m-%d')
    # two notebooks started at the same moment may take the same number: their runs and plans still differ by the tag
    taken = [int(m.group(1)) for x in os.listdir(f'{DRIVE}/runs') + os.listdir(launches) if (m := re.search(rf'{date}_r(\\d+)', x))]
    LAUNCH = f'{date}_r{max(taken, default=0) + 1}'
    seeds = [int(s) for s in str(SEEDS).split(',') if s.strip()]
    plan = dict(launch=LAUNCH, tag=TAG, folder=FOLDER, created=now.strftime('%Y-%m-%d %H:%M'), top=top, jobs=jobs(top, DATASETS, seeds, LAUNCH, N_DOCS))
    json.dump(plan, open(plan_file(LAUNCH), 'w'), ensure_ascii=False, indent=1)
    print(f'new launch {LAUNCH} -> {plan_file(LAUNCH)}')
JOBS = plan['jobs']; RUNS = [j['run'] for j in JOBS]
TIMING_RUNS = list({j['name']: j['run'] for j in reversed(JOBS)}.values())  # CPU speed depends on the model, not on the data or seed
print(f'\\nLAUNCH = {LAUNCH}  (put it into the LAUNCH field to continue this launch after a restart)\\n')
for i, m in enumerate(plan['top'], 1):
    past = f"mean test macro F1 {m['score']:.4f} over {len(m['runs'])} run(s): {', '.join(m['runs'])}" if 'score' in m else 'from MODELS'
    print(f"{i}. {m['name']:22s} {m['model']}  lr {m['lr']} batch {m['batch']}x{m['accum']} epochs {m['epochs']} {m['extra']}\\n   {past}")
print(f'\\n{len(JOBS)} runs on {FOLDER}:', *RUNS, sep='\\n  ')""".replace('__TAG__', tag).replace('__FOLDER__', folder).replace('__SEEDS__', seeds))


RUN_LOOP = code("""#@title 4. Train (every run goes to its own folder runs/<model>_<tag>_<launch>[_s<seed>]; saved to Drive right after training)
import json, os, time
n = len(JOBS); t_all = time.time(); loaded = None
for i, j in enumerate(JOBS, 1):
    run = j['run']
    if j['data'] != loaded:
        if not os.path.exists(f"{DRIVE}/{j['data']}/train.parquet") and not (j['data'] == 'silver' and os.path.exists(f'{DRIVE}/train.parquet')):
            print(f"\\n[{i}/{n}] {run}: no train.parquet in {DRIVE}/{j['data']} - skipped", flush=True)
            continue
        use_data(j['data']); loaded = j['data']
    print(f"\\n[{i}/{n}  {int(100 * (i - 1) / n)}%] ===== {run}: {j['model']} seed {j['seed']} {j['extra']} =====", flush=True)
    opts = f" --seed {j['seed']} {j['extra']}" + (f" --n-docs {j['n_docs']}" if j['n_docs'] else '')
    t0 = time.time()
    # --skip-if-done: a run of this launch that is already on Drive (the notebook was restarted) is not trained again
    rc = os.system(f"python scripts/train.py --model {j['model']} --out runs/{run} --epochs {j['epochs']} --lr {j['lr']} --batch {j['batch']} "
                   f"--grad-accum {j['accum']} --dev-docs 500 --skip-if-done {DRIVE}/runs/{run}{opts}")
    cfg = json.load(open(f'runs/{run}/config.json')) if os.path.exists(f'runs/{run}/config.json') else {}
    if rc != 0 or cfg.get('status') != 'done':
        print(f'\\n[{i}/{n}] {run} FAILED (exit {rc}, status {cfg.get("status")}) - see runs/{run}/train.log; stopping the loop', flush=True)
        break
    old = json.load(open(f'{DRIVE}/runs/{run}/config.json')) if os.path.exists(f'{DRIVE}/runs/{run}/model') else None
    if old and old.get('fingerprint') != cfg.get('fingerprint'):
        # the data on Drive changed since the launch started: the model saved earlier is kept
        print(f'[{i}/{n}] {run}: another model is already saved under this name - NOT overwritten; start a new launch (empty LAUNCH)', flush=True)
        continue
    !rm -rf runs/$run/checkpoints && mkdir -p $DRIVE/runs/$run && cp -r runs/$run/. $DRIVE/runs/$run/
    print(f'[{i}/{n}] {run}: fingerprint={cfg.get("fingerprint")} '
          f'({(time.time()-t0)/60:.1f} min; total {(time.time()-t_all)/60:.0f} min) -> saved to Drive', flush=True)
print(f'\\n[{n}/{n}  100%] done in {(time.time()-t_all)/60:.0f} min')""")

CPU_TIMING = code("""#@title 5. CPU inference timing (Colab CPU is slower than a laptop; use it for ranking, not absolute numbers)
import json, os, subprocess, time
from tqdm.auto import tqdm
p = f'{DRIVE}/runs/cpu_timing_leaderboard.json'
timing = json.load(open(p)) if os.path.exists(p) else {}  # timings of earlier launches are kept
if CPU_TIMING_DOCS:
    !head -n $CPU_TIMING_DOCS ../data/test/court_cases.jsonl > /tmp/test_small.jsonl
    for run in tqdm([r for r in TIMING_RUNS if os.path.exists(f'runs/{r}/model')], desc='cpu timing'):  # one run per model
        t0 = time.time()
        subprocess.run(['python','scripts/predict.py','--model',f'runs/{run}/model','--input','/tmp/test_small.jsonl','--output',f'/tmp/{run}_cpu.jsonl','--device','cpu','--threads','4','--batch','4'], capture_output=True)
        timing[run] = (time.time() - t0) / CPU_TIMING_DOCS
        print(f'{run:40s} {timing[run]:5.2f} s/doc -> test (240 docs) ~ {timing[run]*240/60:4.1f} min')
    json.dump(timing, open(p, 'w'))""")

# Nothing is published by itself: the cell does nothing until PUBLISH is ticked and a run is named. It calls
# scripts/publish.py, the script that is also run by hand, so the repo name and the licence follow the same rules.
CELL_PUBLISH = code("""#@title 7. Publish ONE model to Hugging Face - by hand: choose the run, tick PUBLISH, run the cell
PUBLISH = False  #@param {type:"boolean"}  # unticked = only the list of the runs on Drive is shown
RUN = ""  #@param {type:"string"}  # the run to publish: a folder of MyDrive/ai-camp/runs, as the list below and the leaderboard name it
HF_USER = ""  #@param {type:"string"}  # your HF username or org
HF_REPO = ""  #@param {type:"string"}  # name of the model on the Hub; empty = the name of the run
HF_PRIVATE = False  #@param {type:"boolean"}  # False = public repo, as scripts/publish.py does by default
# licence id of the model card. The task rules ask for an open licence; mit is the licence of the training corpus
# lang-uk/court-decisions-uk and of the wechsel base model. A card that already exists on the Hub is not rewritten.
HF_LICENSE = "mit"  #@param {type:"string"}
import json, os, re, subprocess
runs_dir = f'{DRIVE}/runs'
on_drive = sorted(r for r in os.listdir(runs_dir) if os.path.exists(f'{runs_dir}/{r}/model/config.json'))
mine = set(globals().get('RUNS', []))  # the runs of this launch, when cell 3 was run
print(f'{len(on_drive)} runs with a model in {runs_dir} (* = this launch):')
for r in on_drive:
    cfg = json.load(open(f'{runs_dir}/{r}/config.json')) if os.path.exists(f'{runs_dir}/{r}/config.json') else {}
    print(f"  {'*' if r in mine else ' '} {r:50s} {cfg.get('model', '')}")
RUN, HF_USER, HF_REPO = RUN.strip(), HF_USER.strip(), HF_REPO.strip()
if not PUBLISH:
    print('\\nPUBLISH is not ticked: nothing is published')
elif RUN not in on_drive:
    print(f'\\nNOT published: no run «{RUN}» with a model on Drive - copy a name from the list into RUN')
elif not HF_USER:
    print('\\nNOT published: HF_USER is empty')
else:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')  # Colab: left sidebar -> Secrets -> HF_TOKEN (write access), enable notebook access
    tag = re.search(r'_(v\\d+[a-z]*)_\\d{4}-', RUN)  # wechsel_base_v7large_2026-10-09_r1 -> v7large
    data = f'the synthetic dataset `{tag.group(1)}`' if tag else 'a synthetic dataset'
    note = (f'Trained on {data}: court decisions of the Unified State Register of Court Decisions of Ukraine whose anonymised '
            'fragments are filled with generated values. The generated addresses come from the Ukrposhta directory and from '
            'OpenStreetMap (© OpenStreetMap contributors, ODbL).')
    repo = f'{HF_USER}/{HF_REPO or RUN}'
    print(f"\\npublishing {RUN} -> https://huggingface.co/{repo} ({'private' if HF_PRIVATE else 'PUBLIC'}, licence {HF_LICENSE}) ...", flush=True)
    cmd = ['python', 'scripts/publish.py', '--model', f'{runs_dir}/{RUN}', '--repo', repo, '--license', HF_LICENSE, '--note', note,
           '--message', f'Upload {RUN}'] + (['--private'] if HF_PRIVATE else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f'published: {r.stdout.strip()}' if r.returncode == 0 else f'NOT published: {(r.stderr or r.stdout).strip()[-400:]}')""")

# scripts/leaderboard.py scores the runs on the test set (once per trained model), appends new evaluations to
# runs/test_history.jsonl and rewrites the Markdown file: leaderboard, seed averages, progress, runs over time, history.
CELL_LEADERBOARD = code("""#@title 6. Leaderboard on the organizers' test set: this launch + every earlier run on Drive -> runs/leaderboard.md
from IPython.display import Markdown, display
!python scripts/leaderboard.py --drive $DRIVE --name leaderboard --all
display(Markdown(open(f'{DRIVE}/runs/leaderboard.md', encoding='utf-8').read()))""")


# tag, Drive folder, default seeds, what the data is, rough A100 time of one run: base-size model / large-size model
VARIANTS = {
    "base": dict(tag="v7", folder="silver", seeds="42,43", what="`data/silver/` of the repo: 16k documents, 83M characters",
                 time="about 20 min for a base-size model and 1 hour for a large one, so the default launch (1 base + 1 large "
                      "model, 2 seeds) is about 2.7 hours if windows of 2048 tokens cost what the same text in windows of 512 did "
                      "(not measured)"),
    "large": dict(tag="v7large", folder="silver_large", seeds="42,43",
                  what="`data/silver_large/`: 42k documents, 296M characters - more of the general corpus, longer Supreme Court "
                       "decisions, every decision with an anonymised institution filled twice",
                  time="about 1.2 hours for a base-size model and 3.5 hours for a large one, so the default launch (1 base + 1 "
                       "large model, 2 seeds) is about 9.5 hours by the same guess - more than one Colab session may be needed: continue with `LAUNCH`"),
}


def train_notebook(variant: str) -> list[dict]:
    v = VARIANTS[variant]
    other = "large" if variant == "base" else "base"
    intro = md(f"# Pseudonymizer - train on the «{variant}» dataset\n\n"
               f"One of two training notebooks: this one trains on `MyDrive/ai-camp/{v['folder']}/` (tag `{v['tag']}`) - {v['what']}. "
               f"`colab_train_{other}.ipynb` does the same on the «{other}» dataset; the two can run at the same time in two Colab "
               "sessions and write to the same leaderboard.\n\n"
               "Each launch:\n\n"
               "1. trains the two models of `MODELS`: `wechsel_base` (0.8530 on the test set on 2026-10-08) and "
               "`Goader/modern-liberta-large`, a Ukrainian ModernBERT, with windows of 2048 tokens instead of 512 (no result "
               "yet) - each on this dataset "
               "with every seed of `SEEDS`. `TOP_N` > 0 adds that many of the best other models of the past runs on Drive;\n"
               "2. scores every run of `MyDrive/ai-camp/runs/` on the organizers' test set (a run already scored on the current "
               "test file is not predicted again) and writes the leaderboard over all runs, old and new, to "
               "`MyDrive/ai-camp/runs/leaderboard.md`.\n\n"
               "**Nothing is overwritten.** A launch gets a name from the date (Kyiv time) and its number on that day, "
               "`2026-10-09_r1`, `2026-10-09_r2`, …, and every run of it is saved to its own folder "
               f"`MyDrive/ai-camp/runs/<model>_{v['tag']}_<launch>[_s<seed>]/`, e.g. `wechsel_base_{v['tag']}_2026-10-09_r1_s43` (seed 42 "
               "has no suffix). Nothing goes to Hugging Face by itself: the last cell publishes one run "
               "that you name, and only when `PUBLISH` is ticked - through `scripts/publish.py`, to `<HF_USER>/<HF_REPO or the run name>`, "
               "public unless `HF_PRIVATE` is ticked, with a model card that names the licence (`HF_LICENSE`, mit by default), the "
               "base model and the sources of the training data. The plan "
               f"of a launch (chosen models, runs) is kept in `runs/launches/<launch>_{v['tag']}.json`.\n\n"
               "If Colab disconnects, put the launch name into `LAUNCH` and run all cells again: the finished runs are skipped "
               "and the models stay the ones chosen when the launch started. Leave `LAUNCH` empty for a new launch.\n\n"
               f"Runtime: **A100** (L4 works, ~2x slower); enable *Background execution*. One run takes {v['time']}.\n\n"
               "**Read the table «Averages over seeds».** Two identical runs differ by up to 3.5 points on the test set: a "
               "difference smaller than the spread (max − min) inside a group is noise.")
    return [intro, *SETUP, cell_plan(v["tag"], v["folder"], v["seeds"]), RUN_LOOP, CPU_TIMING, CELL_LEADERBOARD, CELL_PUBLISH]


# The notebooks for a rented machine with several GPUs (vast.ai) follow the Colab ones cell by cell. Cells 3 and 4 are
# scripts/launch.py; Google Drive is reached through rclone (scripts/drive.sh), because a container cannot mount it.
VAST_SETUP = code("""#@title 1. Code and the machine
import os, subprocess, sys
os.chdir('/workspace/ai-camp/train')  # the code is copied here from the laptop: train/scripts/vast.sh push
if os.path.exists('/venv/main/bin') and '/venv/main/bin' not in os.environ['PATH']:
    os.environ['PATH'] = '/venv/main/bin:' + os.environ['PATH']  # the Python environment of the vast.ai image
os.environ.update(TRANSFORMERS_VERBOSITY='error', PYTHONWARNINGS='ignore', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1')
!bash scripts/vast_setup.sh""")


def vast_data(variant: str) -> dict:
    return code("""#@title 2. Google Drive: the dataset and the history of the runs
VARIANT = '__VARIANT__'
sys.path.insert(0, 'scripts')
from launch import VARIANTS, file_sha1
TAG, FOLDER = VARIANTS[VARIANT]['tag'], VARIANTS[VARIANT]['folder']
DRIVE_ON = 'gdrive:' in subprocess.run(['rclone', 'listremotes'], capture_output=True, text=True).stdout.split()
if DRIVE_ON:
    # as in Colab, the training data is the one of MyDrive/ai-camp/<folder>
    !bash scripts/drive.sh data $FOLDER
    !bash scripts/drive.sh history
else:
    print('Google Drive is NOT connected: the runs will stay on this machine and the data is the one copied by vast.sh push.\\n'
          'To connect it, on the laptop: train/scripts/vast.sh drive')
for need in ('train.parquet', 'dev.parquet', 'dev.jsonl'):
    assert os.path.exists(f'../data/{FOLDER}/{need}'), f'{need} is missing in data/{FOLDER} - upload it to MyDrive/ai-camp/{FOLDER} or run vast.sh push'
assert os.path.exists('../data/evaluate.py') and os.path.exists('../data/test/court_cases.jsonl'), 'data/evaluate.py or the test set is missing: vast.sh push'
print(f"OK: data/{FOLDER}, train.parquet {file_sha1(f'../data/{FOLDER}/train.parquet')}, tag {TAG}")""".replace('__VARIANT__', variant))


def vast_plan(seeds: str) -> dict:
    return code("""#@title 3. Models and the plan of this launch
TOP_N = 0            # besides MODELS, add this many of the best models of the test history (0 = none)
LAUNCH = ""          # empty = a new launch (today's date + next number); "2026-10-09_r1" continues that launch
SEEDS = "__SEEDS__"  # comma-separated; every model is trained once per seed and the leaderboard averages them
N_DOCS = 0           # 0 = all; e.g. 2000 for a smoke run
CPU_TIMING_DOCS = 30 # test docs for the CPU speed estimate (0 = skip)
HF_PUBLISH = True    # publish every trained model as the PUBLIC repo <your HF account>/<run> (token: vast.sh hf); runs with N_DOCS are never published
# the models are MODELS of scripts/launch.py: wechsel_base and modern_liberta_large_w2048 (windows of 2048 tokens)
import json, re
args = ['--variant', VARIANT] + (['--launch', LAUNCH] if LAUNCH else ['--seeds', SEEDS, '--n-docs', str(N_DOCS), '--top-n', str(TOP_N)])
out = subprocess.run(['python', 'scripts/launch.py', *args, '--plan-only'], capture_output=True, text=True)
print(out.stdout, out.stderr[-2000:])
LAUNCH = re.search(r'LAUNCH = (\\S+)', out.stdout).group(1)
plan = json.load(open(f'runs/launches/{LAUNCH}_{TAG}.json'))
JOBS = plan['jobs']; RUNS = [j['run'] for j in JOBS]
TIMING_RUNS = list({j['name']: j['run'] for j in reversed(JOBS)}.values())  # CPU speed depends on the model, not on the data or seed""".replace('__SEEDS__', seeds))


VAST_TRAIN = code("""#@title 4. Train: one run per GPU; every run goes to runs/<model>_<tag>_<launch>[_s<seed>] and to Drive right after training
# The training runs in tmux, not in this kernel: a closed tab or a stopped cell (the ■ button) does not stop it. Run the
# cell again to follow the log, or stop it and use the next cell; to stop the training itself: !tmux kill-session -t =train_<tag>
import time
session, log = f'train_{TAG}', f'runs/launch_{TAG}.log'
# '=': the exact session name; without it tmux takes train_v7 for train_v7large (it matches by prefix)
alive = lambda: subprocess.run(['tmux', 'has-session', '-t', '=' + session], capture_output=True).returncode == 0
if alive():
    print(f'a launch of {TAG} is already running - following its log')
else:
    py = subprocess.run('command -v python', shell=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, f"{py} scripts/launch.py --variant {VARIANT} --launch {LAUNCH}{'' if HF_PUBLISH else ' --no-publish'} 2>&1 | tee -a {log}"], check=True)
    time.sleep(3)
with open(log, encoding='utf-8') as f:
    f.seek(max(0, os.path.getsize(log) - 3000))
    while True:
        running = alive()
        print(f.read(), end='', flush=True)
        if not running:
            break
        time.sleep(20)
!python scripts/launch.py --variant $VARIANT --launch $LAUNCH --status""")

VAST_STATUS = code("""#@title 4a. Progress - run at any time (stop cell 4 with ■ first: the training goes on)
# per run: waiting / running (GPU, % of the steps, epoch, time left, dev F1 of the finished epochs) / done (dev F1, minutes,
# Drive, Hugging Face) / FAILED (the error). The same from the laptop: train/scripts/vast.sh status
!python scripts/launch.py --variant base --status
!python scripts/launch.py --variant large --status
!nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader""")

VAST_TIMING = code("""#@title 5. CPU inference timing (use it for ranking, not absolute numbers)
p = 'runs/cpu_timing_leaderboard.json'
timing = json.load(open(p)) if os.path.exists(p) else {}  # timings of earlier launches are kept
if CPU_TIMING_DOCS:
    !head -n $CPU_TIMING_DOCS ../data/test/court_cases.jsonl > /tmp/test_small.jsonl
    for run in [r for r in TIMING_RUNS if os.path.exists(f'runs/{r}/model')]:  # one run per model
        t0 = time.time()
        subprocess.run(['python','scripts/predict.py','--model',f'runs/{run}/model','--input','/tmp/test_small.jsonl','--output',f'/tmp/{run}_cpu.jsonl','--device','cpu','--threads','4','--batch','4'], capture_output=True)
        timing[run] = (time.time() - t0) / CPU_TIMING_DOCS
        print(f'{run:40s} {timing[run]:5.2f} s/doc -> test (240 docs) ~ {timing[run]*240/60:4.1f} min')
    json.dump(timing, open(p, 'w'))""")

VAST_LEADERBOARD = code("""#@title 6. Leaderboard on the organizers' test set: this launch + the history of Drive -> runs/leaderboard.md, also on Drive
from IPython.display import Markdown, display
!bash scripts/drive.sh history
!python scripts/leaderboard.py --drive . --name leaderboard --all
!bash scripts/drive.sh results
display(Markdown(open('runs/leaderboard.md', encoding='utf-8').read()))""")

VAST_PUBLISH = code("""#@title 7. Publish ONE model to Hugging Face - by hand: choose the run, set PUBLISH = True, run the cell
PUBLISH = False      # False = only the list of the runs is shown
RUN = ""             # the run to publish: a folder of train/runs or of MyDrive/ai-camp/runs, as the list below and the leaderboard name it
HF_USER = ""         # your HF username or org
HF_REPO = ""         # name of the model on the Hub; empty = the name of the run
HF_PRIVATE = False   # False = public repo, as scripts/publish.py does by default
# licence id of the model card. The task rules ask for an open licence; mit is the licence of the training corpus
# lang-uk/court-decisions-uk and of the wechsel base model. A card that already exists on the Hub is not rewritten.
HF_LICENSE = "mit"
here = sorted(r for r in os.listdir('runs') if os.path.exists(f'runs/{r}/model/config.json'))
there = [r.rstrip('/') for r in subprocess.run(['rclone', 'lsf', 'gdrive:runs', '--dirs-only'], capture_output=True, text=True).stdout.split()] if globals().get('DRIVE_ON') else []
mine = set(globals().get('RUNS', []))  # the runs of this launch, when cell 3 was run
print(f'{len(here)} runs with a model on this machine (* = this launch); + = only on Drive:')
for r in sorted(set(here) | set(there) - {'launches'}):
    print(f"  {'*' if r in mine else ' ' if r in here else '+'} {r}")
RUN, HF_USER, HF_REPO = RUN.strip(), HF_USER.strip(), HF_REPO.strip()
# the token (write access) is stored by train/scripts/vast.sh hf; `hf auth login` on the instance puts it under HF_HOME
token_files = [os.path.expanduser('~/.cache/huggingface/token'), os.path.join(os.environ.get('HF_HOME', '/workspace/.hf_home'), 'token')]
token = os.environ.get('HF_TOKEN') or next((open(f).read().strip() for f in token_files if os.path.exists(f)), '')
if not PUBLISH:
    print('\\nPUBLISH is False: nothing is published')
elif RUN not in here and RUN not in there:
    print(f'\\nNOT published: no run «{RUN}» - copy a name from the list into RUN')
elif not HF_USER:
    print('\\nNOT published: HF_USER is empty')
elif not token:
    print('\\nNOT published: no Hugging Face token on this machine - on the laptop: train/scripts/vast.sh hf')
else:
    if RUN not in here:
        !rclone copy gdrive:runs/$RUN runs/$RUN
    tag = re.search(r'_(v\\d+[a-z]*)_\\d{4}-', RUN)  # wechsel_base_v7large_2026-10-09_r1 -> v7large
    data = f'the synthetic dataset `{tag.group(1)}`' if tag else 'a synthetic dataset'
    note = (f'Trained on {data}: court decisions of the Unified State Register of Court Decisions of Ukraine whose anonymised '
            'fragments are filled with generated values. The generated addresses come from the Ukrposhta directory and from '
            'OpenStreetMap (© OpenStreetMap contributors, ODbL).')
    repo = f'{HF_USER}/{HF_REPO or RUN}'
    print(f"\\npublishing {RUN} -> https://huggingface.co/{repo} ({'private' if HF_PRIVATE else 'PUBLIC'}, licence {HF_LICENSE}) ...", flush=True)
    cmd = ['python', 'scripts/publish.py', '--model', f'runs/{RUN}', '--repo', repo, '--license', HF_LICENSE, '--note', note,
           '--message', f'Upload {RUN}'] + (['--private'] if HF_PRIVATE else [])
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, 'HF_TOKEN': token})
    print(f'published: {r.stdout.strip()}' if r.returncode == 0 else f'NOT published: {(r.stderr or r.stdout).strip()[-400:]}')""")


def vast_notebook(variant: str) -> list[dict]:
    v = VARIANTS[variant]
    other = "large" if variant == "base" else "base"
    intro = md(f"# Pseudonymizer - train on the «{variant}» dataset on a machine with several GPUs (vast.ai)\n\n"
               f"The vast.ai twin of `colab_train_{variant}.ipynb`: the same models, settings, run names and leaderboard, on "
               f"`MyDrive/ai-camp/{v['folder']}/` (tag `{v['tag']}`) - {v['what']}. `vast_train_{other}.ipynb` does the same on the "
               f"«{other}» dataset; the two can run at the same time on one machine - a GPU trains one model at a time.\n\n"
               "Once per instance, on the laptop: `train/scripts/vast.sh push` (the code and the test set), "
               "`train/scripts/vast.sh drive` (access to Google Drive), `train/scripts/vast.sh hf` (the Hugging Face token, write "
               "access). Then `train/scripts/vast.sh ssh`, open https://localhost:8080 and this file in "
               "`/workspace/ai-camp/train/`. After a change of the code run `vast.sh push` again.\n\n"
               "What differs from Colab:\n\n"
               "- the runs of a launch are trained in parallel, one per GPU, the large models first; the training runs in tmux, "
               "so the notebook may be closed;\n"
               "- Google Drive is not mounted: the dataset is the one of `MyDrive/ai-camp/<folder>` when that folder exists there "
               "(otherwise the one copied from the laptop by `vast.sh push`), every run is copied to "
               "`MyDrive/ai-camp/runs/<run>/` right after training, the leaderboard files are exchanged in cell 6;\n"
               "- the leaderboard predicts the test set only with the models that are on this machine; the earlier runs of "
               "Drive enter it with the test results saved there, so a run of Drive that was scored on another test file or "
               "decoder stays in the history tables only. For the same reason `TOP_N` ranks the history as it is.\n\n"
               "**Nothing is overwritten**: a launch gets the name `<date>_r<N>` (Kyiv time; the numbers taken on Drive are "
               f"skipped), a run the folder `<model>_{v['tag']}_<launch>[_s<seed>]`, the plan is "
               f"`runs/launches/<launch>_{v['tag']}.json`. To continue an interrupted launch put its name into `LAUNCH` and run all "
               "cells again: the finished runs are skipped.\n\n"
               "**Hugging Face.** Unlike in Colab, with `HF_PUBLISH = True` every model trained on the whole dataset is published "
               "right after training as the **public** repo `<your HF account>/<run>` (licence mit, a model card with the base "
               "model and the sources of the data); a smoke run (`N_DOCS`) is never published. Cell 7 publishes one run by hand, "
               "e.g. under another name.\n\n"
               "**Progress.** Cell 4 follows the log; cell 4a (or `train/scripts/vast.sh status` on the laptop) shows every run: "
               "GPU, share of the steps, epoch, time left, dev F1 after each epoch, whether it reached Drive and Hugging Face, "
               "and the error of a failed run.\n\n"
               "**Read the table «Averages over seeds».** Two identical runs differ by up to 3.5 points on the test set: a "
               "difference smaller than the spread (max − min) inside a group is noise.")
    return [intro, VAST_SETUP, vast_data(variant), vast_plan(v["seeds"]), VAST_TRAIN, VAST_STATUS, VAST_TIMING, VAST_LEADERBOARD, VAST_PUBLISH]


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
    OUT.mkdir(exist_ok=True)
    for old in (HERE / "colab_train.ipynb", HERE / "vast_train.ipynb",  # the single notebooks that the pairs replaced
                *(HERE / f"{k}_train_{v}.ipynb" for k in ("colab", "vast") for v in VARIANTS)):  # before train/notebooks/
        if old.exists():
            old.unlink()
    for name, cells in ((f"colab_train_{v}.ipynb", train_notebook(v)) for v in VARIANTS):
        check_syntax(name, cells)
        nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}
        (OUT / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{OUT.relative_to(HERE)}/{name}: {len(cells)} cells")
    sys.path.insert(0, str(HERE / "scripts"))
    import launch  # the vast.ai notebooks train through scripts/launch.py: its datasets must be the ones described here
    assert {k: (v["tag"], v["folder"], v["seeds"]) for k, v in VARIANTS.items()} == \
           {k: (v["tag"], v["folder"], v["seeds"]) for k, v in launch.VARIANTS.items()}, "VARIANTS differ from scripts/launch.py"
    vast_meta = {"kernelspec": meta["kernelspec"], "language_info": meta["language_info"]}
    for name, cells in ((f"vast_train_{v}.ipynb", vast_notebook(v)) for v in VARIANTS):
        check_syntax(name, cells)
        (OUT / name).write_text(json.dumps({"cells": cells, "metadata": vast_meta, "nbformat": 4, "nbformat_minor": 0},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{OUT.relative_to(HERE)}/{name}: {len(cells)} cells")

if __name__ == "__main__":
    main()
