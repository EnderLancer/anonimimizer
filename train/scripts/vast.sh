#!/usr/bin/env bash
# Train on a rented GPU machine (vast.ai) over SSH. Run from the laptop, in any directory.
#
#   export VAST_HOST=<ip> VAST_PORT=<port>     # from the "Connect" button of the instance; the port changes on restart
#   train/scripts/vast.sh push                # copy the code, the test set and the local datasets to the instance
#   train/scripts/vast.sh drive               # once per instance: give it access to MyDrive/ai-camp (needs rclone on the laptop)
#   train/scripts/vast.sh hf                  # once per instance: store a Hugging Face token there: every trained model is published
#   train/scripts/vast.sh setup               # install the packages there and check every GPU
#   train/scripts/vast.sh ssh                 # a shell on the instance; Jupyter is then at https://localhost:8080
#
# The notebooks train/notebooks/vast_train_{base,large}.ipynb (scripts/make_notebooks.py) do the rest. The same without Jupyter:
#   train/scripts/vast.sh train base [args]   # start scripts/launch.py --variant base in tmux (args: --n-docs 2000 --seeds 42 ...)
#   train/scripts/vast.sh train large [args]
#   train/scripts/vast.sh status              # progress of every run (GPU, steps, epoch, time left, dev F1, Drive, HF) + GPU load
#   train/scripts/vast.sh watch               # the same every minute (Ctrl-C stops watching, not the training)
#   train/scripts/vast.sh logs base           # follow the launcher output (Ctrl-C stops following, not the training)
#   train/scripts/vast.sh leaderboard         # score the runs on the test set there -> runs/leaderboard.md, also on Drive
#   train/scripts/vast.sh pull                # copy the runs back to train/runs/ without the models
#   train/scripts/vast.sh pull-models         # the same with the models (1-2 GB each)
set -euo pipefail

# the address of the instance is never written down here: export VAST_HOST and VAST_PORT in the shell
HOST="${VAST_HOST:?export VAST_HOST=<ip> VAST_PORT=<port> from the Connect button of the vast.ai instance}"
PORT="${VAST_PORT:?export VAST_PORT=<port> from the Connect button of the vast.ai instance}"
REMOTE="${VAST_DIR:-/workspace/ai-camp}"
DATASETS="${VAST_DATASETS:-silver silver_large}"
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST")
RSYNC=(rsync -az --progress -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new")
cd "$(dirname "$0")/../.."

# the Python environment of vast.ai images is activated in interactive shells only
remote() { "${SSH[@]}" "[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate >/dev/null; cd $REMOTE/train && mkdir -p runs && $*"; }
tag() { [ "${1:-}" = base ] && echo v7 || { [ "${1:-}" = large ] && echo v7large; } || { echo "say base or large" >&2; exit 1; }; }

case "${1:-}" in
  push)
    files=(train/pyproject.toml train/src train/scripts train/notebooks/vast_train_base.ipynb train/notebooks/vast_train_large.ipynb
           data/evaluate.py data/test/court_cases.jsonl)
    for d in $DATASETS; do
      for f in train.parquet dev.parquet dev.jsonl; do
        [ -f "data/$d/$f" ] && files+=("data/$d/$f") || echo "data/$d/$f is missing - skipped"
      done
    done
    "${SSH[@]}" "mkdir -p $REMOTE"
    "${RSYNC[@]}" -R --exclude __pycache__ "${files[@]}" "root@$HOST:$REMOTE/"
    ;;
  drive)
    command -v rclone >/dev/null || { echo "install rclone first: brew install rclone"; exit 1; }
    if ! rclone listremotes | grep -qx "gdrive:"; then
      # opens the browser for the Google login; the remote starts at the folder MyDrive/ai-camp
      rclone config create gdrive drive scope=drive root_folder_id="${VAST_DRIVE_FOLDER:-1wb4bAKpYwSxRIfUWSYDJoJPrefO4Wnlh}"
    fi
    rclone lsf gdrive: --dirs-only >/dev/null  # fails here when the login did not work
    rclone config show gdrive | "${SSH[@]}" "mkdir -p ~/.config/rclone && umask 077 && cat > ~/.config/rclone/rclone.conf"
    echo "the instance can now write to MyDrive/ai-camp"
    ;;
  hf)
    read -rsp "Hugging Face token (write access): " token; echo
    [ -n "$token" ] || { echo "no token given"; exit 1; }
    printf '%s' "$token" | "${SSH[@]}" "mkdir -p ~/.cache/huggingface && umask 077 && cat > ~/.cache/huggingface/token"
    echo "the token is stored on the instance"
    ;;
  setup)
    remote "bash scripts/vast_setup.sh"
    ;;
  train)
    t="$(tag "${2:-}")"; v="$2"; shift 2
    # "=": the exact session name; without it tmux takes train_v7 for train_v7large (it matches by prefix)
    remote "tmux has-session -t =train_$t 2>/dev/null && { echo 'a launch of $t is already running: vast.sh logs $v'; exit 1; }; \
            bash scripts/drive.sh data \$(python -c \"import sys; sys.path.insert(0, 'scripts'); import launch; print(launch.VARIANTS['$v']['folder'])\"); \
            bash scripts/drive.sh history; \
            tmux new-session -d -s train_$t \"\$(command -v python) scripts/launch.py --variant $v $* 2>&1 | tee -a runs/launch_$t.log\"; \
            sleep 8; tail -n 40 runs/launch_$t.log"
    ;;
  status)
    remote "python scripts/launch.py --variant base --status; python scripts/launch.py --variant large --status; \
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader"
    ;;
  watch)
    while :; do clear; date; "$0" status 2>&1 | grep -v -e "^Welcome" -e "^Have fun" -e "^AI agents"; sleep 60; done
    ;;
  logs)
    remote "tail -n 40 -f runs/launch_$(tag "${2:-}").log"
    ;;
  leaderboard)
    remote "bash scripts/drive.sh history && python scripts/leaderboard.py --drive . --name leaderboard --all && bash scripts/drive.sh results && cat runs/leaderboard.md"
    ;;
  pull)
    "${RSYNC[@]}" --exclude model --exclude checkpoints "root@$HOST:$REMOTE/train/runs/" train/runs/
    ;;
  pull-models)
    "${RSYNC[@]}" --exclude checkpoints "root@$HOST:$REMOTE/train/runs/" train/runs/
    ;;
  ssh)
    "${SSH[@]}" -L 8080:localhost:8080
    ;;
  *)
    sed -n '2,19p' "$0"; exit 1
    ;;
esac
