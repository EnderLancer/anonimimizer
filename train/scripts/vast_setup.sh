#!/usr/bin/env bash
# Prepare a rented GPU machine for training: packages, a PyTorch build that supports the GPUs, a check of every GPU.
# Run on the instance from train/ (scripts/vast.sh setup does that).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux rsync)
if ! command -v rclone >/dev/null; then  # copies the runs to Google Drive
  command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip)
  curl -fsS https://rclone.org/install.sh | bash >/dev/null
fi

gpu_ok() {
  python - <<'EOF'
import sys, torch
if not torch.cuda.is_available():
    sys.exit(1)
for i in range(torch.cuda.device_count()):
    x = torch.ones(8, 8, device=f"cuda:{i}")
    assert float((x @ x).sum()) == 512.0
EOF
}

# The image usually ships PyTorch. RTX 50xx cards need a build for CUDA 12.8 (PyTorch 2.7+): an older one fails on the first GPU operation.
if ! gpu_ok 2>/dev/null; then
  echo "PyTorch is missing or does not support these GPUs: installing the CUDA 12.8 build"
  pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128
fi
pip install -q -e . 2>&1 | tail -1
gpu_ok

python - <<'EOF'
import os, shutil, torch, transformers
print(f"torch {torch.__version__}, transformers {transformers.__version__}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GB")
ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
print(f"CPU cores {os.cpu_count()}, RAM {ram:.0f} GB, free disk {shutil.disk_usage('.').free / 2**30:.0f} GB")
EOF
for d in ../data/*/; do
  [ -f "$d/train.parquet" ] && echo "dataset $(basename "$d"): $(du -sh "$d" | cut -f1)"
done
[ -f ../data/test/court_cases.jsonl ] && echo "test set: OK" || echo "test set: MISSING (scripts/vast.sh push)"
DRIVE="${PSEUDO_DRIVE:-gdrive:}"
if rclone listremotes 2>/dev/null | grep -qx "${DRIVE%%:*}:"; then
  echo "Google Drive: OK, $(rclone lsf "${DRIVE}runs" --dirs-only | wc -l) folders in ${DRIVE}runs"
else
  echo "Google Drive: NOT connected - the runs will stay on this machine (on the laptop: train/scripts/vast.sh drive)"
fi
