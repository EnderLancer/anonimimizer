#!/usr/bin/env bash
# Exchange the leaderboard files with Google Drive (MyDrive/ai-camp/runs) through rclone. Run on the instance from train/.
#
#   scripts/drive.sh data silver   # the training data of MyDrive/ai-camp/silver -> data/silver, as the Colab notebooks take it
#   scripts/drive.sh history    # before the leaderboard: the history and the cached test results of the runs on Drive
#   scripts/drive.sh results    # after it: the leaderboard, the history and the test results of the runs
#
# The models are uploaded by scripts/launch.py right after training. Does nothing when the rclone remote is not set up.
set -euo pipefail
cd "$(dirname "$0")/.."
DRIVE="${PSEUDO_DRIVE:-gdrive:}"
if ! rclone listremotes 2>/dev/null | grep -qx "${DRIVE%%:*}:"; then
  echo "Google Drive is not connected (on the laptop: train/scripts/vast.sh drive) - skipped"
  exit 0
fi
mkdir -p runs
case "${1:-}" in
  data)
    if rclone lsf "${DRIVE}$2/train.parquet" >/dev/null 2>&1; then
      rclone copy "${DRIVE}$2" "../data/$2" --include "/train.parquet" --include "/dev.parquet" --include "/dev.jsonl"
      echo "data/$2 <- ${DRIVE}$2"
    else
      echo "no ${DRIVE}$2/train.parquet on Drive: data/$2 stays the one copied from the laptop (vast.sh push)"
    fi
    ;;
  history)
    rclone copy "${DRIVE}runs" runs --include "/test_history.jsonl" --include "/cpu_timing_*.json" \
      --include "/*/config.json" --include "/*/test_result.json"
    ;;
  results)
    rclone copy runs "${DRIVE}runs" --include "/leaderboard.md" --include "/test_history.jsonl" --include "/cpu_timing_*.json" \
      --include "/*/test_result.json" --include "/*/test_result.txt" --include "/*/test_sub.jsonl"
    echo "leaderboard and test results -> ${DRIVE}runs"
    ;;
  *)
    sed -n '2,8p' "$0"; exit 1
    ;;
esac
