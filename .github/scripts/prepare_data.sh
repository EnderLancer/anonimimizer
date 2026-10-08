#!/usr/bin/env bash
# Копіює $TEST_DATA в out/data.jsonl; з $LIMIT — лише перші N непорожніх рядків (ті самі в усіх jobs).
set -euo pipefail
[ -f "$TEST_DATA" ] || { echo "::error::файл даних $TEST_DATA не знайдено"; exit 1; }
mkdir -p out
if [ -n "${LIMIT:-}" ]; then
  [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "::error::limit має бути додатним числом, отримано '$LIMIT'"; exit 1; }
  awk -v n="$LIMIT" 'NF && k++ < n' "$TEST_DATA" > out/data.jsonl
else
  cp "$TEST_DATA" out/data.jsonl
fi
echo "Документів: $(grep -c . out/data.jsonl) з $TEST_DATA"
