#!/usr/bin/env bash
# Копіює $TEST_DATA в out/data.jsonl; з $LIMIT — лише перші N непорожніх рядків (ті самі в усіх jobs).
# out/ner_input.jsonl = out/data.jsonl + файли з $LARGER (через пробіл): саме на ньому рахується NER.
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
# вхід для NER: основні дані + всі $LARGER (без дублікатів); оцінюється кожен файл окремо, лімит діє лише на основні
cp out/data.jsonl out/ner_input.jsonl
for f in ${LARGER:-}; do
  [ -f "$f" ] || { echo "::error::файл даних $f не знайдено"; exit 1; }
  [ "$f" = "$TEST_DATA" ] && continue
  awk 'NF' "$f" >> out/ner_input.jsonl
  echo "Документів: $(grep -c . "$f") з $f"
done
