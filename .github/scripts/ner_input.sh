#!/usr/bin/env bash
# out/ner_input.jsonl = $TEST_DATA + all files of $LARGER_DATA (space separated): the documents the NER model runs on.
set -euo pipefail
mkdir -p out
: > out/ner_input.jsonl
for f in "$TEST_DATA" ${LARGER_DATA:-}; do
  [ -f "$f" ] || { echo "::error::data file $f not found"; exit 1; }
  awk 'NF' "$f" >> out/ner_input.jsonl
done
echo "NER input: $(grep -c . out/ner_input.jsonl) documents"
