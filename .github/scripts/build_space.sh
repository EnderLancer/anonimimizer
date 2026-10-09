#!/usr/bin/env bash
# Збирає файли Hugging Face Space: space/* + модулі пайплайну + CRF-модель.
#   .github/scripts/build_space.sh [тека]      # за замовчуванням dist/space
set -euo pipefail
cd "$(dirname "$0")/../.."
out="${1:-dist/space}"
rm -rf "$out" && mkdir -p "$out"
cp space/app.py space/README.md space/requirements.txt "$out"/
cp run.py predict_trf.py predict.py common.py regex_persons.py ner_adapters.py ner_postprocess.py crf2.model "$out"/
python -m py_compile "$out"/*.py && rm -rf "$out/__pycache__"
ls -la "$out"
