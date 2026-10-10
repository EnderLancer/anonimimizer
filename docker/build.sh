#!/usr/bin/env bash
# Збирає образ і зберігає його в архів, який запускається без мережі.
#   docker/build.sh                      # pseudonymizer:latest для платформи цієї машини -> dist/pseudonymizer.tar
#   docker/build.sh linux/amd64          # образ для x86-64 (на Apple Silicon через емуляцію)
#   NER_MODEL=org/model docker/build.sh  # інша NER-модель з Hugging Face
set -euo pipefail
cd "$(dirname "$0")/.."
platform="${1:-}"
tag="${TAG:-pseudonymizer:latest}"
out="${OUT:-dist/pseudonymizer${platform:+-${platform##*/}}.tar}"
args=(-f docker/Dockerfile -t "$tag" .)
[ -n "$platform" ] && args+=(--platform "$platform")
[ -n "${NER_MODEL:-}" ] && args+=(--build-arg "NER_MODEL=$NER_MODEL")
[ -n "${HF_TOKEN:-}" ] && args+=(--secret id=hf_token,env=HF_TOKEN)
docker build "${args[@]}"
mkdir -p "$(dirname "$out")"
docker save "$tag" -o "$out"
ls -lh "$out"
echo "Завантажити на іншій машині: docker load -i $(basename "$out")"
