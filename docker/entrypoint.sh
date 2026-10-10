#!/bin/sh
# Точка входу образу: псевдонімізує JSONL-файл.
#   docker run --rm -v "$PWD":/data pseudonymizer [вхід.jsonl] [вихід.jsonl] [інші параметри run.py]
# Відносні шляхи — у /data; за замовчуванням input.jsonl і output.jsonl, режим $MODE (model_regex).
# Формат рядка входу: {"id": ..., "original": "текст"}; виходу: {"id", "annotations": [{start, end, pseudonym}]}.
set -eu
src=${1:-input.jsonl}
dst=${2:-output.jsonl}
[ $# -ge 2 ] && shift 2 || shift $#
# відносні шляхи — відносно змонтованої теки /data
case $src in /*) ;; *) src=/data/$src ;; esac
case $dst in /*) ;; *) dst=/data/$dst ;; esac
if [ ! -f "$src" ]; then
    echo "немає вхідного файлу $src: змонтуйте теку з даними в /data (-v \"\$PWD\":/data)" >&2
    exit 2
fi
# кеш виходу моделі індексується id документа; у контейнері він не має переживати запуск
rm -rf /app/ner_cache && mkdir -p /app/ner_cache
exec python /app/run.py "$src" "$dst" --ner "${NER_SPEC}" --mode "${MODE}" "$@"
