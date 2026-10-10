# syntax=docker/dockerfile:1
# Образ для локального інференсу: правила + NER-модель (ONNX Runtime, CPU), без мережі під час роботи.
#   docker build -f docker/Dockerfile -t pseudonymizer .
#   docker build -f docker/Dockerfile -t pseudonymizer --build-arg NER_MODEL=org/model .   # інша модель з Hugging Face
#   docker build -f docker/Dockerfile -t pseudonymizer --build-arg ONNX_QUANTIZE=0 .       # fp32 замість int8
#   docker run --rm -v "$PWD":/data pseudonymizer input.jsonl output.jsonl
# Стадії: weights (ваги з Hugging Face) ──► export (torch -> ONNX, звірка logits) ──► образ лише з onnxruntime.
# torch і transformers у кінцевий образ не потрапляють. Приватна модель — секрет збірки: --secret id=hf_token,env=HF_TOKEN.
ARG PYTHON=3.11
ARG NER_MODEL=davidmelash/wechsel_base_v7_2026-10-09_r5_s43
ARG ONNX_QUANTIZE=1

# ---- ваги моделі ----
FROM python:${PYTHON}-slim AS weights
ARG NER_MODEL
RUN pip install --no-cache-dir "huggingface_hub>=0.30"
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(os.environ['NER_MODEL'], local_dir='/model', token=os.environ.get('HF_TOKEN') or None, \
allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'LICENSE*'])" \
    && rm -rf /model/.cache && ls -la /model

# ---- експорт у ONNX ----
FROM python:${PYTHON}-slim AS export
ARG ONNX_QUANTIZE
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "transformers>=4.51,<6" onnx onnxruntime onnxscript
COPY --from=weights /model /model
COPY docker/export_onnx.py /export_onnx.py
RUN python /export_onnx.py /model /onnx $([ "$ONNX_QUANTIZE" = 1 ] && echo --quantize) && ls -la /onnx

# ---- образ ----
FROM python:${PYTHON}-slim
ARG NER_MODEL
ARG ONNX_QUANTIZE
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm -rf /root/.cache/pip
WORKDIR /app
COPY run.py predict_trf.py regex_persons.py ner_adapters.py ner_postprocess.py /app/
COPY docker/entrypoint.sh docker/bench.py /app/
COPY --from=export /onnx /app/models/ner
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    NER_SPEC=onnx:/app/models/ner MODE=model_regex NER_MODEL=${NER_MODEL} ONNX_QUANTIZE=${ONNX_QUANTIZE}
LABEL org.opencontainers.image.title="pseudonymizer" \
      org.opencontainers.image.description="Псевдонімізація українських судових рішень: правила + NER (${NER_MODEL}, ONNX Runtime), CPU" \
      org.opencontainers.image.licenses="MIT"
# перевірка, що все імпортується і модель читається без мережі
RUN python -c "import run, ner_adapters as N; print(N.load_adapter('onnx:/app/models/ner').predict(['Позивач Іваненко Петро Васильович.'])[0])"
VOLUME /data
ENTRYPOINT ["/app/entrypoint.sh"]
