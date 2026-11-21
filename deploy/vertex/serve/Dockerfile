# Vertex AI custom serving container: FastAPI predictor (Gemma 3 preset + LoRA adapter).
# Build context = staged repo copy (vertex_stage_context in ../guard.sh); see ../README.md.
#
#   ACCEL=cuda12 (default) for g2-standard-8 + L4, ACCEL=cpu for a (slow) CPU-only replica.
FROM python:3.12-slim

ARG ACCEL=cuda12
ARG JAX_VERSION=0.12.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KERAS_BACKEND=jax \
    AIP_HTTP_PORT=8080 \
    AIP_HEALTH_ROUTE=/health \
    AIP_PREDICT_ROUTE=/predict \
    MODEL_PRESET=gemma3_instruct_1b \
    MODEL_DTYPE=bfloat16

WORKDIR /app
COPY . /app

RUN python -m pip install -U pip \
 && python -m pip install -e ".[train]" tensorflow-cpu safetensors sentencepiece \
        "fastapi>=0.110" "uvicorn[standard]>=0.29" google-cloud-storage google-cloud-secret-manager \
 && if [ "$ACCEL" = "cuda12" ]; then \
        python -m pip install "jax[cuda12]==${JAX_VERSION}" ; \
    elif [ "$ACCEL" != "cpu" ]; then \
        echo "unknown ACCEL=$ACCEL (use cuda12 or cpu)" >&2 ; exit 1 ; \
    fi

EXPOSE 8080
CMD ["python", "/app/deploy/vertex/serve/predictor.py"]
