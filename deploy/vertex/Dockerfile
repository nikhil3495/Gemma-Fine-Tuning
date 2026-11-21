# Vertex AI custom-training container for recipe-gemma (LoRA fine-tuning of Gemma 3).
#
# Build context = a STAGED copy of the repo (see vertex_stage_context in guard.sh), never the
# raw working tree, so .venv/, data/, checkpoints/ and any .env never enter the image.
#
#   ACCEL=cuda12 (default): NVIDIA GPUs (e.g. g2-standard-8 + 1x L4)
#   ACCEL=tpu             : Cloud TPU VMs (libtpu wheels)
#
# docker build --build-arg ACCEL=cuda12 -t recipe-gemma-train <staged-context>
FROM python:3.12-slim

ARG ACCEL=cuda12
ARG JAX_VERSION=0.12.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KERAS_BACKEND=jax \
    APP_DIR=/app \
    CONFIG=configs/train_full.yaml \
    DATA_CONFIG=configs/data.yaml

WORKDIR /app
COPY . /app

# Editable install on purpose: recipe_gemma resolves REPO_ROOT from its source location, so
# configs/, data/, checkpoints/ and results/ live under /app.
RUN python -m pip install -U pip \
 && python -m pip install -e ".[train,eval,gcp]" tensorflow-cpu safetensors sentencepiece \
        google-cloud-storage google-cloud-secret-manager \
 && if [ "$ACCEL" = "tpu" ]; then \
        python -m pip install "jax[tpu]==${JAX_VERSION}" \
            -f https://storage.googleapis.com/jax-releases/libtpu_releases.html ; \
    elif [ "$ACCEL" = "cuda12" ]; then \
        python -m pip install "jax[cuda12]==${JAX_VERSION}" ; \
    else \
        echo "unknown ACCEL=$ACCEL (use tpu or cuda12)" >&2 ; exit 1 ; \
    fi

ENTRYPOINT ["python", "/app/deploy/vertex/train_entrypoint.py"]
