"""Minimal Vertex AI custom-container predictor: Gemma 3 preset + LoRA adapter.

Vertex contract (env vars set by Vertex; defaults allow `docker run` locally):
  AIP_HTTP_PORT       8080
  AIP_HEALTH_ROUTE    /health      -> 200 once the model is loaded, 503 while loading
  AIP_PREDICT_ROUTE   /predict     -> {"instances": [{"prompt": "..."}], "parameters": {...}}
  AIP_STORAGE_URI     gs://... copy of the model's --artifact-uri (holds adapter.lora.h5)

Our env:
  MODEL_PRESET        gemma3_instruct_1b   (keras_hub preset or hf://...)
  MODEL_DTYPE         bfloat16             ("" -> float32)
  ADAPTER_PATH        local adapter.lora.h5 (overrides AIP_STORAGE_URI; "none" = base model)
  MAX_LENGTH          512                  (prompt + generated tokens; per-request override)
  *_SECRET            Secret Manager refs, e.g. KAGGLE_USERNAME_SECRET / KAGGLE_KEY_SECRET

Response: {"predictions": [{"prompt": ..., "output": ...}, ...]}
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "jax")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # deploy/vertex (resolve_secrets)

log = logging.getLogger("predictor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
PORT = int(os.environ.get("AIP_HTTP_PORT", "8080"))

_state: dict[str, Any] = {"lm": None, "error": None}
_lock = threading.Lock()  # JAX generation is not run concurrently


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    threading.Thread(target=load_model, daemon=True).start()  # health = 503 until ready
    yield


app = FastAPI(title="recipe-gemma predictor", lifespan=_lifespan)


def _download_adapter(storage_uri: str) -> Path:
    """Fetch ``adapter.lora.h5`` from ``gs://bucket/prefix`` (searched recursively)."""
    from google.cloud import storage

    bucket_name, _, prefix = storage_uri.removeprefix("gs://").partition("/")
    client = storage.Client()
    blobs = [
        b
        for b in client.list_blobs(bucket_name, prefix=prefix.rstrip("/"))
        if b.name.endswith("adapter.lora.h5")
    ]
    if not blobs:
        raise FileNotFoundError(f"no adapter.lora.h5 under {storage_uri}")
    dest = Path(tempfile.mkdtemp()) / "adapter.lora.h5"
    blobs[0].download_to_filename(str(dest))
    log.info("downloaded gs://%s/%s", bucket_name, blobs[0].name)
    return dest


def _adapter_path() -> Path | None:
    explicit = os.environ.get("ADAPTER_PATH", "")
    if explicit.lower() == "none":
        return None
    if explicit:
        return Path(explicit)
    uri = os.environ.get("AIP_STORAGE_URI", "")
    if uri.startswith("gs://"):
        return _download_adapter(uri)
    if uri:  # Vertex may also hand a local path
        cands = sorted(Path(uri).rglob("adapter.lora.h5"))
        if cands:
            return cands[0]
    raise FileNotFoundError("set AIP_STORAGE_URI (gs://...) or ADAPTER_PATH (or ADAPTER_PATH=none)")


def load_model() -> None:
    """Load preset + adapter in the background; /health flips to 200 when done."""
    try:
        from resolve_secrets import resolve_secret_env

        resolve_secret_env()
        from recipe_gemma.train.lm import load_lm

        preset = os.environ.get("MODEL_PRESET", "gemma3_instruct_1b")
        dtype = os.environ.get("MODEL_DTYPE", "bfloat16") or None
        log.info("loading %s (dtype=%s)", preset, dtype)
        lm = load_lm({"preset": preset, "dtype": dtype})
        adapter = _adapter_path()
        if adapter is not None:
            lm.load_adapter(adapter)  # keras_hub enables LoRA with the rank stored in the file
            log.info("loaded LoRA adapter %s", adapter)
        lm.set_sampler("greedy", 1.0, 1, 1.0, 0)
        _state["lm"] = lm
        log.info("model ready")
    except Exception as exc:  # surfaced via /health
        log.exception("model load failed")
        _state["error"] = repr(exc)


@app.get(HEALTH_ROUTE)
def health() -> JSONResponse:
    if _state["lm"] is not None:
        return JSONResponse({"status": "ok"})
    body = {"status": "error" if _state["error"] else "loading", "error": _state["error"]}
    return JSONResponse(body, status_code=503)


@app.post(PREDICT_ROUTE)
async def predict(request: Request) -> dict[str, Any]:
    lm = _state["lm"]
    if lm is None:
        raise HTTPException(503, "model not loaded yet")
    body = await request.json()
    instances = body.get("instances")
    if not isinstance(instances, list) or not instances:
        raise HTTPException(400, 'expected {"instances": [{"prompt": "..."}]}')
    params = body.get("parameters") or {}
    max_length = int(params.get("max_length", os.environ.get("MAX_LENGTH", "512")))

    from recipe_gemma.data.prompts import START_USER, chat_prompt

    prompts = []
    for inst in instances:
        text = inst.get("prompt") if isinstance(inst, dict) else inst
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(400, "each instance needs a non-empty 'prompt' string")
        prompts.append(text if text.startswith(START_USER) else chat_prompt(text))

    with _lock:
        outputs = lm.generate(prompts, max_length=max_length)
    return {
        "predictions": [
            {"prompt": inst.get("prompt") if isinstance(inst, dict) else inst, "output": out}
            for inst, out in zip(instances, outputs, strict=True)
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
