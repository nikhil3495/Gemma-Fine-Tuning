# Build environment (recon, 2026-11-21)

The machine this repo was built and smoke-trained on. Credentials are recorded as **present/absent only**.

| Item | Value |
|---|---|
| OS | Windows 11 Home (10.0.26300), x86_64; WSL2 Ubuntu available (2 vCPU / 6 GB, not used) |
| CPU | 12th Gen Intel Core i5-12500H (12 cores / 16 threads) |
| RAM | 15.6 GB |
| GPU | Intel Iris Xe (integrated). **No NVIDIA GPU / CUDA**: `nvidia-smi` not found |
| Disk free | ~81 GB on C: |
| Python | 3.12.14 (uv-managed venv); JAX 0.12.1 (CPU), Keras 3.15.1, keras-hub 0.33.0, tensorflow-cpu 2.21.0 (preprocessing only) |
| `make` | absent on Windows (Makefile targets are used in CI/Linux; every target has a `python -m ...` equivalent) |

## Credentials and services

| Credential / service | Status |
|---|---|
| `HF_TOKEN` | **absent** (checked process, Windows User/Machine env, WSL, `~/.cache/huggingface/token`) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` / `kaggle.json` | **absent** |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | **absent** |
| `ALLOW_GCP_SPEND` | absent |
| `gcloud auth list` | gcloud not on PATH in this session (not authenticated): Vertex AI path not run |
| `gh auth status` | logged in (account `nikhil3495`, scopes include `repo`, `workflow`) |

## Model access checks

| Path | Result |
|---|---|
| keras_hub Kaggle preset `gemma3_instruct_270m` | not attempted to download: needs Kaggle creds (absent) |
| HF `google/gemma-3-270m-it` | gated (`manual`), needs HF_TOKEN plus license acceptance (absent) |
| HF `unsloth/gemma-3-270m-it` via `keras_hub ... from_preset("hf://unsloth/gemma-3-270m-it")` | **works without credentials** (ungated redistribution under the Gemma license). Loaded 268.1M params in 12 s on CPU |

## Measured CPU throughput (Gemma 3 270M IT, LoRA r=8, JAX CPU)

- LoRA trainable params: 534,528 (0.2% of 268M).
- Train step, batch 4 x seq 256: ~4-5 s per step after compilation.
- `generate()` batch 4, max_length 256: ~17 s.

Library versions were looked up on PyPI on 2026-11-21: keras 3.15.1, keras-hub 0.33.0, jax 0.12.1, gradio 6.32.0, google-genai 3.6.1.
