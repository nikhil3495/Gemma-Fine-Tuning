# Blockers (need the repo owner)

Everything else is built, tested and run. Each item below is something only you can provide. In each case the code path exists and degrades gracefully.

| # | Blocker | Impact now | What runs once fixed |
|---|---|---|---|
| 1 | **`HF_TOKEN` not available to the build session** (the `export HF_TOKEN=...` in the launch shell never reached it; checked process env, Windows User/Machine env, WSL and `~/.cache/huggingface/token`) | Adapter not pushed to `nikhil3495/gemma-recipe-lora`; **HF Space `nikhil3495/gemma-recipe-pairing` not deployed** | `scripts/push_adapter.py`, `scripts/deploy_space.py` (waits for the build and checks HTTP 200) |
| 2 | **`GEMINI_API_KEY` absent** | Zero-shot Gemini column and LLM-judge preference show "not run (no API key)". Both are implemented and tested with a mocked client | `python -m recipe_gemma.eval` generates only the missing Gemini outputs (base/fine-tuned are cached), then the judge runs |
| 3 | **No NVIDIA GPU / TPU locally** (Intel Iris Xe) and **no Kaggle credentials** | Only the CPU *smoke* run (Gemma 3 270M, 400 steps) was possible; the full run (Gemma 3 1B, `configs/train_full.yaml`) was not | `notebooks/train_kaggle_tpu.ipynb` or `notebooks/train_colab.ipynb` |
| 4 | **No human annotators** | Human-preference table: "pending annotation" | `python -m annotation.app` with 2+ people, then `python scripts/score_human_prefs.py` |
| 5 | **gcloud not authenticated / `ALLOW_GCP_SPEND` unset** | Vertex AI scripts written and guarded, not executed; no GCP resources were created | `deploy/vertex/README.md` |

## Exact commands (Git Bash; on Linux/macOS you can use the `make` targets)

```bash
cd Gemma-Fine-Tuning && source .venv/Scripts/activate   # Linux/macOS: .venv/bin/activate
export HF_TOKEN=hf_...  GEMINI_API_KEY=...               # never commit these

# 1) Gemini baseline + LLM judge (reuses cached Gemma generations), then report, model card, README table
python -m recipe_gemma.eval --config configs/eval.yaml
python -m recipe_gemma.eval.report --config configs/eval.yaml
python scripts/make_model_card.py --config configs/train_smoke.yaml --eval-config configs/eval.yaml

# 2) Publish the adapter and deploy the demo Space (verifies HTTP 200)
python scripts/push_adapter.py --config configs/train_smoke.yaml
python scripts/deploy_space.py
#    then replace the "not deployed yet" Space/model links at the top of README.md

# 3) Full run on Kaggle TPU / Colab GPU (accept the Gemma license on Kaggle first)
#    open notebooks/train_kaggle_tpu.ipynb, or locally with a GPU:
make data && make train CONFIG=full && make eval EVAL_CONFIG=configs/eval_full.yaml \
  && make push-model CONFIG=full && make deploy-space

# 4) Human preference
python -m annotation.app --generations-dir results/generations   # share the URL with annotators
python scripts/score_human_prefs.py

# 5) Optional Vertex AI (billable!)
export ALLOW_GCP_SPEND=1 GCP_PROJECT=... GCP_REGION=us-central1 && gcloud auth login
bash deploy/vertex/submit_training.sh && bash deploy/vertex/deploy_endpoint.sh
```
