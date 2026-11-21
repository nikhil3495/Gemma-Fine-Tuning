# Handoff

## Links

| What | Status |
|---|---|
| Repo | https://github.com/nikhil3495/Gemma-Fine-Tuning (public, CI green) |
| HF Space `nikhil3495/gemma-recipe-pairing` | **not deployed**: `HF_TOKEN` was not available to the session (`scripts/deploy_space.py` is ready) |
| HF model `nikhil3495/gemma-recipe-lora` | **not published**: no `HF_TOKEN`, and no adapter yet (see below) |
| GitHub release v1.0.0 | **not created on purpose**: the benchmark it would advertise has not run yet |

## What ran

- **Recon:** CPU-only Windows laptop (i5-12500H, 15.6 GB RAM, Intel Iris Xe); no HF/Kaggle/Gemini credentials. A real Gemma path still exists through the ungated mirror `hf://unsloth/gemma-3-270m-it` (Gemma Terms of Use) loaded by keras_hub. Details: `docs/ENVIRONMENT.md`.
- **Data (complete, real):** 150k corpus rows scanned -> 131,616 valid -> 61,925 after dedupe -> 30,000 sampled -> 26,990 / 1,501 / 1,509 split. Flavor graph: 1,509 ingredients, 221,777 edges. Task B: 551 anchors. Eval set: 100 recipe + 72 pairing held-out prompts. Source: `results/data_stats.json`.
- **Training (smoke, INTERRUPTED):** Gemma 3 270M + LoRA r=8 (534,528 trainable params = 0.2%), 1,826 train / 56 val examples, seq 320, batch 4, ~9 s/step on CPU. The host killed it at **step 50 of 400** because the laptop ran critically low on memory. The agent harness forbids auto-restarting a job it reaped, so it was not restarted. Real numbers that did come out (`results/train_smoke_partial.json`, `results/train_smoke_log.csv`):
  - base Gemma val perplexity **31.13** (exact, 56 val examples)
  - val loss **2.098** after 50 LoRA steps (train-window loss fell 3.19 -> 2.38)
  - No adapter was saved, so **there are no benchmark metrics yet**. Nothing in the README or model card claims any.
- **Code, all pushed and tested (57 tests; ruff clean; CI green on GitHub):** data pipeline, flavor graph + normalizer + PMI + cuisine tagger, LoRA trainer (Keras 3/JAX, AdamW + warmup-cosine, prompt-masked loss, early stopping, CSV + TensorBoard), eval runner (perplexity, PMI/compound pairing validity, structure validity, bootstrap CIs, Gemini baseline, both-order pairwise LLM judge), report/README/model-card generators, Gradio demo + Space deploy script, blind A/B annotation tool + scorer, guarded Vertex AI scripts, Kaggle TPU / Colab / analysis notebooks.

## Pending, with exact commands (details in `docs/BLOCKERS.md`)

```bash
cd C:/Users/nikhil/Gemma-Fine-Tuning && source .venv/Scripts/activate
# 0) Finish the smoke run in YOUR terminal (not subject to the agent's memory reaper); close other apps first.
python -m recipe_gemma.train --config configs/train_smoke.yaml                      # ~60 min
export GEMINI_API_KEY=...   # optional, enables the Gemini column + LLM judge
python -m recipe_gemma.eval --config configs/eval.yaml                               # ~40 min CPU
python -m recipe_gemma.eval.report --config configs/eval.yaml                        # report.md + README table
python scripts/make_model_card.py --config configs/train_smoke.yaml --eval-config configs/eval.yaml
# 1) Publish + deploy
export HF_TOKEN=hf_...
python scripts/push_adapter.py --config configs/train_smoke.yaml && python scripts/deploy_space.py
# 2) Full run: notebooks/train_kaggle_tpu.ipynb (Gemma 3 1B, LoRA r=16), then eval with configs/eval_full.yaml
# 3) Humans: python -m annotation.app ; python scripts/score_human_prefs.py
# 4) Release once numbers exist
git add -A && git commit -m "feat(results): smoke benchmark" && git push
git tag v1.0.0 && git push --tags && gh release create v1.0.0 results/report.md --title "v1.0.0" --notes-file results/report.md
```

## Truthful resume bullet (as of now)

> **Gemma Fine-Tuning for Recipe & Flavor-Pairing Generation** | Python, JAX/Keras 3, keras_hub, Gemma 3, LoRA
> Built an end-to-end LoRA fine-tuning and evaluation pipeline for Gemma 3 on a 30k-recipe corpus: data cleaning and dedupe, a flavor-compound graph (Ahn et al.) with PMI-grounded pairing targets, Keras 3/JAX LoRA training (0.2% trainable params), and a benchmark harness against base Gemma and zero-shot Gemini (perplexity, pairing validity with bootstrap CIs, position-debiased LLM-judge proxy, and a blind human A/B tool), with CI and guarded Vertex AI deployment scripts.

After step 0 completes, add the measured result, e.g. "...cut held-out perplexity from 31.1 to X". Mention Vertex AI only as "deployment scripts" unless you actually run them.

## Interview talking points

1. I made a gated, GPU-hungry stack (keras_hub + Gemma) run on a Windows CPU laptop: eager SentencePiece tokenization instead of tf-text, bypassing keras_hub's tf.data `fit`, and a prompt-masked loss. CI tests the same code paths with a tiny random Keras LM, with no secrets.
2. The evaluation is designed to be honest: train-only PMI, held-out anchors, bootstrap CIs, a judge run in both orders that ties on disagreement to cancel position bias, and documented target/metric coupling. Nothing is reported that a script didn't produce.
