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
- **Training (smoke, INTERRUPTED):** Gemma 3 270M + LoRA r=8, 1,826 train / 56 val examples, seq 320, batch 4, ~9 s/step on CPU. It was stopped at **step 50 of 400** when the laptop ran critically low on memory. My session was not allowed to restart a job stopped for memory pressure, so you need to rerun it (step 0 below). Real numbers that did come out (`results/train_smoke_partial.json`, `results/train_smoke_log.csv`):
  - base Gemma val perplexity **31.13** (exact, 56 val examples)
  - Keras val loss 2.098 after 50 LoRA steps; the running-mean train loss went from 3.19 (step 10) to 2.38 (step 50). These are Keras running means, not comparable to the perplexity above.
  - No adapter was saved, so **there are no benchmark metrics yet**. Nothing in the README or model card claims any.
- **Code, all pushed and tested (61 tests; ruff clean; CI green on GitHub):** data pipeline, flavor graph + normalizer + PMI + cuisine tagger, LoRA trainer (Keras 3/JAX, AdamW + warmup-cosine, prompt-masked loss, early stopping, CSV + TensorBoard), eval runner (perplexity, PMI/compound pairing validity, structure validity, bootstrap CIs, Gemini baseline, both-order pairwise LLM judge), report/README/model-card generators, Gradio demo + Space deploy script, blind A/B annotation tool + scorer, guarded Vertex AI scripts, Kaggle TPU / Colab / analysis notebooks.

## Pending, with exact commands (details in `docs/BLOCKERS.md`)

```bash
cd Gemma-Fine-Tuning && source .venv/Scripts/activate
# 0) Finish the smoke run in your own terminal, with other apps closed (it needs ~7 GB RAM).
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
> Built an end-to-end LoRA fine-tuning and evaluation pipeline for Gemma 3 270M (Keras 3/JAX, keras_hub): cleaned and deduped a 30k-recipe corpus, built a flavor-compound graph (Ahn et al.) with PMI-based pairing targets, and wrote a benchmark harness (perplexity vs base Gemma; pairing/structure validity with bootstrap CIs vs base Gemma and zero-shot Gemini; order-debiased LLM judge; blind human A/B tool). A CPU smoke LoRA run reached 50/400 steps; the full benchmark is pending.

The original line ("benchmarked vs. zero-shot Gemini on perplexity, pairing validity and human preference", "Vertex AI") is **not yet true**: no benchmark has run, Gemini perplexity is impossible by design (no API log-probs), there are no human annotations, and Vertex AI was not executed. After the steps above, update the bullet with numbers quoted from the *same* file, e.g. base vs LoRA from `results/perplexity.json` (`by_model.base` / `by_model.finetuned`). Don't mix them with the 31.1 val figure, which uses a different split and sequence length. List Vertex AI only as "deployment scripts" unless you run them.

## Interview talking points

1. I made a gated, GPU-hungry stack (keras_hub + Gemma) run on a Windows CPU laptop: eager SentencePiece tokenization instead of tf-text, bypassing keras_hub's tf.data `fit`, and a prompt-masked loss. CI tests the same code paths with a tiny random Keras LM, with no secrets.
2. The evaluation is designed to be honest: train-only PMI, held-out anchors, bootstrap CIs, a judge run in both orders that ties on disagreement to cancel position bias, and documented target/metric coupling and judge self-preference. Every reported number traces to a file in `results/`.
