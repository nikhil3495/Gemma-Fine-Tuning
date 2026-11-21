# Handoff

**Repo:** https://github.com/nikhil3495/Gemma-Fine-Tuning (public, CI green). Offline, local-only project: nothing is deployed and no cloud service is used.

## Done (with real, committed results)

- **Data:** 150k rows scanned -> 131,616 valid -> 61,925 after dedupe -> 30,000 sampled -> 26,990 / 1,501 / 1,509 split. Flavor graph: 1,509 ingredients, 221,777 edges. Source: `results/data_stats.json`. A fresh-clone rebuild reproduced `eval_prompts.jsonl` byte-for-byte, and the flavor graph and co-occurrence stats with identical decompressed content (manual check).
- **LoRA training (CPU, memory-safe, resumable):** Gemma 3 270M IT + LoRA rank 8 (534,528 trainable params, 0.20%). 1,396 train / 48 val examples after the seq-256 length filter (604 / 36 dropped). Micro-batch 1 x grad-accum 4. 400 steps; the run was stopped by host memory pressure at about step 230, resumed from the step-200 checkpoint, and completed. Peak RSS 3.38 GB. Source: `results/train_smoke.json`, `results/train_smoke_log.csv`, raw logs `results/smoke_train_*.log`.
  - **Validation perplexity: 30.80 (base) -> 4.21 (LoRA)** on 48 held-out validation examples (response tokens).
  - Validation loss: 1.994 (step 50) -> 1.4345 (step 400, best).
- Generated `results/report.md`, `results/plots/loss_curve.png`, `MODEL_CARD.md` and the README results block from those files.
- Code: gradient accumulation, checkpoint every 50 steps with `--resume`, `--dry-run` memory check, per-model eval (`--models`) with cached perplexity, single-model live demo. 63 tests, ruff clean, CI green.

## Not done

- **Benchmark (base vs fine-tuned on 25 + 25 held-out test prompts: generations, test perplexity, pairing validity, structure validity).** The base-model eval was stopped by host memory pressure about 40 s after launch, before anything was cached, and was not restarted automatically. Run it in your own terminal with browsers closed (~2.5 GB RAM, ~1 h):

```bash
cd Gemma-Fine-Tuning && source .venv/Scripts/activate
python -m recipe_gemma.eval --config configs/eval.yaml --models base
python -m recipe_gemma.eval --config configs/eval.yaml --models finetuned
python -m recipe_gemma.eval --config configs/eval.yaml --stages metrics,judge
python -m recipe_gemma.eval.report --config configs/eval.yaml
python scripts/make_model_card.py --config configs/train_smoke.yaml --eval-config configs/eval.yaml
bash scripts/secret_scan.sh && git add results MODEL_CARD.md README.md && git commit -m "feat(eval): local benchmark results" && git push
git tag v1.0.0 && git push --tags && gh release create v1.0.0 results/report.md --title v1.0.0 --notes-file results/report.md
```

- **No v1.0.0 tag or release yet:** your brief makes it conditional on the benchmark.
- **No demo screenshot:** the demo's recipe tab has nothing cached until the benchmark runs (`LIVE_MODEL=1 python app/app.py` works with the trained adapter).
- Optional or future: Gemini baseline and LLM judge (needs `GEMINI_API_KEY`), human annotation, a larger 1B run on a GPU. See `docs/FUTURE_WORK.md`.

## Resume bullet (matches the committed results)

> **Gemma Fine-Tuning for Recipe & Flavor-Pairing Generation** | Python, JAX/Keras 3, keras_hub, Gemma 3, LoRA
> LoRA fine-tuned Gemma 3 270M (0.2% trainable params) on 1,396 examples (996 recipes + 400 pairing prompts) sampled from a cleaned, deduplicated 30k-recipe corpus with PMI- and flavor-compound-grounded pairing targets, entirely on a laptop CPU (gradient accumulation, resumable checkpoints, 3.4 GB peak RAM). Cut validation perplexity on 48 held-out examples from 30.8 to 4.2, largely from learning the target format. Implemented and unit-tested (not yet run) an eval harness for base vs fine-tuned Gemma (pairing/structure validity with bootstrap CIs), with optional Gemini-baseline and position-debiased LLM-judge hooks.

Once the benchmark runs, replace the last clause with its numbers from `results/eval_metrics.json` and `results/perplexity.json`.

## Interview talking points

1. Made a keras_hub/Gemma LoRA pipeline train on a 16 GB Windows laptop: eager tokenization instead of tf-text, bypassing keras_hub's tf.data `fit`, gradient accumulation, and checkpoint/resume that recovered a run stopped by host memory pressure at step 230. Peak RSS was 3.24 GB in a 5-step dry run and 3.38 GB in the full run.
2. Honest evaluation design: train-only PMI, held-out anchors, bootstrap CIs, a judge that sees both orders to cancel position bias, and documented biases. Every number traces to a file in `results/`.
