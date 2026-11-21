---
license: gemma
base_model: google/gemma-3-270m-it
library_name: keras-hub
tags: [gemma, gemma3, lora, keras, jax, recipes, flavor-pairing]
---

# gemma-recipe-lora (model card: pending)

**No adapter has been released yet.** The CPU smoke fine-tune (Gemma 3 270M + LoRA r=8) was interrupted
at step 50/400 by host memory pressure, before an adapter was saved. See
[`results/train_smoke_partial.json`](results/train_smoke_partial.json) and [docs/BLOCKERS.md](docs/BLOCKERS.md).

This file is generated from real training and evaluation results by
`python scripts/make_model_card.py --config configs/train_smoke.yaml --eval-config configs/eval.yaml`
once `python -m recipe_gemma.train --config configs/train_smoke.yaml` and `make eval` have completed.

License: [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (adapter is a Gemma derivative).
Data: RecipeNLG-derived corpus (non-commercial), Ahn et al. 2011 flavor network (CC BY-NC-SA 3.0).
