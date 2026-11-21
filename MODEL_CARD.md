---
license: gemma
base_model: google/gemma-3-270m-it
library_name: keras-hub
tags: [gemma, gemma3, lora, keras, jax, recipes, food, flavor-pairing]
datasets: [corbt/all-recipes]
---

# gemma-recipe-lora: LoRA adapter for recipe & flavor-pairing generation

LoRA adapter (`adapter.lora.h5`, keras_hub format) for **google/gemma-3-270m-it** (`hf://unsloth/gemma-3-270m-it`), fine-tuned with Keras 3 + JAX
on a recipe corpus for two tasks: (A) recipe generation from ingredients + cuisine and (B) flavor-pairing
suggestions with reasons. Code, data pipeline and benchmark: https://github.com/nikhil3495/Gemma-Fine-Tuning

> **Status: SMOKE run.** A short CPU fine-tune (proof of the end-to-end pipeline with real numbers), not a converged model. A larger config (`configs/train_full.yaml`, Gemma 3 1B) needs a GPU. Local, offline artifact: not published to any model hub. Weights were obtained from `hf://unsloth/gemma-3-270m-it` (an ungated copy of the base model above).

## Usage

```python
import os

os.environ["KERAS_BACKEND"] = "jax"
import keras_hub

lm = keras_hub.models.Gemma3CausalLM.from_preset("hf://unsloth/gemma-3-270m-it")
lm.backbone.load_lora_weights("adapter.lora.h5")  # rank read from the file
prompt = (
    "<start_of_turn>user\nSuggest 5 ingredients that pair well with basil, with a one-line "
    "reason each.<end_of_turn>\n<start_of_turn>model\n"
)
print(lm.generate(prompt, max_length=256))
```

## Training (all values from `results/train_smoke.json`)

| | |
|---|---|
| Base model | `hf://unsloth/gemma-3-270m-it` (268,632,704 params) |
| Method | LoRA rank 8 on attention query/value projections; 534,528 trainable params (0.20%) |
| Data | 1,396 train examples (recipe: 996, pairing: 400); 48 val |
| Optimizer | AdamW (lr 0.0002, wd 0.01, no decay on norms/biases), warmup 20 + cosine, grad-clip 1.0, micro-batch 1 x grad-accum 4 (effective 4), seq len 256 |
| Steps | 400 (best val loss at step 400) |
| Hardware | Windows-11-10.0.26300-SP0 (CPU only), 21.2 s/optimizer step, 141 min training, peak RSS 3.38 GB |
| Val perplexity | **30.80 -> 4.21** (base -> LoRA, 48 held-out val examples; the same set selected the best checkpoint) |
| Provenance | git `5c3df23-dirty`, config hash `f8e68ee8279a`, 2026-11-21T17:25:34+00:00 |

## Evaluation

**Benchmark (base vs fine-tuned generations, perplexity, pairing/structure validity) not yet run.** Run one model at a time: `python -m recipe_gemma.eval --config configs/eval.yaml --models base`, then `--models finetuned`, then `--stages metrics,judge`, then this report.

Protocol: [docs/EVAL_PROTOCOL.md](https://github.com/nikhil3495/Gemma-Fine-Tuning/blob/main/docs/EVAL_PROTOCOL.md).

## Intended use and limitations

- Research / portfolio demonstration of parameter-efficient fine-tuning. **Not** for commercial use
  (training data is RecipeNLG-derived, non-commercial terms; flavor data is CC BY-NC-SA 3.0).
- Not food-safety advice: generated recipes can be wrong about quantities, temperatures, allergens or
  doneness. Always apply common sense.
- 270M parameters and a short CPU run: outputs are often generic and sometimes incoherent.
- The corpus is US-centric, so non-American cuisines are under-represented.
- Pairing "reasons" imitate templated statistics (co-occurrence, NPMI, shared compounds); the numbers a
  model writes in its reasons are generated text, not looked up.

## License

Use of this adapter is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
Gemma Prohibited Use Policy, as it is a derivative of Gemma. Flavor-network data: Ahn et al. (2011),
CC BY-NC-SA 3.0.
