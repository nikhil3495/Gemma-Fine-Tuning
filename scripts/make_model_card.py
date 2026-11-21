"""Generate MODEL_CARD.md from results/train_*.json and results/eval_metrics.json.

python scripts/make_model_card.py --config configs/train_smoke.yaml --eval-config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recipe_gemma.eval.report import results_table  # noqa: E402
from recipe_gemma.utils.config import load_config  # noqa: E402

TEMPLATE = """---
license: gemma
base_model: {base_model}
library_name: keras-hub
tags: [gemma, gemma3, lora, keras, jax, recipes, food, flavor-pairing]
datasets: [corbt/all-recipes]
---

# gemma-recipe-lora: LoRA adapter for recipe & flavor-pairing generation

LoRA adapter (`{adapter_file}`, keras_hub format) for **{base_model}** (`{preset}`), fine-tuned with Keras 3 + JAX
on a recipe corpus for two tasks: (A) recipe generation from ingredients + cuisine and (B) flavor-pairing
suggestions with reasons. Code, data pipeline and benchmark: https://github.com/nikhil3495/Gemma-Fine-Tuning

> **Status: {status}.** {status_note}

## Usage

```python
import os; os.environ["KERAS_BACKEND"] = "jax"
import keras_hub
lm = keras_hub.models.Gemma3CausalLM.from_preset("{preset}")
lm.backbone.load_lora_weights("{adapter_file}")   # rank read from the file
prompt = ("<start_of_turn>user\\nSuggest 5 ingredients that pair well with basil, with a one-line "
          "reason each.<end_of_turn>\\n<start_of_turn>model\\n")
print(lm.generate(prompt, max_length=256))
```

## Training (all values from `{train_results}`)

| | |
|---|---|
| Base model | `{preset}` ({total_params:,} params) |
| Method | LoRA rank {rank} on attention query/value projections; {trainable:,} trainable params ({trainable_pct:.2f}%) |
| Data | {train_examples:,} train examples ({tasks}); {val_examples} val |
| Optimizer | AdamW (lr {lr}, wd {wd}, no decay on norms/biases), warmup {warmup} + cosine, grad-clip {clip}, batch {bs}, seq len {seq} |
| Steps | {steps} (best val loss at step {best_step}){early} |
| Hardware | {platform}, {sec_per_step} s/step, {minutes:.0f} min total |
| Val perplexity | **{ppl_base:.2f} -> {ppl_ft:.2f}** (base -> LoRA, {ppl_n} held-out val examples) |
| Provenance | git `{sha}`, config hash `{chash}`, {ts} |

## Evaluation

{eval_table}

Protocol: [docs/EVAL_PROTOCOL.md](https://github.com/nikhil3495/Gemma-Fine-Tuning/blob/main/docs/EVAL_PROTOCOL.md).

## Intended use and limitations

- Research / portfolio demonstration of parameter-efficient fine-tuning. **Not** for commercial use
  (training data is RecipeNLG-derived, non-commercial terms; flavor data is CC BY-NC-SA 3.0).
- Not food-safety advice: generated recipes can be wrong about quantities, temperatures, allergens or
  doneness. Always apply common sense.
- 270M parameters with a short CPU smoke run: outputs are often generic and sometimes incoherent.
- The corpus is US-centric, so non-American cuisines are under-represented.
- Pairing "reasons" imitate templated statistics (co-occurrence, NPMI, shared compounds); the numbers a
  model writes in its reasons are generated text, not looked up.

## License

Use of this adapter is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
Gemma Prohibited Use Policy, as it is a derivative of Gemma. Flavor-network data: Ahn et al. (2011),
CC BY-NC-SA 3.0.
"""


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train_smoke.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    ecfg = load_config(ROOT / args.eval_config)
    tr = json.loads((ROOT / cfg["out"]["results"]).read_text(encoding="utf-8"))
    t, m, d, c = tr["training"], tr["model"], tr["data"], tr["config"]["train"]
    smoke = tr["config"]["run_name"] == "smoke"
    text = TEMPLATE.format(
        adapter_file=Path(tr["adapter"]["path"]).name,
        status="SMOKE run" if smoke else "full run",
        status_note=(
            "A short CPU fine-tune (proof of the end-to-end pipeline with real numbers), not a "
            "converged model. The full-run config is `configs/train_full.yaml` (Gemma 3 1B, Kaggle TPU)."
            if smoke
            else ""
        ),
        preset=m["preset"],
        base_model={
            "hf://unsloth/gemma-3-270m-it": "google/gemma-3-270m-it",
            "gemma3_instruct_270m": "google/gemma-3-270m-it",
            "gemma3_instruct_1b": "google/gemma-3-1b-it",
        }.get(m["preset"], m["preset"]),
        total_params=m["total_params"],
        rank=m["lora_rank"],
        trainable=m["trainable_params"],
        trainable_pct=m["trainable_pct"],
        train_examples=d["train_examples"],
        tasks=", ".join(f"{k}: {v}" for k, v in d["train_tasks"].items()),
        val_examples=d["val_examples"],
        lr=c["lr"],
        wd=c["weight_decay"],
        warmup=c["warmup_steps"],
        clip=c["grad_clip"],
        bs=c["batch_size"],
        seq=c["seq_len"],
        steps=t["steps_run"],
        best_step=t["best_step"],
        early=" (early-stopped)" if t["early_stopped"] else "",
        platform=tr["meta"]["platform"],
        sec_per_step=t["sec_per_step"],
        minutes=t["train_seconds"] / 60,
        ppl_base=tr["val_perplexity"]["base"]["ppl"],
        ppl_ft=tr["val_perplexity"]["fine_tuned"]["ppl"],
        ppl_n=tr["val_perplexity"]["base"]["n"],
        sha=tr["meta"]["git_sha"],
        chash=tr["meta"]["config_hash"],
        ts=tr["meta"]["timestamp_utc"],
        train_results=cfg["out"]["results"],
        eval_table=results_table(ROOT / ecfg["out_dir"], ecfg),
    )
    (ROOT / "MODEL_CARD.md").write_text(text, encoding="utf-8")
    print("wrote MODEL_CARD.md")


if __name__ == "__main__":
    main()
