# Future work

The project runs offline on a laptop CPU. Data and LoRA training are complete with committed results; the gaps below remain.

| Item | Why it isn't done | How to do it |
|---|---|---|
| **Run the local benchmark** (base vs fine-tuned on 25 + 25 held-out prompts: generations, test perplexity, pairing/structure validity) | The run was stopped by host memory pressure (the agent's background-job protection) during model load; per the brief it was not auto-restarted. The trained adapter is in `checkpoints/smoke/` | In a normal terminal with browsers closed (~2.5 GB RAM per model, ~1 h total on CPU): `python -m recipe_gemma.eval --config configs/eval.yaml --models base`, `python -m recipe_gemma.eval --config configs/eval.yaml --models finetuned`, `python -m recipe_gemma.eval --config configs/eval.yaml --stages metrics,judge`, `python -m recipe_gemma.eval.report --config configs/eval.yaml`, `python scripts/make_model_card.py`. Then `git tag v1.0.0 && gh release create v1.0.0 results/report.md` |
| **Human preference study** | Needs real annotators. The tool and scorer exist; results stay "pending annotation" | `python -m annotation.app --generations-dir results/generations` (2+ annotators, ~30-45 min each), then `python scripts/score_human_prefs.py` -> `results/human_preference.json` |
| **Larger model / longer training** (Gemma 3 1B, LoRA r=16, seq 512, full 25k-example mix) | Needs a GPU/TPU; a laptop CPU can do the 270M smoke run only | `python -m recipe_gemma.train --config configs/train_full.yaml` on a GPU machine, then eval with `configs/eval_full.yaml` |
| **Bigger eval set** (100 recipe + 72 pairing prompts) | CPU generation time; the smoke eval uses 25 + 25 | `python -m recipe_gemma.eval --config configs/eval_full.yaml` (with the full-run adapter) |
| **Zero-shot Gemini baseline + LLM-judge proxy** (optional) | Need a `GEMINI_API_KEY`; implemented and tested with a mocked client | `export GEMINI_API_KEY=... && python -m recipe_gemma.eval --config configs/eval.yaml --stages generate,metrics,judge && python -m recipe_gemma.eval.report --config configs/eval.yaml` |
| **Save optimizer state in checkpoints** | Adam moments restart from zero on `--resume` (adapter-only checkpoints) | Store `optimizer.variables` next to the adapter in `Checkpointer` |
| **Judge from a different model family** | Gemini as judge favours Gemini outputs (self-preference) | Set `judge.model` to a non-Gemini judge client |
