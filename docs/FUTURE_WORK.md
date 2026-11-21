# Future work

The project runs end to end offline on a laptop CPU (data -> LoRA training -> benchmark -> local demo). These gaps remain; none of them blocks the committed results.

| Item | Why it isn't done | How to do it |
|---|---|---|
| **Human preference study** | Needs real annotators. The tool and scorer exist; results stay "pending annotation" | `python -m annotation.app --generations-dir results/generations` (2+ annotators, ~30-45 min each), then `python scripts/score_human_prefs.py` -> `results/human_preference.json` |
| **Larger model / longer training** (Gemma 3 1B, LoRA r=16, seq 512, full 25k-example mix) | Needs a GPU/TPU; a laptop CPU can do the 270M smoke run only | `python -m recipe_gemma.train --config configs/train_full.yaml` on a GPU machine, then eval with `configs/eval_full.yaml` |
| **Bigger eval set** (100 recipe + 72 pairing prompts) | CPU generation time; the smoke eval uses 25 + 25 | `python -m recipe_gemma.eval --config configs/eval_full.yaml` (with the full-run adapter) |
| **Zero-shot Gemini baseline + LLM-judge proxy** (optional) | Need a `GEMINI_API_KEY`; implemented and tested with a mocked client | `export GEMINI_API_KEY=... && python -m recipe_gemma.eval --config configs/eval.yaml --stages generate,metrics,judge && python -m recipe_gemma.eval.report --config configs/eval.yaml` |
| **Save optimizer state in checkpoints** | Adam moments restart from zero on `--resume` (adapter-only checkpoints) | Store `optimizer.variables` next to the adapter in `Checkpointer` |
| **Judge from a different model family** | Gemini as judge favours Gemini outputs (self-preference) | Set `judge.model` to a non-Gemini judge client |
