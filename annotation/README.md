# Blind human-preference annotation

A small Gradio tool for blind pairwise (A/B) comparison of recipe and flavor-pairing outputs
from three systems: zero-shot Gemma (`base`), LoRA fine-tuned Gemma (`finetuned`) and zero-shot
Gemini (`gemini`). Annotators never see which model wrote which response.

> **Status:** `results/human_preference.json` stays `"status": "pending annotation"` until real
> people have used this tool. LLM-judge scores elsewhere in this repo are a **proxy** for human
> judgement, not a replacement for it.

## For annotators

**Time:** aim for one 30-45 minute session. You can stop at any time; when you come back,
enter the **same annotator id** and the tool resumes where you left off.

For each comparison you see an instruction (a recipe request or a flavor-pairing question)
and two anonymous answers, **Response A** and **Response B**. Left/right placement is random,
so don't read anything into position.

1. Read the instruction, then both responses in full.
2. Pick an **overall preference**: *A better*, *Tie*, or *B better*. This is the only required
   field.
3. Optionally rate each response 1-5 on the four criteria below (1 = poor, 3 = acceptable,
   5 = excellent).
4. Optionally leave a short comment (e.g. "B burns the garlic in step 2").
5. Press **Submit & next**.

### Rubric

| Criterion | What it means | Low score (1-2) | High score (4-5) |
|---|---|---|---|
| **Coherence** | The answer reads as one consistent, logically ordered piece. | Contradictions, repeated or missing steps, ingredients used but never listed, text that trails off. | Ingredients and steps line up; order makes sense; nothing dangling. |
| **Feasibility** | A home cook could actually make this and it would work. | Impossible quantities/times/temperatures, unsafe practices (e.g. undercooked chicken), unobtainable equipment. | Realistic quantities, times and techniques; safe; would plausibly turn out well. |
| **Flavor harmony** | The ingredients and suggested pairings taste good together. | Clashing or random combinations, no justification for pairings. | Balanced (salt/acid/fat/sweet/bitter), sensible pairings, ideally with a reason why they work. |
| **Instruction-following** | It does what the user asked. | Ignores requested ingredients, cuisine, diet or format; answers a different question. | Uses all requested constraints (ingredients, diet, count, format) and nothing contradictory. |

### Guidance

- **Judge the food, not the formatting or length.** A longer answer with more bullet points is
  not better unless the extra content makes the dish or pairing better. Do not reward padding.
- **Ties** are fine when the two responses are genuinely equally good (or equally bad), or when
  each has a different flaw of similar weight. Don't use *Tie* just to avoid a hard call: if you
  would rather cook from one of them, pick it.
- Serious problems outweigh polish: an unsafe or unworkable recipe should lose to a plain but
  correct one.
- Don't try to guess which model wrote what, and don't look anything up beyond basic cooking
  facts you'd normally check.

## For the experimenter

### Launch

From the repo root (Windows: use `.venv\Scripts\python.exe` in place of `python`):

```bash
pip install -e ".[app]"        # gradio
python -m annotation.app --generations-dir results/generations
# options: --out annotation/data/annotations.csv  --port 7861  --share
```

Inputs are `results/generations/{base,finetuned,gemini}.jsonl` (fields `id`, `task`, `user`,
`output`, `model`). Missing files are skipped, and so are the pairs that need them (e.g. without
`gemini.jsonl` only `finetuned|base` is shown). A local demo uses the test fixtures:

```bash
python -m annotation.app --generations-dir tests/fixtures/generations --out annotation/data/demo.csv
```

Each judgement is appended to `annotation/data/annotations.csv`, including the hidden model
identities (`model_a`, `model_b`, `winner_model`), which are used for scoring only. For
inter-annotator agreement, have at least two people label the same items (everyone sees the
full item set, in a different order).

### Score

```bash
python scripts/score_human_prefs.py
# --annotations annotation/data/annotations.csv --out-json results/human_preference.json
# --out-csv results/human_preference.csv --n-boot 2000
```

This writes win/tie/loss rates per pair, from the first model's perspective (e.g.
`finetuned|base` = how often finetuned beats base), with 95% bootstrap CIs, overall and by task.
It also writes inter-annotator agreement over items labeled by 2 or more people: the mean
pairwise Cohen's kappa, plus Fleiss' kappa when every such item has the same number of raters.
Without annotations it writes a `"pending annotation"` stub with the launch command and exits 0.
