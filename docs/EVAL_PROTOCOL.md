# Evaluation protocol

Code: `src/recipe_gemma/eval/`. Run: `python -m recipe_gemma.eval --config configs/eval.yaml && python -m recipe_gemma.eval.report --config configs/eval.yaml` (or `make eval`).

## Systems

| Key | System | Prompting |
|---|---|---|
| `base` | Gemma 3 270M IT, no adapter (zero-shot) | Gemma chat template |
| `finetuned` | Same model + our LoRA adapter | Gemma chat template |
| `gemini` | Gemini (`gemini-3.8-flash`, configurable) zero-shot via the API | Same user instruction text, no Gemma markup |

## Prompts

A fixed set of held-out test prompts (`data/artifacts/eval_prompts.jsonl`), ordered by a seeded hash: the first `n_per_task` of each task. Smoke uses 25 recipe + 25 pairing = **50**. Full uses 100 recipe + 72 pairing = **172**, because only 72 held-out anchors have >= 5 reference partners. Test recipes and test anchors never appear in training data or in the PMI statistics.

## Decoding (identical for all systems)

Top-k sampling, k = 40, temperature 0.7, fixed seed 1234. Max length is 576 total tokens for recipes and 256 for pairings (Gemma counts prompt + output). Gemini gets 512 / 200 output tokens with thinking disabled where the SDK allows. Generations are cached in `results/generations/{model}.jsonl` and never regenerated if present.

## Metrics

1. **Perplexity** of the held-out *reference* responses given their prompts, for `base` and `finetuned`: token-level `exp(sum NLL / sum tokens)` over response tokens only (prompt masked), seq_len 576. The 95% CI is an item-level bootstrap (2,000 resamples). **Gemini perplexity is not reported**: the API does not expose log-probabilities for scoring arbitrary reference text, and its tokenizer differs, so no comparable number exists.
2. **Pairing validity.** Model outputs are parsed and ingredients canonicalized with the same normalizer as the data pipeline. Pairs evaluated: Task B uses anchor x each suggestion (the anchor itself is excluded); Task A uses all unordered pairs of distinct generated ingredients. For each pair:
   - `pmi_positive`: training-corpus PMI > 0. Unseen ingredients count as *not* positive.
   - `shares_compound`: >= 1 shared flavor compound (Ahn et al. graph). Missing ingredients count as not sharing.
   - `graph_covered`: both ingredients are graph nodes. `shares_compound_covered` restricts to covered pairs.
   The item score is the mean over its pairs; we report the mean over items with a 95% bootstrap CI.
3. **Structure validity (Task A).** `well_formed`: parses into title + >= 1 ingredient + >= 1 step (the parser tolerates markdown headings, bold, bullets and numbered lists). `all_requested` / `requested_recall`: the requested ingredients appear in the ingredient list. `unrecognized_rate`: the share of generated ingredient lines whose canonical name is in neither the flavor graph nor the training-corpus vocabulary, a **proxy** for hallucinated / non-food ingredients (it also catches unusual but real ingredients, so read it comparatively). **Task B format:** `format_ok` = exactly 5 parseable suggestions; `anchor_repeated` = the anchor suggested as its own partner.
4. **Preference.**
   - **LLM-judge preference (proxy for human preference).** Pairwise and blind ("Response A/B", no model names), with a fixed rubric: coherence, feasibility, flavor harmony, instruction-following, and overall. Every item is judged in **both** orders; which order comes first is a seeded random choice per item. If the two orders disagree, the item is a tie, which cancels pure position bias (a unit test checks this). We report win/tie/loss with 95% bootstrap CIs per model pair. **This is never reported as human preference.**
   - **Human preference.** `annotation/app.py` shows anonymized, per-annotator randomized pairs and logs to CSV; `scripts/score_human_prefs.py` computes win rates with CIs and inter-annotator agreement (Cohen's / Fleiss' kappa). The table stays **"pending annotation"** until real people annotate.

## Known biases and threats to validity

- **Target/metric coupling.** Task-B training targets are ranked by the same training-split PMI that `pmi_positive` measures, so the fine-tuned model is optimized toward that metric. `shares_compound` is only partly independent (the ranking includes a small compound bonus). Treat the Task-B PMI gain as "learned the corpus's pairing statistics", not as proof of better taste.
- The fine-tuned model also learns our output format, which helps `well_formed` and `format_ok` relative to zero-shot systems. That is intended (instruction adherence), but it is not culinary quality.
- **Judge self-preference.** The default judge is Gemini, which is also a competitor in the `finetuned|gemini` and `gemini|base` pairs; LLMs tend to favour their own outputs. Treat those two pairs with extra caution, or set `judge.model` to a different model family.
- Smoke-scale evaluation (n = 25 per task) gives wide CIs; compare intervals, not point estimates.
- Normalizer and parser errors affect all systems, but not necessarily equally (e.g., markdown-heavy outputs).
