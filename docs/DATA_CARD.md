# Data card

All counts below come from `results/data_stats.json` (written by `python -m recipe_gemma.data.build`).

## Sources and licenses

| Source | What we use | License / terms | Redistributed here? |
|---|---|---|---|
| [`corbt/all-recipes`](https://huggingface.co/datasets/corbt/all-recipes) (HF, ungated parquet; 2.15M recipes) | Shard `train-00000-of-00004` (536,812 rows); 150k scanned, 30k sampled | The card states no license. Text and rows match **RecipeNLG** (Bień et al., INLG 2020), whose terms allow **non-commercial research and educational use only**. We treat it under those terms. | No raw data. Only held-out prompts/references (`data/artifacts/eval_prompts.jsonl`), plus model generations once the benchmark runs |
| Ahn, Ahnert, Bagrow & Barabási (2011), *Flavor network and the principles of food pairing*, Sci. Rep. 1:196, doi:10.1038/srep00196: supplementary `srep00196-s2.csv` (ingredient pairs + #shared flavor compounds) and `srep00196-s3.csv` (56,498 cuisine-labelled recipes) | Flavor graph (1,509 ingredient nodes, 221,777 weighted edges), cuisine tagger | **CC BY-NC-SA 3.0** | Yes: compiled graph `data/artifacts/flavor_graph.json.gz` (attribution here; same license applies to that file) |
| Gemma 3 weights (`unsloth/gemma-3-270m-it`, an ungated mirror of `google/gemma-3-270m-it`) | Base model | **Gemma Terms of Use** + Prohibited Use Policy | No base weights. Only the LoRA adapter (a derivative, distributed under the same terms) |

Consequence: this project, its adapter and its demo are **non-commercial research/portfolio artifacts**.

## Pipeline (`src/recipe_gemma/data/`)

1. **Parse** `Title / Ingredients: / Directions:` text into fields.
2. **Clean**: HTML-unescape and strip tags, normalize whitespace, normalize units (`c.` -> `cup`, `Tbsp.` -> `tbsp`, `pkg.` -> `package`, ...), and re-join direction fragments split mid-sentence (a known corpus artifact).
3. **Filter**: >= 3 ingredients, >= 2 steps, <= 2,000 characters.
4. **Dedupe**: exact (SHA-1 of the normalized full text) **and** normalized-title (lowercase alphanumerics). This is aggressive by design: 131,616 -> 61,925.
5. **Sample** 30,000 in a fixed-seed shuffled order (seed 1234).
6. **Canonicalize ingredients** (`flavor/normalize.py`): strip quantities, units, parentheticals and descriptors; singularize; apply the alias table; longest n-gram match against flavor-graph node names. 71% of ingredient lines map to a graph node. The rest are mostly ingredients absent from the Ahn data (sugar, salt, water, baking powder, ...), which still count for corpus PMI.
7. **Cuisine label**: the corpus has none, so a naive-Bayes tagger trained on Ahn's 56k cuisine-labelled recipes assigns one when posterior >= 0.6 with >= 3 known ingredients, else `any`. 61% North American and 28% `any`: the corpus is US-centric, a known limitation.
8. **Split** recipes 90/5/5 by a seeded hash of the recipe id (26,990 / 1,501 / 1,509). The test split is never used for training or for building PMI statistics.

## Tasks and targets

**Task A: recipe generation.** Prompt: `Create a recipe using: {2-4 key ingredients}. Cuisine: {cuisine}` plus a format hint. Key ingredients are the recipe's canonical non-staple ingredients (staples such as salt, water, sugar, butter, flour, eggs and oil are excluded), sampled with a fixed seed. Target: `Title / Ingredients (original quantities) / Steps (numbered) / Flavor notes`. **Flavor notes are templated**, not human-written: the top-2 ingredient pairs in the recipe by shared flavor compounds (non-staple pairs first), e.g. *"cocoa and pecan share 33 flavor compounds"* (from the first held-out prompt).

**Task B: flavor pairing.** Prompt: `Suggest 5 ingredients that pair well with {ingredient}, with a one-line reason each.` Targets are built **only from training-split statistics**:
- Anchors: ingredients appearing in >= 10 training recipes (551), split 80/5/15 into train/val/test anchors (test anchors never appear as Task-B training prompts).
- Partners: co-occur >= 3 times with NPMI > 0, ranked by `NPMI + 0.1*log(1 + shared_compounds)` (`flavor/scorer.py`), top 5.
- Reason line (templated from the evidence): `co-occurs in N recipes (NPMI x.xx) and shares K flavor compounds.`

## Sizes

| Split | Recipe examples | Pairing examples |
|---|---|---|
| train (full profile) | 22,827 unique (+2,536 pairing rows oversampled from 449 anchors to 10%) | 449 anchors |
| train (smoke profile) | 1,600 | 400 (subsampled: 400 of 449 anchors, seeded shuffle) |
| val | 1,408 | 16 |
| test (held-out eval prompts) | 100 used (of 1,435) | 72 (all test anchors with >= 5 partners) |

The smoke profile then drops examples longer than 320 tokens (174 of 2,000). See `results/train_smoke_partial.json`.

## Known issues

- RecipeNLG-style text has typos, US units and brand names (e.g. "Bisquick", "Cool Whip").
- PMI from 27k recipes is noisy for rare ingredients; the anchor/partner count thresholds mitigate this.
- Task-B targets are derived from the same PMI statistics used by the pairing-validity metric. This favours the fine-tuned model on `pmi_positive`; the compound-overlap metric is a partially independent check. See docs/EVAL_PROTOCOL.md.
