# Engineering decisions log

One line per decision: what was chosen and why. Newest at the bottom.

- **Compute plan = CPU smoke + notebooks for full run.** No NVIDIA GPU (Intel Iris Xe only), so: real LoRA smoke fine-tune locally on CPU + `notebooks/train_kaggle_tpu.ipynb` / `train_colab.ipynb` for the full run.
- **Base model = Gemma 3 270M instruction-tuned** (smallest Gemma 3 variant in keras_hub presets: `gemma3_instruct_270m`). Feasible on CPU (~9 s/step logged at seq 320); full config targets `gemma3_instruct_1b` on TPU/GPU.
- **Weights source = `hf://unsloth/gemma-3-270m-it`** because HF_TOKEN and Kaggle creds are absent and `google/*` Gemma repos are gated. The unsloth repo is an ungated redistribution of the same Gemma 3 weights under the Gemma Terms of Use. Config key `model.preset` switches to the official Kaggle preset when creds exist.
- **Tokenize eagerly in Python, fit on NumPy arrays.** keras_hub's tf.data path needs `tensorflow-text`, which has no Windows wheels; the pure-Python SentencePiece path works everywhere. It also enables loss masking on the prompt (we train on responses only).
- **Python 3.12** (jax 0.12 requires >=3.12; system Python is 3.11).
- **Gemini baseline model = `gemini-3.8-flash`** (newest GA Flash per ai.google.dev "What's new", 2026-11), configurable in `configs/eval.yaml`. Not run: no API key.
- **LLM judge = Gemini** (same key). Without a key the judge does not run, and the table says "not run (no API key)". No substitute judge was invented.
- **Recipe corpus = HF `corbt/all-recipes`** (2.15M recipes, ungated parquet, loads without a form). Its card has no license; the text format and rows match RecipeNLG (Bień et al. 2020), whose terms allow non-commercial research/education only. We use it under those terms and redistribute only small derived samples in `results/`.
- **Flavor graph = Ahn et al. 2011 supplementary data** (`srep00196-s2.csv`: ingredient pairs with shared-compound counts; `s3.csv`: 56k cuisine-labelled recipes), CC BY-NC-SA 3.0. The pair file gives shared-compound counts, not per-ingredient compound lists, so the "compound set" API is exposed as pairwise shared-compound counts (that is what the metric needs).
- **Step fragments re-joined + unit regex fix** after inspecting real examples: the corpus splits directions mid-sentence, so fragments are merged until terminal punctuation.
- **Only clear aliases onto flavor-graph names** (chocolate->cocoa, molasses->cane_molasses, sausage->pork_sausage, cornstarch->starch). Sugar, salt, water etc. are absent from Ahn's data, so they stay corpus-only and count as "not covered" rather than being force-mapped.
- **Task-B anchors: >=10 training recipes, partners >=3 co-occurrences, ranked by NPMI + 0.1*log1p(shared compounds)**: gives 551 anchors (458/19/74 train/val/test) vs 359 at stricter thresholds.
- **Anchor split 80/5/15** (not 90/5/5) so there are enough held-out pairing prompts (72 usable).
- **Pairing share fixed** at 20% (smoke: 400 of 449 anchors, seeded shuffle) / 10% (full: 2,536 rows, i.e. ~5.6 passes over 449 anchors); with no reweighting, 449 anchor examples vs 25k recipes would be ~2%.
- **Full eval = 100 recipe + 72 pairing = 172 prompts** (not 200): only 72 held-out anchors have >=5 reference partners. Smoke = 25 + 25 = 50.
- **Smoke seq_len 320 with over-length examples dropped** (keeps 89% of recipes, 100% of pairings) so the model always sees complete targets ending in `<end_of_turn>`; 400 steps, batch 4, LoRA r=8, lr 2e-4, float32 (CPU).
- **Decoding = top-k 40, T=0.7, seed 1234 for every system** (greedy loops badly on a 270M model; identical settings keep the comparison fair).
- **Demo defaults to cached REAL generations** (closest held-out prompt by ingredient overlap) with a banner; live Gemma generation is opt-in (`LIVE_MODEL=1`) because a free CPU Space plus a JAX/TF install is slow and fragile. "The demo must never be broken."
- **Judge ties when the two presentation orders disagree** (cancels position bias); judge temperature 0.
- **Lint:** E501 ignored only for `eval/report.py` (Markdown template strings); notebooks excluded from ruff.
- **Annotation agreement measured on the winning model, not on A/B position**, since every annotator sees a different left/right order.
- **Smoke run interrupted at step 50/400 (host memory pressure); left for the owner to rerun** rather than restarting it on a memory-starved machine. The real progress was recorded with `scripts/collect_partial_train.py` -> `results/train_smoke_partial.json`; README/BLOCKERS say "pending" instead of showing benchmark numbers.
- **No v1.0.0 tag/release yet:** tagging a release without the benchmark it advertises would be misleading. Tag after the eval completes (`git tag v1.0.0 && gh release create v1.0.0 results/report.md`).
- **Self-review fixes:** shuffle the pairing pool before subsampling (was alphabetical); aliases for misleading head nouns (marshmallow cream, cream of mushroom soup, ...); separate HF repo for the full-run adapter; data rebuilt.
- The interrupted 50-step smoke run (results/train_smoke_partial.json) used the dataset built *before* the pairing-shuffle fix; the next full smoke run uses the rebuilt data, so its numbers supersede it.
