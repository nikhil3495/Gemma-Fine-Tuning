# Engineering decisions log

One line per decision: what was chosen and why. Newest at the bottom.

- **Compute plan = CPU smoke + notebooks for full run.** No NVIDIA GPU (Intel Iris Xe only), so: real LoRA smoke fine-tune locally on CPU + `notebooks/train_kaggle_tpu.ipynb` / `train_colab.ipynb` for the full run.
- **Base model = Gemma 3 270M instruction-tuned** (smallest Gemma 3 variant in keras_hub presets: `gemma3_instruct_270m`). Feasible on CPU at ~4-5 s/step; full config targets `gemma3_instruct_1b` on TPU/GPU.
- **Weights source = `hf://unsloth/gemma-3-270m-it`** because HF_TOKEN and Kaggle creds are absent and `google/*` Gemma repos are gated. The unsloth repo is an ungated redistribution of the same Gemma 3 weights under the Gemma Terms of Use. Config key `model.preset` switches to the official Kaggle preset when creds exist.
- **Tokenize eagerly in Python, fit on NumPy arrays.** keras_hub's tf.data path needs `tensorflow-text`, which has no Windows wheels; the pure-Python SentencePiece path works everywhere. It also enables loss masking on the prompt (we train on responses only).
- **Python 3.12** (jax 0.12 requires >=3.12; system Python is 3.11).
- **Gemini baseline model = `gemini-3.8-flash`** (newest GA Flash per ai.google.dev "What's new", 2026-11), configurable in `configs/eval.yaml`. Not run: no API key.
- **LLM judge = Gemini** (same key). Without a key the judge does not run, and the table says "not run (no API key)". No substitute judge was invented.
- **Recipe corpus = HF `corbt/all-recipes`** (2.15M recipes, ungated parquet, loads without a form). Its card has no license; the text format and rows match RecipeNLG (Bień et al. 2020), whose terms allow non-commercial research/education only. We use it under those terms and redistribute only small derived samples in `results/`.
- **Flavor graph = Ahn et al. 2011 supplementary data** (`srep00196-s2.csv`: ingredient pairs with shared-compound counts; `s3.csv`: 56k cuisine-labelled recipes), CC BY-NC-SA 3.0. The pair file gives shared-compound counts, not per-ingredient compound lists, so the "compound set" API is exposed as pairwise shared-compound counts (that is what the metric needs).
