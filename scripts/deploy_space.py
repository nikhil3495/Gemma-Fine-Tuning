"""Assemble and upload the Gradio demo to a Hugging Face Space, then verify HTTP 200.

python scripts/deploy_space.py [--space nikhil3495/gemma-recipe-pairing] [--dry-run]
Requires HF_TOKEN (read from the environment only).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SPACE_README = """---
title: Gemma Recipe & Flavor Pairing
emoji: 🍅
colorFrom: red
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: LoRA fine-tuned Gemma for recipes & flavor pairing
---

Demo for https://github.com/nikhil3495/Gemma-Fine-Tuning. Pairings use corpus PMI and the
Ahn et al. (2011) flavor network (CC BY-NC-SA 3.0). Model outputs come from Gemma 3 270M
(Gemma Terms of Use) with a LoRA adapter.
"""


def stage(out: Path) -> Path:
    """Copy the files the Space needs into ``out``."""
    shutil.copy(ROOT / "app" / "app.py", out / "app.py")
    shutil.copy(ROOT / "app" / "requirements.txt", out / "requirements.txt")
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    shutil.copytree(ROOT / "src" / "recipe_gemma", out / "src" / "recipe_gemma",
                    ignore=shutil.ignore_patterns("__pycache__"))  # fmt: skip
    (out / "data" / "artifacts").mkdir(parents=True)
    for f in ("flavor_graph.json.gz", "cooccur_train.json.gz"):
        shutil.copy(ROOT / "data" / "artifacts" / f, out / "data" / "artifacts" / f)
    gen = ROOT / "results" / "generations"
    if gen.exists():
        shutil.copytree(gen, out / "results" / "generations")
    return out


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", default="nikhil3495/gemma-recipe-pairing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(Path(tmp))
        files = sorted(str(p.relative_to(staged)) for p in staged.rglob("*") if p.is_file())
        print(f"staged {len(files)} files")
        if args.dry_run:
            return 0
        if not token:
            print("HF_TOKEN not set: skipping Space deploy (see docs/BLOCKERS.md).")
            return 2
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=str(staged), repo_id=args.space, repo_type="space",
                          commit_message="Deploy demo from GitHub repo")  # fmt: skip
    owner, name = args.space.split("/")
    url = f"https://{owner}-{name}.hf.space".lower().replace("_", "-")
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        stage_name = api.get_space_runtime(args.space).stage
        print("space stage:", stage_name)
        if stage_name == "RUNNING":
            code = requests.get(url, timeout=30).status_code
            print(f"{url} -> HTTP {code}")
            return 0 if code == 200 else 1
        if stage_name in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
            return 1
        time.sleep(20)
    return 1


if __name__ == "__main__":
    sys.exit(main())
