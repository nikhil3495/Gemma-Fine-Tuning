"""Push the LoRA adapter + model card + training results to the Hugging Face Hub.

python scripts/push_adapter.py --config configs/train_smoke.yaml
Requires HF_TOKEN (environment only). Exits 2 (skipped) without it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recipe_gemma.utils.config import load_config  # noqa: E402


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train_smoke.yaml")
    ap.add_argument("--repo-id", default=None)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    repo_id = args.repo_id or cfg["hub"]["repo_id"]
    out_dir = ROOT / cfg["out"]["dir"]
    adapter = next(iter(sorted(out_dir.glob("adapter.lora.*"))), None)
    if adapter is None:
        print(f"No adapter in {out_dir}; train first.")
        return 1
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set: skipping Hub upload (optional; unused offline).")
        return 2
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        shutil.copy(adapter, t / adapter.name)
        shutil.copy(ROOT / "MODEL_CARD.md", t / "README.md")
        for f in (cfg["out"]["results"], cfg["out"]["log_csv"]):
            if (ROOT / f).exists():
                shutil.copy(ROOT / f, t / Path(f).name)
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=repo_id, commit_message="Upload LoRA adapter")
    print(f"https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
