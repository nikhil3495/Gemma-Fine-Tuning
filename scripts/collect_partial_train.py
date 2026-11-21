"""Record an INTERRUPTED training run's real progress from its log (no numbers typed by hand).

python scripts/collect_partial_train.py --log runs/smoke_train.log --out results/train_smoke_partial.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recipe_gemma.utils.config import load_config, provenance, write_json  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="runs/smoke_train.log")
    ap.add_argument("--config", default="configs/train_smoke.yaml")
    ap.add_argument("--out", default="results/train_smoke_partial.json")
    ap.add_argument(
        "--reason", default="process stopped by the host (critically low system memory)"
    )
    args = ap.parse_args()
    text = (ROOT / args.log).read_text(encoding="utf-8", errors="ignore")
    train = [
        {"step": int(s), "loss": float(v), "sec_per_step": float(r)}
        for s, v, r in re.findall(r"\[train\] step=(\d+) loss=([\d.]+) sec_per_step=([\d.]+)", text)
    ]
    evals = [
        {"step": int(s), "train_loss_window": float(a), "val_loss": float(b)}
        for s, a, b in re.findall(r"\[eval\] step=(\d+) loss=([\d.]+) val_loss=([\d.]+)", text)
    ]
    base_ppl = re.findall(r"base val ppl=([\d.]+)", text)
    sizes = re.findall(r"train=(\d+) val=(\d+) dropped=(\{.*?\})", text)
    cfg = load_config(ROOT / args.config)
    out = {
        "meta": provenance(cfg, model_preset=cfg["model"]["preset"], run_name=cfg["run_name"]),
        "status": "INTERRUPTED",
        "reason": args.reason,
        "completed_steps": max((t["step"] for t in train), default=0),
        "planned_steps": cfg["train"]["max_steps"],
        "adapter_saved": False,
        "data": {"train_examples": int(sizes[-1][0]), "val_examples": int(sizes[-1][1])}
        if sizes
        else None,
        "base_val_ppl": float(base_ppl[-1]) if base_ppl else None,
        "train_log": train,
        "evals": evals,
        "source_log": args.log,
        "resume_command": f"python -m recipe_gemma.train --config {args.config}",
    }
    write_json(ROOT / args.out, out)
    print(f"wrote {args.out}: {out['completed_steps']} steps, {len(evals)} val evals")


if __name__ == "__main__":
    main()
