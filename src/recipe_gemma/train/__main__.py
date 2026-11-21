"""CLI: python -m recipe_gemma.train --config configs/train_smoke.yaml"""

from __future__ import annotations

import argparse
import logging

from recipe_gemma.utils.config import load_config


def main() -> None:
    """Parse args and train."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="LoRA fine-tune Gemma on recipes + pairings.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    from recipe_gemma.train.trainer import train

    train(cfg)


if __name__ == "__main__":
    main()
