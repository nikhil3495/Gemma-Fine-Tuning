"""CLI: python -m recipe_gemma.train --config configs/train_smoke.yaml [--resume] [--dry-run]"""

from __future__ import annotations

import argparse
import logging

from recipe_gemma.utils.config import load_config
from recipe_gemma.utils.memory import configure_low_memory_jax

configure_low_memory_jax()  # must run before jax is imported


def main() -> None:
    """Parse args and train."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="LoRA fine-tune Gemma on recipes + pairings.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="5 optimizer steps + tiny val set into checkpoints/dryrun; reports peak memory",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    if args.dry_run:
        cfg["run_name"] = f"{cfg['run_name']}-dryrun"
        cfg["train"].update(max_steps=5, eval_every=5, log_every=1, warmup_steps=1)
        cfg["data"]["max_val"] = 4
        cfg["out"] = {
            "dir": "checkpoints/dryrun",
            "results": "checkpoints/dryrun/train_dryrun.json",
            "log_csv": "checkpoints/dryrun/log.csv",
            "tensorboard": None,
        }
    from recipe_gemma.train.trainer import train

    res = train(cfg, resume=args.resume)
    print(
        f"[done] steps={res['training']['steps_run']} peak_rss_gb={res['memory']['peak_rss_gb']} "
        f"val_ppl base={res['val_perplexity']['base']['ppl']:.3f} "
        f"tuned={res['val_perplexity']['fine_tuned']['ppl']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
