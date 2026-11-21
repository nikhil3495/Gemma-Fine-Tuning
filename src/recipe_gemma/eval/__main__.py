"""Run the benchmark: generations (cached), perplexity, pairing/structure metrics, LLM judge.

python -m recipe_gemma.eval --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import logging

from recipe_gemma.utils.config import load_config
from recipe_gemma.utils.memory import configure_low_memory_jax

configure_low_memory_jax()  # before jax is imported


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument(
        "--stages",
        default="generate,perplexity,metrics,judge",
        help="comma-separated subset of generate,perplexity,metrics,judge",
    )
    ap.add_argument(
        "--models",
        default=None,
        help="comma-separated subset of the config's models (e.g. base) to save memory",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.models:
        cfg["models"] = [m for m in args.models.split(",") if m]
    from recipe_gemma.eval.runner import run

    run(cfg, set(args.stages.split(",")))


if __name__ == "__main__":
    main()
