"""Run the benchmark: generations (cached), perplexity, pairing/structure metrics, LLM judge.

python -m recipe_gemma.eval --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import logging

from recipe_gemma.eval.runner import run
from recipe_gemma.utils.config import load_config


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
    args = ap.parse_args()
    run(load_config(args.config), set(args.stages.split(",")))


if __name__ == "__main__":
    main()
