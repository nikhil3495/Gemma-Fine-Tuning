"""Per-item evaluation metrics for generated recipes and pairings.

Definitions (also in docs/EVAL_PROTOCOL.md):

Pairing validity (both tasks). Generated ingredients are normalized to canonical names; then for
every evaluated pair (Task B: anchor x each suggestion; Task A: all unordered pairs of distinct
generated ingredients):
  * pmi_positive      - corpus PMI (train split) > 0. Pairs with unseen ingredients count as 0.
  * shares_compound   - >= 1 shared flavor compound in the Ahn et al. graph (missing -> 0).
  * graph_covered     - both ingredients are flavor-graph nodes.
  * shares_compound_covered - shares_compound restricted to covered pairs.
Item score = mean over its pairs; reported value = mean over items with a 95% bootstrap CI.

Structure validity (Task A).
  * well_formed       - parses into title + >=1 ingredient + >=1 step.
  * all_requested     - every requested ingredient appears in the generated ingredient list.
  * requested_recall  - fraction of requested ingredients that appear.
  * unrecognized_rate - fraction of generated ingredient lines whose normalized name is in
                        neither the flavor graph nor the training-corpus vocabulary
                        (proxy for hallucinated / non-food ingredients).
Task B format: format_ok = exactly 5 parseable suggestions; anchor_repeated = anchor suggested.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

from recipe_gemma.data.prompts import parse_pairing_output, parse_recipe_output
from recipe_gemma.flavor.normalize import IngredientNormalizer, clean_ingredient_line
from recipe_gemma.flavor.scorer import PairingScorer


def _pair_stats(scorer: PairingScorer, pairs: list[tuple[str, str]]) -> dict[str, float]:
    if not pairs:
        nan = math.nan
        return {
            "pmi_positive": nan,
            "shares_compound": nan,
            "graph_covered": nan,
            "shares_compound_covered": nan,
            "n_pairs": 0,
        }
    scores = [scorer.score(a, b) for a, b in pairs]
    covered = [s for s in scores if s.in_graph]
    return {
        "pmi_positive": sum(s.pmi_positive for s in scores) / len(scores),
        "shares_compound": sum(s.shares_compound for s in scores) / len(scores),
        "graph_covered": len(covered) / len(scores),
        "shares_compound_covered": (
            sum(s.shares_compound for s in covered) / len(covered) if covered else math.nan
        ),
        "n_pairs": len(scores),
    }


def _mentions(requested: str, ingredient_lines: list[str], canon: list[str]) -> bool:
    """Requested canonical ingredient present (canonical match or word match in text)."""
    if requested in canon:
        return True
    words = requested.replace("_", " ")
    return any(words in clean_ingredient_line(line) for line in ingredient_lines)


def recipe_item_metrics(
    output: str,
    requested: list[str],
    normalizer: IngredientNormalizer,
    scorer: PairingScorer,
) -> dict[str, Any]:
    """Metrics for one Task-A generation."""
    parsed = parse_recipe_output(output)
    canon = [normalizer(line) for line in parsed.ingredients]
    canon = [c for c in canon if c]
    known = [c for c in canon if normalizer.in_vocab(c) or scorer.stats.known(c)]
    distinct = sorted(set(canon))
    hits = [_mentions(r, parsed.ingredients, canon) for r in requested]
    out: dict[str, Any] = {
        "well_formed": float(parsed.well_formed),
        "has_notes": float(bool(parsed.notes)),
        "n_ingredients": len(parsed.ingredients),
        "n_steps": len(parsed.steps),
        "all_requested": float(all(hits)) if requested else math.nan,
        "requested_recall": sum(hits) / len(hits) if hits else math.nan,
        "unrecognized_rate": (1 - len(known) / len(canon)) if canon else math.nan,
    }
    out.update(_pair_stats(scorer, list(combinations(distinct, 2))))
    return out


def pairing_item_metrics(
    output: str, anchor: str, normalizer: IngredientNormalizer, scorer: PairingScorer, k: int = 5
) -> dict[str, Any]:
    """Metrics for one Task-B generation."""
    names = parse_pairing_output(output, k=k)
    canon = []
    for n in names:
        c = normalizer.canonical(clean_ingredient_line(n) or n)
        if c and c not in canon:
            canon.append(c)
    partners = [c for c in canon if c != anchor]
    out: dict[str, Any] = {
        "format_ok": float(len(names) == k),
        "n_suggestions": len(names),
        "anchor_repeated": float(anchor in canon),
    }
    out.update(_pair_stats(scorer, [(anchor, p) for p in partners]))
    return out


def item_metrics(
    row: dict[str, Any], normalizer: IngredientNormalizer, scorer: PairingScorer
) -> dict[str, Any]:
    """Dispatch on ``row["task"]``; ``row`` must carry ``output`` and ``meta``."""
    if row["task"] == "recipe":
        return recipe_item_metrics(row["output"], row["meta"]["requested"], normalizer, scorer)
    return pairing_item_metrics(row["output"], row["meta"]["anchor"], normalizer, scorer)


RECIPE_KEYS = [
    "well_formed",
    "all_requested",
    "requested_recall",
    "unrecognized_rate",
    "pmi_positive",
    "shares_compound",
    "graph_covered",
    "shares_compound_covered",
]
PAIRING_KEYS = [
    "format_ok",
    "anchor_repeated",
    "pmi_positive",
    "shares_compound",
    "graph_covered",
    "shares_compound_covered",
]
