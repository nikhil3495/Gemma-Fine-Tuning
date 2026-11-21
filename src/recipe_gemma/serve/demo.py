"""Demo backend (UI-agnostic): pairing lookup and recipe generation (live or cached).

Modes
* cached (default, always works): serves REAL generations saved by the benchmark in
  ``results/generations/*.jsonl`` for the closest held-out prompt, with a clear banner.
* live (``LIVE_MODEL=1``): loads one Gemma + the trained LoRA adapter on CPU and generates on
  demand; falls back to cached mode if loading fails.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recipe_gemma.data.prompts import chat_prompt, recipe_user_message
from recipe_gemma.flavor.graph import FlavorGraph
from recipe_gemma.flavor.normalize import IngredientNormalizer, display_name
from recipe_gemma.flavor.pmi import CooccurrenceStats
from recipe_gemma.flavor.scorer import PairingScorer
from recipe_gemma.utils.config import read_jsonl

log = logging.getLogger(__name__)
MODEL_LABELS = {
    "base": "Base Gemma 3 270M (zero-shot)",
    "finetuned": "LoRA fine-tuned Gemma 3 270M",
}


@dataclass
class Demo:
    """Holds the flavor resources, cached generations and (optionally) a live model."""

    root: Path
    graph: FlavorGraph = field(init=False)
    stats: CooccurrenceStats = field(init=False)
    normalizer: IngredientNormalizer = field(init=False)
    scorer: PairingScorer = field(init=False)
    cached: dict[str, dict[str, dict[str, Any]]] = field(init=False, default_factory=dict)
    live: Any = field(init=False, default=None)
    live_error: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        art = self.root / "data" / "artifacts"
        self.graph = FlavorGraph.from_file(art / "flavor_graph.json.gz")
        self.stats = CooccurrenceStats.from_file(art / "cooccur_train.json.gz")
        self.normalizer = IngredientNormalizer(self.graph.nodes)
        self.scorer = PairingScorer(self.stats, self.graph)
        gen_dir = self.root / "results" / "generations"
        for key in ("base", "finetuned"):
            path = gen_dir / f"{key}.jsonl"
            if path.exists():
                self.cached[key] = {r["id"]: r for r in read_jsonl(path)}

    # ------------------------------------------------------------------ pairing tab
    def pairings(self, ingredient: str, k: int = 8) -> tuple[str, list[list[Any]]]:
        """Ranked pairings for ``ingredient`` from corpus PMI + flavor-compound overlap."""
        canon = self.normalizer.canonical(ingredient.strip())
        if not canon:
            return "Enter an ingredient.", []
        ranked = self.scorer.rank(canon, k=k, min_count=3)
        rows = [
            [
                display_name(s.partner),
                s.cooccur,
                round(s.npmi, 3),
                s.shared_compounds if s.in_graph else "n/a (not in flavor graph)",
            ]
            for s in ranked
        ]
        if not rows:
            return (
                f"No corpus evidence for **{display_name(canon)}** (seen in "
                f"{self.stats.single.get(canon, 0)} training recipes).",
                [],
            )
        note = (
            f"**{display_name(canon)}** appears in {self.stats.single[canon]:,} of "
            f"{self.stats.n_docs:,} training recipes. Ranked by NPMI + 0.1*log(1 + shared "
            f"compounds); compound counts from Ahn et al. (2011)."
        )
        return note, rows

    def cached_pairing_outputs(self, ingredient: str) -> dict[str, str]:
        """Cached model answers if ``ingredient`` is one of the held-out eval anchors."""
        canon = self.normalizer.canonical(ingredient.strip())
        key = f"anchor:{canon}"
        return {m: g[key]["output"] for m, g in self.cached.items() if key in g}

    # ------------------------------------------------------------------ recipe tab
    def recipe_examples(self) -> list[str]:
        """Instructions of cached held-out recipe prompts (for the example picker)."""
        rows = self.cached.get("finetuned") or self.cached.get("base") or {}
        return sorted({r["user"].split("\n")[0] for r in rows.values() if r["task"] == "recipe"})

    def closest_cached(self, ingredients: list[str]) -> dict[str, Any] | None:
        """Cached recipe prompt whose requested ingredients best overlap (Jaccard)."""
        want = {self.normalizer.canonical(i) for i in ingredients if i.strip()}
        rows = self.cached.get("finetuned") or self.cached.get("base") or {}
        best, best_score = None, -1.0
        for r in rows.values():
            if r["task"] != "recipe":
                continue
            have = set(r["meta"]["requested"])
            score = len(want & have) / max(1, len(want | have))
            if score > best_score:
                best, best_score = r, score
        if best is None:
            return None
        return {**best, "overlap": best_score}

    def try_load_live(self, preset: str, adapter: str | None) -> bool:
        """Load ONE Gemma instance with the LoRA adapter for live generation (~2 GB RAM).

        Returns success; any failure (missing adapter, no keras_hub, ...) leaves cached mode."""
        try:
            if not adapter or not Path(adapter).exists():
                raise FileNotFoundError(f"adapter not found: {adapter} (train first)")
            from recipe_gemma.train.lm import load_lm

            lm = load_lm({"preset": preset})
            lm.load_adapter(adapter)
            lm.set_sampler("top_k", 0.7, 40, 0.95, 1234)
            self.live = {"finetuned": lm}
            return True
        except Exception as exc:
            self.live_error = f"{type(exc).__name__}: {exc}"
            log.warning("live model unavailable: %s", self.live_error)
            return False

    def generate_live(self, ingredients: list[str], cuisine: str, which: str = "finetuned") -> str:
        """Generate a recipe with the live fine-tuned model."""
        canon = [self.normalizer.canonical(i) for i in ingredients if i.strip()]
        prompt = chat_prompt(recipe_user_message(canon, cuisine or "any"))
        return self.live[which].generate([prompt], max_length=576)[0]


def default_root() -> Path:
    """Repo root; override with RECIPE_GEMMA_ROOT."""
    env = os.environ.get("RECIPE_GEMMA_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "artifacts" / "flavor_graph.json.gz").exists():
            return parent
    return Path.cwd()
