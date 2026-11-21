"""Multinomial naive-Bayes cuisine tagger trained on Ahn et al. (2011) cuisine-labelled recipes.

The recipe corpus has no cuisine field, so we infer one from the ingredient set. Low-confidence
predictions return ``"any"`` instead of guessing.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

PRETTY = {
    "African": "African",
    "EastAsian": "East Asian",
    "EasternEuropean": "Eastern European",
    "LatinAmerican": "Latin American",
    "MiddleEastern": "Middle Eastern",
    "NorthAmerican": "North American",
    "NorthernEuropean": "Northern European",
    "SouthAsian": "South Asian",
    "SoutheastAsian": "Southeast Asian",
    "SouthernEuropean": "Southern European",
    "WesternEuropean": "Western European",
}


class CuisineTagger:
    """Bernoulli-style naive Bayes over ingredient presence with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_counts: Counter[str] = Counter()
        self.feat_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, rows: Iterable[tuple[str, Iterable[str]]]) -> CuisineTagger:
        """Fit on ``(cuisine, ingredients)`` rows."""
        for label, ingredients in rows:
            label = PRETTY.get(label, label)
            items = set(ingredients)
            self.class_counts[label] += 1
            self.feat_counts[label].update(items)
            self.vocab.update(items)
        return self

    def posterior(self, ingredients: Iterable[str]) -> dict[str, float]:
        """Posterior over cuisines given the (in-vocabulary) ingredients."""
        items = [i for i in set(ingredients) if i in self.vocab]
        total = sum(self.class_counts.values())
        logp: dict[str, float] = {}
        for label, n in self.class_counts.items():
            lp = math.log(n / total)
            for i in items:
                lp += math.log((self.feat_counts[label][i] + self.alpha) / (n + 2 * self.alpha))
            logp[label] = lp
        if not logp:
            return {}
        m = max(logp.values())
        z = sum(math.exp(v - m) for v in logp.values())
        return {k: math.exp(v - m) / z for k, v in logp.items()}

    def predict(self, ingredients: Iterable[str], min_prob: float = 0.6, min_known: int = 3) -> str:
        """Most likely cuisine, or ``"any"`` if confidence/evidence is too low."""
        items = [i for i in set(ingredients) if i in self.vocab]
        if len(items) < min_known:
            return "any"
        post = self.posterior(items)
        label, prob = max(post.items(), key=lambda kv: kv[1])
        return label if prob >= min_prob else "any"
