"""Corpus co-occurrence statistics and (normalized) pointwise mutual information.

For recipes r_1..r_N, each a *set* of canonical ingredients:
    p(a)   = #recipes containing a / N
    p(a,b) = #recipes containing both / N
    PMI(a,b)  = log( p(a,b) / (p(a) p(b)) )
    NPMI(a,b) = PMI / -log p(a,b)      in [-1, 1]
Pairs that never co-occur have PMI = -inf (reported as ``None``/negative).
"""

from __future__ import annotations

import gzip
import json
import math
import os
from collections import Counter
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path


class CooccurrenceStats:
    """Ingredient and ingredient-pair document frequencies over a recipe corpus."""

    def __init__(self) -> None:
        self.n_docs = 0
        self.single: Counter[str] = Counter()
        self.pair: Counter[tuple[str, str]] = Counter()

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def add(self, ingredients: Iterable[str]) -> None:
        """Add one recipe (duplicates within a recipe are ignored)."""
        items = sorted(set(i for i in ingredients if i))
        self.n_docs += 1
        self.single.update(items)
        self.pair.update(combinations(items, 2))

    @classmethod
    def from_recipes(cls, recipes: Iterable[Iterable[str]]) -> CooccurrenceStats:
        """Build from an iterable of ingredient collections."""
        stats = cls()
        for r in recipes:
            stats.add(r)
        return stats

    def count(self, a: str, b: str) -> int:
        """Number of recipes containing both ``a`` and ``b``."""
        return self.pair.get(self._key(a, b), 0)

    def pmi(self, a: str, b: str) -> float:
        """PMI(a, b); ``-inf`` if they never co-occur, ``nan`` if either is unseen."""
        if self.single[a] == 0 or self.single[b] == 0 or self.n_docs == 0:
            return math.nan
        c = self.count(a, b)
        if c == 0:
            return -math.inf
        n = self.n_docs
        return math.log((c / n) / ((self.single[a] / n) * (self.single[b] / n)))

    def npmi(self, a: str, b: str) -> float:
        """Normalized PMI in [-1, 1] (``-1`` for never co-occurring, ``nan`` for unseen)."""
        p = self.pmi(a, b)
        if math.isnan(p):
            return p
        if p == -math.inf:
            return -1.0
        pab = self.count(a, b) / self.n_docs
        if pab >= 1.0:
            return 1.0
        return p / -math.log(pab)

    def known(self, name: str) -> bool:
        """Whether ``name`` appears in at least one recipe."""
        return self.single[name] > 0

    def partners(self, a: str, min_count: int = 1) -> list[tuple[str, int]]:
        """All ingredients co-occurring with ``a`` at least ``min_count`` times."""
        index = getattr(self, "_index", None)
        if index is None or getattr(self, "_index_size", -1) != len(self.pair):
            index = {}
            for (x, y), c in self.pair.items():
                index.setdefault(x, []).append((y, c))
                index.setdefault(y, []).append((x, c))
            self._index, self._index_size = index, len(self.pair)
        return [(p, c) for p, c in index.get(a, []) if c >= min_count]

    # ---- (de)serialization ------------------------------------------------------------
    def to_file(self, path: str | os.PathLike[str], min_pair_count: int = 1) -> None:
        """Save as gzipped JSON (pairs below ``min_pair_count`` dropped to keep it small)."""
        payload = {
            "n_docs": self.n_docs,
            "single": dict(self.single),
            "pair": [[a, b, c] for (a, b), c in sorted(self.pair.items()) if c >= min_pair_count],
            "min_pair_count": min_pair_count,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> CooccurrenceStats:
        """Load stats saved with :meth:`to_file`."""
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        stats = cls()
        stats.n_docs = payload["n_docs"]
        stats.single = Counter(payload["single"])
        stats.pair = Counter({(a, b): c for a, b, c in payload["pair"]})
        return stats
