"""Pairing scorer: combines corpus PMI and flavor-compound overlap.

Used (1) to build Task-B training targets from the *training split* and (2) to score model
outputs at evaluation time. Both share one definition so the metric and the targets agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from recipe_gemma.flavor.graph import FlavorGraph
from recipe_gemma.flavor.pmi import CooccurrenceStats


@dataclass(frozen=True)
class PairScore:
    """Evidence for one (anchor, partner) pair."""

    anchor: str
    partner: str
    cooccur: int
    pmi: float  # nan if either ingredient unseen in corpus; -inf if never co-occur
    npmi: float
    shared_compounds: int
    in_graph: bool  # both ingredients are flavor-graph nodes

    @property
    def pmi_positive(self) -> bool:
        """Corpus PMI strictly positive (pair co-occurs more than chance)."""
        return not math.isnan(self.pmi) and self.pmi > 0

    @property
    def shares_compound(self) -> bool:
        """At least one shared flavor compound."""
        return self.shared_compounds >= 1


class PairingScorer:
    """Score and rank ingredient pairings."""

    def __init__(self, stats: CooccurrenceStats, graph: FlavorGraph) -> None:
        self.stats = stats
        self.graph = graph

    def score(self, anchor: str, partner: str) -> PairScore:
        """Evidence for one pair of canonical ingredient names."""
        return PairScore(
            anchor=anchor,
            partner=partner,
            cooccur=self.stats.count(anchor, partner),
            pmi=self.stats.pmi(anchor, partner),
            npmi=self.stats.npmi(anchor, partner),
            shared_compounds=self.graph.shared_compounds(anchor, partner),
            in_graph=self.graph.covered(anchor, partner),
        )

    def rank(self, anchor: str, k: int = 5, min_count: int = 5) -> list[PairScore]:
        """Top-``k`` partners: NPMI > 0 and >= ``min_count`` co-occurrences, ranked by
        ``npmi + 0.1 * log1p(shared_compounds)`` (compound overlap is a tie-breaker-sized bonus)."""
        cands = []
        for partner, _count in self.stats.partners(anchor, min_count=min_count):
            s = self.score(anchor, partner)
            if s.npmi > 0:
                cands.append(s)
        cands.sort(key=lambda s: (-(s.npmi + 0.1 * math.log1p(s.shared_compounds)), s.partner))
        return cands[:k]
