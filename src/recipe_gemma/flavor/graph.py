"""Flavor network (Ahn et al. 2011): ingredient pairs and their number of shared flavor compounds.

Source: Y.-Y. Ahn, S. E. Ahnert, J. P. Bagrow, A.-L. Barabási, "Flavor network and the principles
of food pairing", Scientific Reports 1:196 (2011), doi:10.1038/srep00196, supplementary data
``srep00196-s2.csv`` (CC BY-NC-SA 3.0).
"""

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

SOURCE_URLS = {
    "pairs": "https://static-content.springer.com/esm/art%3A10.1038%2Fsrep00196/MediaObjects/41598_2011_BFsrep00196_MOESM2_ESM.zip",
    "recipes": "https://static-content.springer.com/esm/art%3A10.1038%2Fsrep00196/MediaObjects/41598_2011_BFsrep00196_MOESM3_ESM.zip",
}


class FlavorGraph:
    """Undirected weighted graph: weight = number of flavor compounds two ingredients share."""

    def __init__(self, edges: Iterable[tuple[str, str, int]] = ()) -> None:
        self._adj: dict[str, dict[str, int]] = defaultdict(dict)
        self.nodes: set[str] = set()
        for a, b, w in edges:
            self.add_edge(a, b, w)

    def add_edge(self, a: str, b: str, weight: int) -> None:
        """Add an undirected edge."""
        if a == b:
            return
        self._adj[a][b] = int(weight)
        self._adj[b][a] = int(weight)
        self.nodes.update((a, b))

    def add_node(self, name: str) -> None:
        """Register an ingredient that may have no edges."""
        self.nodes.add(name)

    def shared_compounds(self, a: str, b: str) -> int:
        """Number of flavor compounds shared by ``a`` and ``b`` (0 if none or unknown)."""
        return self._adj.get(a, {}).get(b, 0)

    def covered(self, *names: str) -> bool:
        """True if every name is a node of the graph."""
        return all(n in self.nodes for n in names)

    def neighbors(self, name: str, k: int = 10) -> list[tuple[str, int]]:
        """Top-``k`` neighbours by shared compounds (ties broken alphabetically)."""
        items = self._adj.get(name, {}).items()
        return sorted(items, key=lambda kv: (-kv[1], kv[0]))[:k]

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of undirected edges."""
        return sum(len(v) for v in self._adj.values()) // 2

    # ---- (de)serialization -------------------------------------------------------------
    def to_file(self, path: str | os.PathLike[str]) -> None:
        """Save as gzipped JSON ``{"nodes": [...], "edges": [[a, b, w], ...]}``."""
        edges = [[a, b, w] for a, nb in self._adj.items() for b, w in nb.items() if a < b]
        edges.sort()
        payload = {"nodes": sorted(self.nodes), "edges": edges}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> FlavorGraph:
        """Load a graph saved with :meth:`to_file`."""
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        graph = cls((a, b, w) for a, b, w in payload["edges"])
        for n in payload["nodes"]:
            graph.add_node(n)
        return graph

    @classmethod
    def from_ahn_csv(cls, path: str | os.PathLike[str]) -> FlavorGraph:
        """Parse ``srep00196-s2.csv`` (``ingredient_a,ingredient_b,shared_compounds``)."""
        graph = cls()
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b, w = line.rsplit(",", 2)
                graph.add_edge(a.strip(), b.strip(), int(w))
        return graph


def read_ahn_recipes(path: str | os.PathLike[str]) -> list[tuple[str, list[str]]]:
    """Parse ``srep00196-s3.csv`` into ``(cuisine, [ingredients])`` rows."""
    rows: list[tuple[str, list[str]]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2:
                rows.append((parts[0], parts[1:]))
    return rows
