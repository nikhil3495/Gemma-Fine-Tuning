"""Download the Ahn et al. (2011) flavor-network data and compile the graph artifact.

python -m recipe_gemma.flavor.build --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import requests

from recipe_gemma.flavor.graph import SOURCE_URLS, FlavorGraph, read_ahn_recipes
from recipe_gemma.utils.config import REPO_ROOT, load_config

log = logging.getLogger(__name__)


def download_ahn(raw_dir: Path) -> dict[str, Path]:
    """Download + unzip the two supplementary CSVs (idempotent)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, url in SOURCE_URLS.items():
        name = "srep00196-s2.csv" if key == "pairs" else "srep00196-s3.csv"
        target = raw_dir / name
        if not target.exists():
            log.info("downloading %s", url)
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                member = next(m for m in zf.namelist() if m.endswith(".csv"))
                target.write_bytes(zf.read(member))
        out[key] = target
    return out


def build_graph(cfg: dict) -> tuple[FlavorGraph, list[tuple[str, list[str]]]]:
    """Return the flavor graph (also saved to ``flavor.graph_out``) and Ahn cuisine recipes."""
    raw_dir = REPO_ROOT / cfg["flavor"]["raw_dir"]
    paths = download_ahn(raw_dir)
    graph = FlavorGraph.from_ahn_csv(paths["pairs"])
    ahn_recipes = read_ahn_recipes(paths["recipes"])
    for _, ings in ahn_recipes:  # recipe-only ingredients become (isolated) nodes too
        for i in ings:
            graph.add_node(i)
    graph.to_file(REPO_ROOT / cfg["flavor"]["graph_out"])
    log.info("flavor graph: %d nodes, %d edges", len(graph), graph.num_edges)
    return graph, ahn_recipes


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    build_graph(load_config(args.config))


if __name__ == "__main__":
    main()
