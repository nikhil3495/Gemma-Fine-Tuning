"""End-to-end data build.

python -m recipe_gemma.data.build --config configs/data.yaml

Outputs
  data/processed/{smoke,full}/{train,val}.jsonl      training examples (gitignored)
  data/processed/test_recipes.jsonl                   held-out test recipes (gitignored)
  data/artifacts/flavor_graph.json.gz                 compiled flavor graph (committed)
  data/artifacts/cooccur_train.json.gz                train-split co-occurrence stats (committed)
  data/artifacts/eval_prompts.jsonl                   fixed held-out eval prompts (committed)
  results/data_stats.json                             counts at every stage (committed)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
from typing import Any

from recipe_gemma.data.clean import Recipe, clean_recipe, dedupe, is_valid, parse_recipe_text
from recipe_gemma.data.prompts import (
    chat_prompt,
    format_pairing_target,
    format_recipe_target,
    pairing_user_message,
    recipe_user_message,
)
from recipe_gemma.flavor.build import build_graph
from recipe_gemma.flavor.cuisine import CuisineTagger
from recipe_gemma.flavor.graph import FlavorGraph
from recipe_gemma.flavor.normalize import IngredientNormalizer, display_name
from recipe_gemma.flavor.pmi import CooccurrenceStats
from recipe_gemma.flavor.scorer import PairingScorer, PairScore
from recipe_gemma.utils.config import REPO_ROOT, load_config, provenance, write_json, write_jsonl

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- helpers
def stable_bucket(key: str, seed: int) -> float:
    """Deterministic pseudo-uniform number in [0,1) from a string key and a seed."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)


def assign_split(key: str, seed: int, fracs: dict[str, float]) -> str:
    """Assign ``train``/``val``/``test`` by hashing ``key`` (stable under re-ordering)."""
    u = stable_bucket(key, seed)
    acc = 0.0
    for name in ("train", "val", "test"):
        acc += fracs[name]
        if u < acc:
            return name
    return "test"


def flavor_notes(
    canon: list[str], graph: FlavorGraph, cuisine: str, staples: frozenset[str] = frozenset()
) -> str:
    """Templated flavor notes from the flavor graph (deterministic, documented in DATA_CARD)."""
    pairs = []
    items = sorted(set(canon))
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            w = graph.shared_compounds(a, b)
            if w > 0:
                pairs.append((w, a, b))
    # Prefer pairs of non-staple ingredients, then more shared compounds.
    pairs.sort(key=lambda t: ((t[1] in staples) + (t[2] in staples), -t[0], t[1], t[2]))
    parts = [
        f"{display_name(a)} and {display_name(b)} share {w} flavor compounds"
        for w, a, b in pairs[:2]
    ]
    style = f"{cuisine} " if cuisine != "any" else ""
    if parts:
        return f"A {style}dish built on compatible flavors: " + "; ".join(parts) + "."
    names = ", ".join(display_name(x) for x in items[:3])
    return f"A {style}dish centered on {names}."


def pairing_reason(s: PairScore) -> str:
    """One-line templated reason built from the evidence."""
    reason = f"co-occurs in {s.cooccur} recipes (NPMI {s.npmi:.2f})"
    if s.shared_compounds:
        reason += f" and shares {s.shared_compounds} flavor compounds"
    return reason + "."


def key_ingredients(r: Recipe, staples: set[str], k: int, rng: random.Random) -> list[str]:
    """Up to ``k`` non-staple canonical ingredients (order preserved, then sampled)."""
    seen: list[str] = []
    for c in r.canon:
        if c and c not in staples and c not in seen:
            seen.append(c)
    if len(seen) > k:
        seen = sorted(rng.sample(seen, k), key=seen.index)
    return seen


# --------------------------------------------------------------------------- stages
def load_recipes(cfg: dict, stats: dict[str, Any]) -> list[Recipe]:
    """Download one parquet shard, parse, clean, filter, dedupe and sample."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rc = cfg["recipes"]
    path = hf_hub_download(
        repo_id=rc["hf_dataset"],
        filename=rc["shard_file"],
        repo_type="dataset",
        local_dir=REPO_ROOT / "data" / "raw",
    )
    texts = pq.read_table(path, columns=[rc["text_column"]]).column(0).to_pylist()
    stats["rows_in_shard"] = len(texts)
    order = list(range(len(texts)))
    random.Random(cfg["seed"]).shuffle(order)
    order = order[: rc["max_rows_scan"]]
    stats["rows_scanned"] = len(order)

    parsed, valid = 0, []
    for idx in order:
        r = parse_recipe_text(texts[idx])
        if r is None:
            continue
        parsed += 1
        r = clean_recipe(r)
        r.uid = f"{rc['hf_dataset']}#{idx}"
        if is_valid(r, rc["min_ingredients"], rc["min_steps"], rc["max_chars"]):
            valid.append(r)
    stats["parsed"] = parsed
    stats["after_filters"] = len(valid)
    unique = list(dedupe(valid))
    stats["after_dedupe"] = len(unique)
    sample = unique[: rc["n_recipes"]]
    stats["sampled"] = len(sample)
    return sample


def build(cfg: dict) -> dict[str, Any]:
    """Run the whole pipeline and return the stats dict."""
    seed = cfg["seed"]
    stats: dict[str, Any] = {}
    graph, ahn_recipes = build_graph(cfg)
    stats["flavor_graph"] = {"nodes": len(graph), "edges": graph.num_edges}

    vocab = set(graph.nodes)
    normalizer = IngredientNormalizer(vocab)
    tagger = CuisineTagger().fit(ahn_recipes)

    recipes = load_recipes(cfg, stats)
    in_graph = total = 0
    for r in recipes:
        r.canon = [normalizer(line) for line in r.ingredients]
        total += len(r.canon)
        in_graph += sum(c in vocab for c in r.canon)
        r.cuisine = tagger.predict(r.canon)
    stats["ingredient_lines"] = total
    stats["ingredient_lines_mapped_to_graph"] = in_graph
    stats["cuisine_counts"] = _count(r.cuisine for r in recipes)

    splits: dict[str, list[Recipe]] = {"train": [], "val": [], "test": []}
    for r in recipes:
        splits[assign_split(r.uid, seed, cfg["split"])].append(r)
    stats["recipe_splits"] = {k: len(v) for k, v in splits.items()}

    # Co-occurrence statistics from the TRAIN split only (no test leakage).
    co = CooccurrenceStats.from_recipes(r.canon for r in splits["train"])
    pc = cfg["pairing"]
    co.to_file(REPO_ROOT / pc["stats_out"], min_pair_count=2)
    scorer = PairingScorer(co, graph)

    # ---- Task A examples
    rt = cfg["recipe_task"]
    staples = set(rt["staples"])
    rng = random.Random(seed)

    def recipe_example(r: Recipe) -> dict[str, Any] | None:
        keys = key_ingredients(r, staples, rt["max_prompt_ingredients"], rng)
        if len(keys) < 2:
            return None
        user = recipe_user_message(keys, r.cuisine)
        target = format_recipe_target(
            r.title,
            r.ingredients,
            r.steps,
            flavor_notes(r.canon, graph, r.cuisine, frozenset(staples)),
        )
        return {
            "id": r.uid,
            "task": "recipe",
            "prompt": chat_prompt(user),
            "user": user,
            "response": target,
            "meta": {"requested": keys, "cuisine": r.cuisine, "title": r.title},
        }

    # ---- Task B examples
    anchors = sorted(a for a, n in co.single.items() if n >= pc["min_anchor_count"])
    anchor_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for a in anchors:
        anchor_split[assign_split("anchor:" + a, seed, pc["anchor_split"])].append(a)

    def pairing_example(a: str) -> dict[str, Any] | None:
        ranked = scorer.rank(a, k=pc["top_k"], min_count=pc["min_pair_count"])
        if len(ranked) < pc["top_k"]:
            return None
        user = pairing_user_message(a)
        target = format_pairing_target(a, [(s.partner, pairing_reason(s)) for s in ranked])
        return {
            "id": f"anchor:{a}",
            "task": "pairing",
            "prompt": chat_prompt(user),
            "user": user,
            "response": target,
            "meta": {"anchor": a, "reference_partners": [s.partner for s in ranked]},
        }

    examples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in ("train", "val", "test"):
        rec = [e for e in map(recipe_example, splits[split]) if e]
        pair = [e for e in map(pairing_example, anchor_split[split]) if e]
        examples[split] = {"recipe": rec, "pairing": pair}
    stats["examples"] = {s: {t: len(v) for t, v in d.items()} for s, d in examples.items()}
    stats["anchors"] = {k: len(v) for k, v in anchor_split.items()}

    # ---- write profiles (smoke is a subset of full; test/eval prompts shared)
    out_dir = REPO_ROOT / cfg["out_dir"]
    for profile, pcfg in cfg["profiles"].items():
        prng = random.Random(seed)
        rec_tr = list(examples["train"]["recipe"])
        prng.shuffle(rec_tr)
        n_train = pcfg["train"] or len(rec_tr)
        # Task B has far fewer unique examples (one per anchor ingredient); oversample it
        # with replacement so it makes up `pairing_frac` of the training mix.
        n_pair = round(n_train * pcfg.get("pairing_frac", 0.0))
        pair_pool = examples["train"]["pairing"]
        pairs = [pair_pool[i % len(pair_pool)] for i in range(n_pair)] if pair_pool else []
        tr = rec_tr[: n_train - len(pairs)] + pairs
        prng.shuffle(tr)
        va = examples["val"]["recipe"] + examples["val"]["pairing"]
        prng.shuffle(va)
        write_jsonl(out_dir / profile / "train.jsonl", tr[:n_train])
        write_jsonl(out_dir / profile / "val.jsonl", va[: pcfg["val"]])
        stats.setdefault("profiles", {})[profile] = {
            "train": len(tr),
            "train_pairing": len(pairs),
            "val": min(pcfg["val"], len(va)),
        }

    # Fixed held-out eval prompts: full = first N of each task; smoke = first n of those.
    n_full = cfg["profiles"]["full"]["eval_per_task"]
    eval_rows = []
    for task in ("recipe", "pairing"):
        rows = sorted(examples["test"][task], key=lambda e: stable_bucket(e["id"], seed))
        for rank, e in enumerate(rows[:n_full]):
            eval_rows.append({**e, "eval_rank": rank})
    write_jsonl(REPO_ROOT / cfg["eval_prompts_out"], eval_rows)
    write_jsonl(out_dir / "test_recipes.jsonl", [r.to_dict() for r in splits["test"]])
    stats["eval_prompts"] = _count(e["task"] for e in eval_rows)

    result = {"meta": provenance(cfg), "stats": stats}
    write_json(REPO_ROOT / "results" / "data_stats.json", result)
    log.info("data stats: %s", stats)
    return result


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build recipe + pairing datasets.")
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()


__all__ = ["assign_split", "build", "flavor_notes", "key_ingredients", "stable_bucket"]
