"""Benchmark orchestration (see docs/EVAL_PROTOCOL.md)."""

from __future__ import annotations

import csv
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from recipe_gemma.eval.gemini import GeminiClient, api_key
from recipe_gemma.eval.judge import CRITERIA, judge_item
from recipe_gemma.eval.metrics import PAIRING_KEYS, RECIPE_KEYS, item_metrics
from recipe_gemma.eval.stats import bootstrap_ci, win_rates
from recipe_gemma.flavor.graph import FlavorGraph
from recipe_gemma.flavor.normalize import IngredientNormalizer
from recipe_gemma.flavor.pmi import CooccurrenceStats
from recipe_gemma.flavor.scorer import PairingScorer
from recipe_gemma.utils.config import (
    REPO_ROOT,
    provenance,
    read_jsonl,
    write_json,
    write_jsonl,
)
from recipe_gemma.utils.memory import free_memory

log = logging.getLogger(__name__)
NO_KEY = "optional, not run (needs GEMINI_API_KEY)"


# ------------------------------------------------------------------------------ helpers
def select_prompts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Fixed held-out prompts: the first ``n_per_task`` (by eval_rank) of each task."""
    rows = read_jsonl(REPO_ROOT / cfg["eval_prompts"])
    n = int(cfg["n_per_task"])
    return [r for r in rows if r["eval_rank"] < n]


def gen_path(cfg: dict[str, Any], model_key: str) -> Path:
    """Cache file for one model's generations."""
    return REPO_ROOT / cfg["generations_dir"] / f"{model_key}.jsonl"


def load_cached(path: Path) -> dict[str, dict[str, Any]]:
    """Existing generations keyed by item id."""
    return {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _gen_row(p: dict[str, Any], output: str, model_key: str, model_id: str, sec: float) -> dict:
    return {
        "id": p["id"],
        "task": p["task"],
        "eval_rank": p["eval_rank"],
        "user": p["user"],
        "output": output,
        "model": model_key,
        "model_id": model_id,
        "meta": p["meta"],
        "gen_seconds": round(sec, 2),
        "timestamp_utc": _now(),
    }


def generate_with_lm(
    lm: Any, prompts: list[dict[str, Any]], cfg: dict[str, Any], model_key: str, model_id: str
) -> list[dict[str, Any]]:
    """Generate (with caching) for a local CausalLM; returns all rows for these prompts."""
    dec = cfg["decoding"]
    path = gen_path(cfg, model_key)
    cache = load_cached(path)
    todo = [p for p in prompts if p["id"] not in cache]
    bs = int(dec["batch_size"])
    for task in ("recipe", "pairing"):
        items = [p for p in todo if p["task"] == task]
        for i in range(0, len(items), bs):
            batch = items[i : i + bs]
            # Seed per batch (keyed by task + eval rank) so outputs don't depend on what was
            # already cached when an interrupted eval is resumed.
            seed = int(dec["seed"]) + 1000 * (task == "pairing") + int(batch[0]["eval_rank"])
            lm.set_sampler(dec["sampler"], dec["temperature"], dec["k"], dec["p"], seed)
            t0 = time.time()
            outs = lm.generate([p["prompt"] for p in batch], int(dec["max_length"][task]))
            sec = (time.time() - t0) / len(batch)
            for p, o in zip(batch, outs, strict=True):
                cache[p["id"]] = _gen_row(p, o, model_key, model_id, sec)
            write_jsonl(path, list(cache.values()))  # checkpoint after every batch
            log.info("%s %s %d/%d", model_key, task, min(i + bs, len(items)), len(items))
    return [cache[p["id"]] for p in prompts]


def generate_with_gemini(
    client: Any, prompts: list[dict[str, Any]], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Zero-shot Gemini: the same user instruction (no Gemma chat markup)."""
    path = gen_path(cfg, "gemini")
    cache = load_cached(path)
    for p in prompts:
        if p["id"] in cache:
            continue
        t0 = time.time()
        out = client(
            p["user"], max_output_tokens=int(cfg["decoding"]["max_output_tokens"][p["task"]])
        )
        cache[p["id"]] = _gen_row(p, out, "gemini", client.model, time.time() - t0)
        write_jsonl(path, list(cache.values()))
    return [cache[p["id"]] for p in prompts]


def perplexity_summary(per_item: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    """Corpus-level PPL = exp(sum NLL / sum tokens) with an item-bootstrap 95% CI."""
    nll, tok = per_item[:, 0], per_item[:, 1]
    ppl = math.exp(nll.sum() / tok.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(nll), size=(n_boot, len(nll)))
    boots = np.exp(nll[idx].sum(axis=1) / tok[idx].sum(axis=1))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "ppl": ppl,
        "lo": float(lo),
        "hi": float(hi),
        "nll_per_token": float(nll.sum() / tok.sum()),
        "tokens": int(tok.sum()),
        "n": len(nll),
    }


# ------------------------------------------------------------------------------ stages
def _ppl_item_path(cfg: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / cfg["out_dir"] / "perplexity" / f"{key}.csv"


def _load_ppl_items(path: Path, prompts: list[dict]) -> np.ndarray | None:
    """Cached per-item (NLL, tokens) for exactly these prompts, else None."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        rows = {r["id"]: (float(r["nll_sum"]), float(r["tokens"])) for r in csv.DictReader(fh)}
    if not all(p["id"] in rows for p in prompts):
        return None
    return np.array([rows[p["id"]] for p in prompts])


def run_local_models(cfg: dict[str, Any], prompts: list[dict], stages: set[str]) -> dict:
    """Base / fine-tuned Gemma: generations and reference perplexity (both cached per model).

    The model is loaded only if something is missing; run ``--models base`` and
    ``--models finetuned`` separately to keep one model in memory at a time."""
    status: dict[str, str] = {}
    wanted = [m for m in ("base", "finetuned") if m in cfg["models"]]
    adapter = REPO_ROOT / cfg["model"]["adapter"]
    ppl_cfg = cfg["perplexity"]
    ids = {p["id"] for p in prompts}
    lm = None
    for key in wanted:
        if key == "finetuned" and not adapter.exists():
            status[key] = f"not run (adapter missing: {cfg['model']['adapter']})"
            continue
        need_gen = "generate" in stages and not ids <= set(load_cached(gen_path(cfg, key)))
        need_ppl = (
            "perplexity" in stages and _load_ppl_items(_ppl_item_path(cfg, key), prompts) is None
        )
        if not (need_gen or need_ppl):
            continue
        if lm is None or key == "base":  # base must never inherit an adapter
            from recipe_gemma.train.lm import load_lm

            lm = load_lm(cfg["model"])
        if key == "finetuned":
            lm.load_adapter(adapter)
        if need_gen:
            model_id = cfg["model"]["preset"] + (
                f" + LoRA({cfg['model']['adapter']})" if key == "finetuned" else ""
            )
            generate_with_lm(lm, prompts, cfg, key, model_id)
            free_memory()
        if need_ppl:
            t0 = time.time()
            arr = lm.score(
                [p["prompt"] for p in prompts], [p["response"] for p in prompts],
                int(ppl_cfg["seq_len"]), int(ppl_cfg["batch_size"]),
            )  # fmt: skip
            log.info("%s perplexity done in %.0fs", key, time.time() - t0)
            path = _ppl_item_path(cfg, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["id", "task", "nll_sum", "tokens"])
                for p, (nll, tok) in zip(prompts, arr, strict=True):
                    w.writerow([p["id"], p["task"], f"{nll:.6f}", int(tok)])
            free_memory()
        status[key] = "ran"
    return {"status": status, "perplexity": write_perplexity(cfg, prompts)}


def write_perplexity(cfg: dict[str, Any], prompts: list[dict]) -> dict | None:
    """Aggregate cached per-model perplexity items into perplexity.json (+ CI)."""
    b = cfg["bootstrap"]
    by_model: dict[str, Any] = {}
    for key in ("base", "finetuned"):
        arr = _load_ppl_items(_ppl_item_path(cfg, key), prompts)
        if arr is None:
            continue
        entry = {"all": perplexity_summary(arr, b["n"], b["seed"])}
        for task in ("recipe", "pairing"):
            mask = np.array([p["task"] == task for p in prompts])
            if mask.any():
                entry[task] = perplexity_summary(arr[mask], b["n"], b["seed"])
        by_model[key] = entry
    if not by_model:
        return None
    by_model["gemini"] = {
        "status": "not comparable: the Gemini API does not expose token log-probabilities "
        "for scoring arbitrary reference text under the same tokenizer"
    }
    out = {
        "meta": provenance(cfg, model_preset=cfg["model"]["preset"]),
        "seq_len": int(cfg["perplexity"]["seq_len"]),
        "by_model": by_model,
    }
    write_json(REPO_ROOT / cfg["out_dir"] / "perplexity.json", out)
    return out


def compute_metrics(cfg: dict[str, Any], prompts: list[dict], status: dict[str, str]) -> dict:
    """Pairing + structure metrics for every model with cached generations."""
    graph = FlavorGraph.from_file(REPO_ROOT / cfg["graph"])
    stats = CooccurrenceStats.from_file(REPO_ROOT / cfg["cooccur"])
    normalizer = IngredientNormalizer(graph.nodes)
    scorer = PairingScorer(stats, graph)
    ids = {p["id"] for p in prompts}
    b = cfg["bootstrap"]
    summary: dict[str, Any] = {}
    item_rows: list[dict[str, Any]] = []
    for key in cfg["models"]:
        path = gen_path(cfg, key)
        rows = [r for r in load_cached(path).values() if r["id"] in ids]
        if len(rows) < len(prompts):
            status.setdefault(key, f"incomplete ({len(rows)}/{len(prompts)} generations)")
            if not rows:
                continue
        per_task: dict[str, Any] = {}
        for task, keys in (("recipe", RECIPE_KEYS), ("pairing", PAIRING_KEYS)):
            trows = [r for r in rows if r["task"] == task]
            ms = [item_metrics(r, normalizer, scorer) for r in trows]
            for r, m in zip(trows, ms, strict=True):
                item_rows.append({"model": key, "id": r["id"], "task": task, **m})
            per_task[task] = {
                k: bootstrap_ci([m[k] for m in ms], b["n"], seed=b["seed"]) for k in keys
            }
        summary[key] = per_task
    out_dir = REPO_ROOT / cfg["out_dir"]
    result = {
        "meta": provenance(cfg, model_preset=cfg["model"]["preset"]),
        "n_prompts": {t: sum(p["task"] == t for p in prompts) for t in ("recipe", "pairing")},
        "status": status,
        "metrics": summary,
    }
    write_json(out_dir / "eval_metrics.json", result)
    with (out_dir / "eval_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task", "metric", "mean", "ci_lo", "ci_hi", "n"])
        for key, per_task in summary.items():
            for task, mets in per_task.items():
                for name, v in mets.items():
                    w.writerow([key, task, name, v["mean"], v["lo"], v["hi"], v["n"]])
    if item_rows:
        cols = sorted({k for r in item_rows for k in r} - {"model", "id", "task"})
        with (out_dir / "eval_items.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "id", "task", *cols])
            w.writeheader()
            w.writerows(item_rows)
    return result


def run_judge(cfg: dict[str, Any], prompts: list[dict], judge_fn: Any = None) -> dict:
    """LLM-judge pairwise preference (PROXY for human preference)."""
    jcfg = cfg["judge"]
    out_dir = REPO_ROOT / cfg["out_dir"]
    label = "LLM-judge preference (proxy for human preference)"
    base = {"meta": provenance(cfg), "label": label, "judge_model": jcfg["model"]}
    if judge_fn is None:
        if not jcfg.get("enabled", True):
            res = {**base, "status": "disabled"}
            write_json(out_dir / "judge_preference.json", res)
            return res
        if not api_key():
            res = {
                **base,
                "status": NO_KEY,
                "command": (
                    "export GEMINI_API_KEY=... && python -m recipe_gemma.eval "
                    "--config configs/eval.yaml --stages generate,metrics,judge"
                ),
            }
            write_json(out_dir / "judge_preference.json", res)
            return res
        judge_fn = GeminiClient(
            jcfg["model"], temperature=jcfg.get("temperature", 0.0), max_output_tokens=300
        )
    gens = {k: load_cached(gen_path(cfg, k)) for k in cfg["models"]}
    pairs_out: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for x, y in jcfg["pairs"]:
        items = [p for p in prompts if p["id"] in gens.get(x, {}) and p["id"] in gens.get(y, {})]
        if not items:
            pairs_out[f"{x}|{y}"] = {"status": "not run (missing generations)"}
            continue
        recs = []
        for p in items:
            outputs = {x: gens[x][p["id"]]["output"], y: gens[y][p["id"]]["output"]}
            r = judge_item(judge_fn, p["user"], outputs, p["id"], (x, y), int(cfg["seed"]))
            r["task"] = p["task"]
            recs.append(r)
        records.extend(recs)
        b = cfg["bootstrap"]
        pairs_out[f"{x}|{y}"] = {
            "n": len(recs),
            **{c: win_rates([r[c] for r in recs], b["n"], b["seed"]) for c in CRITERIA},
            "by_task": {
                t: win_rates([r["overall"] for r in recs if r["task"] == t], b["n"], b["seed"])
                for t in ("recipe", "pairing")
            },
        }
    res = {**base, "status": "ran", "pairs": pairs_out}
    write_json(out_dir / "judge_preference.json", res)
    write_jsonl(out_dir / "judge_records.jsonl", records)
    return res


def run(cfg: dict[str, Any], stages: set[str]) -> dict[str, Any]:
    """Run the requested stages."""
    prompts = select_prompts(cfg)
    log.info("eval prompts: %d", len(prompts))
    local = run_local_models(cfg, prompts, stages)
    status = dict(local["status"])
    if "gemini" in cfg["models"] and "generate" in stages:
        if api_key():
            dec = cfg["decoding"]
            client = GeminiClient(
                cfg["gemini"]["model"],
                temperature=dec["temperature"],
                top_k=dec["k"],
                seed=dec["seed"],
            )
            generate_with_gemini(client, prompts, cfg)
            status["gemini"] = "ran"
        else:
            status["gemini"] = NO_KEY
    elif "gemini" in cfg["models"] and not gen_path(cfg, "gemini").exists():
        status["gemini"] = NO_KEY if not api_key() else "not generated"
    out: dict[str, Any] = {"status": status}
    if "metrics" in stages:
        out["metrics"] = compute_metrics(cfg, prompts, status)
    if "judge" in stages:
        out["judge"] = run_judge(cfg, prompts)
    return out
