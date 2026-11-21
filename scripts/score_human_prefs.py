"""Score human pairwise preferences collected by ``annotation/app.py``.

Outputs win/tie/loss rates per model pair (from the first model's perspective, e.g.
``finetuned|base`` -> finetuned wins) with 95% bootstrap CIs, overall and by task, plus
inter-annotator agreement (mean pairwise Cohen's kappa; Fleiss' kappa when every multiply
labeled item has the same number of raters).

Usage::

    python scripts/score_human_prefs.py --annotations annotation/data/annotations.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from recipe_gemma.eval.stats import cohen_kappa, fleiss_kappa, win_rates
from recipe_gemma.utils.config import provenance, write_json

LAUNCH_COMMAND = "python -m annotation.app --generations-dir results/generations"
SCORE_COMMAND = "python scripts/score_human_prefs.py"
OUTCOMES: tuple[str, ...] = ("win", "tie", "loss")


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """Read the annotations CSV; missing file -> ``[]``."""
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("pair")]


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the latest judgement per (annotator, item, pair)."""
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in rows:
        latest[(r["annotator_id"].strip(), r["item_id"], r["pair"])] = r
    return list(latest.values())


def outcome(row: dict[str, str]) -> str:
    """Outcome from the first model of ``row['pair']``'s perspective."""
    first, second = row["pair"].split("|")
    w = row["winner_model"]
    if w == first:
        return "win"
    if w == second:
        return "loss"
    if w == "tie":
        return "tie"
    raise ValueError(f"winner_model {w!r} not in pair {row['pair']!r}")


def _clean(x: Any) -> Any:
    """Replace NaN with None so the JSON stays valid."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def preference_table(
    rows: list[dict[str, str]], n_boot: int = 2000, seed: int = 0
) -> dict[str, dict[str, Any]]:
    """``{pair: {"all": rates, "<task>": rates}}`` with win/tie/loss rates and CIs."""
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        o = outcome(r)
        groups[r["pair"]]["all"].append(o)
        groups[r["pair"]][r.get("task") or "unknown"].append(o)
    out: dict[str, dict[str, Any]] = {}
    for pair in sorted(groups):
        out[pair] = {}
        for task in ["all", *sorted(t for t in groups[pair] if t != "all")]:
            outs = groups[pair][task]
            out[pair][task] = {"n": len(outs), **win_rates(outs, n_boot=n_boot, seed=seed)}
    return out


def agreement(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Inter-annotator agreement on (item, pair) units labeled by >= 2 annotators.

    Labels are the pair-relative outcome (win/tie/loss), so A/B placement is irrelevant.
    """
    labels: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for r in rows:
        labels[(r["item_id"], r["pair"])][r["annotator_id"].strip()] = outcome(r)
    multi = {k: v for k, v in labels.items() if len(v) >= 2}
    annotators = sorted({a for v in multi.values() for a in v})
    res: dict[str, Any] = {
        "n_items_multi_rated": len(multi),
        "n_annotators": len(annotators),
        "mean_pairwise_cohen_kappa": None,
        "pairwise_cohen_kappa": {},
        "fleiss_kappa": None,
        "method": None,
    }
    if not multi:
        res["note"] = "no item labeled by >= 2 annotators"
        return res

    kappas: list[float] = []
    for a, b in itertools.combinations(annotators, 2):
        shared = sorted(k for k, v in multi.items() if a in v and b in v)
        if not shared:
            continue
        k = cohen_kappa([multi[s][a] for s in shared], [multi[s][b] for s in shared])
        res["pairwise_cohen_kappa"][f"{a}|{b}"] = {"kappa": k, "n_items": len(shared)}
        kappas.append(k)
    if kappas:
        res["mean_pairwise_cohen_kappa"] = sum(kappas) / len(kappas)

    raters_per_item = {len(v) for v in multi.values()}
    if len(annotators) == 2:
        res["method"] = "cohen_kappa (2 annotators)"
    if len(raters_per_item) == 1:
        table = [[list(v.values()).count(c) for c in OUTCOMES] for v in multi.values()]
        res["fleiss_kappa"] = fleiss_kappa(table)
        if len(annotators) > 2:
            res["method"] = "fleiss_kappa (equal raters per item)"
    elif len(annotators) > 2:
        res["method"] = "mean_pairwise_cohen_kappa (unequal raters per item)"
    return res


def write_csv(path: str | Path, table: dict[str, dict[str, Any]]) -> None:
    """Flat CSV: one row per (pair, task)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pair", "task", "n"] + [f"{o}{s}" for o in OUTCOMES for s in ("", "_lo", "_hi")]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for pair, by_task in table.items():
            for task, st in by_task.items():
                vals = [st[o][k] for o in OUTCOMES for k in ("mean", "lo", "hi")]
                w.writerow([pair, task, st["n"], *(f"{v:.4f}" for v in vals)])


def score(
    annotations: str | Path,
    out_json: str | Path,
    out_csv: str | Path,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute and write results. Returns the JSON payload."""
    rows = dedupe(read_rows(annotations))
    meta = provenance(annotations=str(Path(annotations).as_posix()), n_boot=n_boot, seed=seed)
    if not rows:
        payload: dict[str, Any] = {
            "status": "pending annotation",
            "message": "No human annotations yet. Launch the tool, annotate, then re-score.",
            "launch_command": LAUNCH_COMMAND,
            "score_command": SCORE_COMMAND,
            "provenance": meta,
        }
        write_json(out_json, payload)
        return payload
    table = preference_table(rows, n_boot=n_boot, seed=seed)
    payload = {
        "status": "ok",
        "n_judgements": len(rows),
        "n_annotators": len({r["annotator_id"].strip() for r in rows}),
        "perspective": "rates are for the first model in each pair (e.g. finetuned|base)",
        "preferences": table,
        "agreement": agreement(rows),
        "note": "Human preferences are the primary quality signal; LLM-judge scores are a proxy.",
        "provenance": meta,
    }
    payload = _clean(payload)
    write_json(out_json, payload)
    write_csv(out_csv, table)
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Score blind pairwise human preferences.")
    p.add_argument("--annotations", default="annotation/data/annotations.csv")
    p.add_argument("--out-json", default="results/human_preference.json")
    p.add_argument("--out-csv", default="results/human_preference.csv")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    payload = score(args.annotations, args.out_json, args.out_csv, args.n_boot, args.seed)
    if payload["status"] != "ok":
        print(f"pending annotation - launch with: {LAUNCH_COMMAND}")
    else:
        for pair, by_task in payload["preferences"].items():
            st = by_task["all"]
            print(
                f"{pair}: win {st['win']['mean']:.2f} tie {st['tie']['mean']:.2f} "
                f"loss {st['loss']['mean']:.2f} (n={st['n']})"
            )
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
