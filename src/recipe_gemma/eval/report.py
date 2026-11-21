"""Generate results/report.md, plots, and the README results table from results/*.json.

python -m recipe_gemma.eval.report --config configs/eval.yaml

Nothing here is hand-typed: every number is read from a results file written by a script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from recipe_gemma.utils.config import REPO_ROOT, load_config

MODELS = ["base", "finetuned", "gemini"]
NAMES = {
    "base": "Base Gemma (zero-shot)",
    "finetuned": "LoRA fine-tuned Gemma",
    "gemini": "Gemini (zero-shot)",
}
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(ci: dict[str, float] | None) -> str:
    if not ci or ci.get("n", 0) == 0 or math.isnan(ci["mean"]):
        return "n/a"
    return f"{100 * ci['mean']:.1f}% [{100 * ci['lo']:.1f}, {100 * ci['hi']:.1f}]"


def _ppl(p: dict[str, float] | None) -> str:
    return "n/a" if not p else f"{p['ppl']:.2f} [{p['lo']:.2f}, {p['hi']:.2f}]"


def _cell(metrics: dict, status: dict, model: str, task: str, key: str) -> str:
    if model not in metrics:
        return status.get(model, "not yet run")
    return _pct(metrics[model].get(task, {}).get(key))


def results_table(res_dir: Path, cfg: dict[str, Any]) -> str:
    """Markdown headline table (rows = metrics, columns = models)."""
    ev = _load(res_dir / "eval_metrics.json")
    ppl = _load(res_dir / "perplexity.json")
    judge = _load(res_dir / "judge_preference.json")
    human = _load(REPO_ROOT / "results" / "human_preference.json")
    if ev is None:
        return (
            "_Evaluation not yet run._ Run: `make eval` (or "
            f"`python -m recipe_gemma.eval --config {cfg.get('_config_path')}`)."
        )
    m, st = ev["metrics"], ev.get("status", {})
    hdr = "| Metric (95% bootstrap CI) | " + " | ".join(NAMES[x] for x in MODELS) + " |"
    sep = "|---|" + "---|" * len(MODELS)
    rows = [hdr, sep]

    def ppl_cell(model: str, task: str) -> str:
        if model == "gemini":
            return "not comparable (no API log-probs)"
        if not ppl or model not in ppl["by_model"]:
            return st.get(model, "not yet run")
        return _ppl(ppl["by_model"][model].get(task))

    rows.append(
        "| Perplexity on held-out references (lower=better) | "
        + " | ".join(ppl_cell(x, "all") for x in MODELS)
        + " |"
    )
    spec = [
        ("Pairing task: suggestions with positive corpus PMI", "pairing", "pmi_positive"),
        ("Pairing task: suggestions sharing >=1 flavor compound", "pairing", "shares_compound"),
        ("Pairing task: flavor-graph coverage", "pairing", "graph_covered"),
        ("Pairing task: exactly 5 suggestions (format)", "pairing", "format_ok"),
        ("Recipe task: ingredient pairs with positive PMI", "recipe", "pmi_positive"),
        ("Recipe task: ingredient pairs sharing a compound", "recipe", "shares_compound"),
        ("Recipe task: well-formed (title/ingredients/steps)", "recipe", "well_formed"),
        ("Recipe task: uses all requested ingredients", "recipe", "all_requested"),
        ("Recipe task: unrecognized-ingredient rate (lower=better)", "recipe", "unrecognized_rate"),
    ]
    for label, task, key in spec:
        rows.append(f"| {label} | " + " | ".join(_cell(m, st, x, task, key) for x in MODELS) + " |")
    n = ev.get("n_prompts", {})
    out = [
        f"Held-out test prompts: **{n.get('recipe', 0)} recipe + {n.get('pairing', 0)} pairing** "
        f"(profile: `{ev['meta'].get('config_path')}`, git `{ev['meta']['git_sha']}`).",
        "",
        *rows,
        "",
        "**LLM-judge preference (proxy for human preference):** " + _judge_summary(judge),
        "",
        "**Human preference:** " + _human_summary(human),
    ]
    return "\n".join(out)


def _judge_summary(judge: dict | None) -> str:
    if not judge:
        return "not yet run."
    if judge.get("status") != "ran":
        return f"{judge.get('status')}. Command: `{judge.get('command', 'make eval')}`."
    parts = []
    for pair, d in judge["pairs"].items():
        if "overall" not in d:
            parts.append(f"{pair}: {d.get('status')}")
            continue
        o = d["overall"]
        x, y = pair.split("|")
        parts.append(
            f"{NAMES[x]} vs {NAMES[y]} (n={d['n']}): win {_pct(o['win'])}, tie {_pct(o['tie'])}, loss {_pct(o['loss'])}"
        )
    return "; ".join(parts)


def _human_summary(human: dict | None) -> str:
    if not human or human.get("status") == "pending annotation":
        return "pending annotation (tool: `python -m annotation.app`; see annotation/README.md)."
    return "see results/human_preference.json."


def plots(res_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    """Loss curve + metric bars with CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out: list[Path] = []
    plot_dir = res_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tr = _load(REPO_ROOT / cfg["train_results"])
    if tr:
        h, every = tr["training"]["history"], tr["training"]["eval_every"]
        steps = [every * (i + 1) for i in range(len(h["loss"]))]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(steps, h["loss"], marker="o", label="train loss (window mean)")
        ax.plot(steps, h["val_loss"], marker="s", label="val loss")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("token NLL (response only)")
        ax.set_title(f"LoRA fine-tuning: {tr['config']['run_name']} run")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = plot_dir / "loss_curve.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)
    ev = _load(res_dir / "eval_metrics.json")
    if ev:
        keys = [("pairing", "pmi_positive", "Pairing: PMI>0"), ("pairing", "shares_compound", "Pairing: shares compound"),
                ("recipe", "pmi_positive", "Recipe: PMI>0"), ("recipe", "well_formed", "Recipe: well-formed"),
                ("recipe", "all_requested", "Recipe: all requested")]  # fmt: skip
        present = [x for x in MODELS if x in ev["metrics"]]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        width = 0.8 / max(1, len(present))
        for j, model in enumerate(present):
            means, errs = [], [[], []]
            for task, key, _ in keys:
                ci = ev["metrics"][model][task][key]
                mean = 0 if math.isnan(ci["mean"]) else ci["mean"]
                means.append(100 * mean)
                errs[0].append(
                    100 * max(0, mean - (ci["lo"] if not math.isnan(ci["lo"]) else mean))
                )
                errs[1].append(
                    100 * max(0, (ci["hi"] if not math.isnan(ci["hi"]) else mean) - mean)
                )
            xs = [i + j * width for i in range(len(keys))]
            ax.bar(xs, means, width, yerr=errs, capsize=3, label=NAMES[model])
        ax.set_xticks([i + width * (len(present) - 1) / 2 for i in range(len(keys))])
        ax.set_xticklabels([k[2] for k in keys], rotation=15, fontsize=8)
        ax.set_ylabel("% (95% bootstrap CI)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = plot_dir / "metrics_bars.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)
    return out


def training_summary(cfg: dict[str, Any]) -> str:
    """Markdown block summarizing the training run from results/train_*.json."""
    tr = _load(REPO_ROOT / cfg["train_results"])
    if not tr:
        return f"_Training not yet run._ Run: `python -m recipe_gemma.train --config configs/train_{cfg['profile']}.yaml`."
    t, mdl, d, vp = tr["training"], tr["model"], tr["data"], tr["val_perplexity"]
    return "\n".join([
        f"- Run: **{tr['config']['run_name']}** (git `{tr['meta']['git_sha']}`, config hash `{tr['meta']['config_hash']}`, {tr['meta']['timestamp_utc']})",
        f"- Model: `{mdl['preset']}`: {mdl['total_params']:,} params, LoRA rank {mdl['lora_rank']} -> {mdl['trainable_params']:,} trainable ({mdl['trainable_pct']:.2f}%)",
        f"- Data: {d['train_examples']:,} train / {d['val_examples']} val examples ({d['train_tasks']}); dropped for length: {d['dropped_overlength']}",
        f"- Steps: {t['steps_run']} optimizer steps (best val loss at {t['best_step']}), micro-batch {t.get('batch_size', '?')} x grad-accum {t.get('grad_accum_steps', 1)} = effective batch {t.get('effective_batch', '?')}" + (f"; resumed once from the step-{t['resumed_from_step']} checkpoint" if t.get('resumed_from_step') else ""),
        f"- Compute: CPU only ({tr['meta']['platform']}); {t['train_seconds'] / 60:.0f} min wall-clock ({t['train_seconds'] / max(1, t['steps_run']):.1f} s/step incl. host stalls); peak RSS {tr.get('memory', {}).get('peak_rss_gb', 'n/a')} GB",
        f"- Val loss: best {t['best_val_loss']:.4f}; val perplexity **{vp['base']['ppl']:.2f} (base) -> {vp['fine_tuned']['ppl']:.2f} (LoRA)** on {vp['base']['n']} val examples",
    ])  # fmt: skip


def write_report(cfg: dict[str, Any]) -> Path:
    """Write report.md (and update README between markers if present)."""
    res_dir = REPO_ROOT / cfg["out_dir"]
    res_dir.mkdir(parents=True, exist_ok=True)
    pngs = plots(res_dir, cfg)
    table = results_table(res_dir, cfg)
    train_md = training_summary(cfg)
    label = (
        "SMOKE run (CPU, Gemma 3 270M, small data): pipeline proof, not a final benchmark"
        if cfg["profile"] == "smoke"
        else "FULL run"
    )
    img = "\n".join(f"![{p.stem}]({p.relative_to(res_dir).as_posix()})" for p in pngs)
    md = f"# Benchmark report: {label}\n\n_Auto-generated by `python -m recipe_gemma.eval.report`; do not edit by hand._\n\n## Training\n\n{train_md}\n\n## Results\n\n{table}\n\n{img}\n\nMetric definitions: [docs/EVAL_PROTOCOL.md](../docs/EVAL_PROTOCOL.md).\n"
    report = res_dir / "report.md"
    report.write_text(md, encoding="utf-8")
    _write_summary_csv(res_dir)
    readme = REPO_ROOT / "README.md"
    if readme.exists() and cfg.get("update_readme", True):
        text = readme.read_text(encoding="utf-8")
        if README_START in text and README_END in text:
            block = f"{README_START}\n_{label}. Auto-generated from `results/*.json` by `make eval`._\n\n{train_md}\n\n{table}\n{README_END}"
            text = re.sub(
                re.escape(README_START) + ".*?" + re.escape(README_END),
                lambda _m: block,
                text,
                flags=re.S,
            )
            readme.write_text(text, encoding="utf-8")
    return report


def _write_summary_csv(res_dir: Path) -> None:
    ppl = _load(res_dir / "perplexity.json")
    if not ppl:
        return
    with (res_dir / "perplexity.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "subset", "ppl", "ci_lo", "ci_hi", "tokens", "n"])
        for model, d in ppl["by_model"].items():
            for subset, v in d.items():
                if isinstance(v, dict):
                    w.writerow([model, subset, v["ppl"], v["lo"], v["hi"], v["tokens"], v["n"]])


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()
    print(write_report(load_config(args.config)))


if __name__ == "__main__":
    main()
