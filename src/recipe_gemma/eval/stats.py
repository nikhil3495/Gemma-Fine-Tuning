"""Statistics helpers: bootstrap confidence intervals, win rates and inter-annotator agreement."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Mean with a percentile bootstrap (1 - alpha) CI. NaNs are dropped first."""
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def win_rates(
    outcomes: Sequence[str], n_boot: int = 2000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Win/tie/loss rates (outcomes in {"win","tie","loss"}) each with a bootstrap CI."""
    out = {}
    for k in ("win", "tie", "loss"):
        out[k] = bootstrap_ci([1.0 if o == k else 0.0 for o in outcomes], n_boot, seed=seed)
    return out


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for two raters over the same items."""
    if len(a) != len(b) or not a:
        raise ValueError("raters must label the same non-empty item list")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(table: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa; ``table[i][j]`` = #raters putting item i in category j (same #raters/item)."""
    m = np.asarray(table, dtype=float)
    if m.ndim != 2 or m.shape[0] == 0:
        return math.nan
    n_r = m.sum(axis=1)
    if not np.all(n_r == n_r[0]) or n_r[0] < 2:
        raise ValueError("every item needs the same number (>=2) of ratings")
    r = n_r[0]
    p_j = m.sum(axis=0) / (m.shape[0] * r)
    p_i = (np.square(m).sum(axis=1) - r) / (r * (r - 1))
    p_bar, pe = p_i.mean(), np.square(p_j).sum()
    return 1.0 if pe == 1 else float((p_bar - pe) / (1 - pe))
