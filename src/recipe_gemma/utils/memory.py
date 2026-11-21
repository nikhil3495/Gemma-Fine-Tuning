"""Process memory helpers (peak / current RSS) and memory-safe JAX settings."""

from __future__ import annotations

import gc
import os
import sys


def configure_low_memory_jax() -> None:
    """Make XLA allocate on demand and return memory to the OS. Call before importing jax."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


def rss_gb() -> float:
    """Current resident set size of this process in GiB (nan if unavailable)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except ImportError:
        return float("nan")


def peak_rss_gb() -> float:
    """Peak resident set size of this process in GiB (nan if unavailable)."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak is not None:
            return peak / 2**30
    except ImportError:
        pass
    if sys.platform != "win32":
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / (2**30 if sys.platform == "darwin" else 2**20)
    return float("nan")


def available_gb() -> float:
    """System memory available to new allocations in GiB (nan if unavailable)."""
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except ImportError:
        return float("nan")


def free_memory() -> None:
    """Drop Python garbage and JAX compilation caches."""
    gc.collect()
    try:
        import jax

        jax.clear_caches()
    except ImportError:  # pragma: no cover
        pass
    gc.collect()
