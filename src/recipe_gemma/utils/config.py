"""YAML config loading, hashing and run provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML config. Supports a top-level ``extends: other.yaml`` (shallow-deep merge)."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = deep_merge(base, cfg)
    cfg.setdefault("_config_path", str(path.as_posix()))
    return cfg


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable 12-char SHA-256 of a config (keys starting with ``_`` are ignored)."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def git_sha() -> str:
    """Return the current git commit SHA (short) or ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance(cfg: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    """Metadata recorded in every results file."""
    meta: dict[str, Any] = {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if cfg is not None:
        meta["config_hash"] = config_hash(cfg)
        meta["config_path"] = cfg.get("_config_path")
    meta.update(extra)
    return meta


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if importable) keras."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # keras is an optional dependency
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def write_json(path: str | os.PathLike[str], obj: Any) -> None:
    """Write JSON as UTF-8 (Windows defaults to cp1252)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a JSONL file."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL (UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
