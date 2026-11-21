"""Pure logic for blind pairwise A/B annotation (no Gradio import, fully testable).

A *comparison* is one prompt (``item_id``) answered by two models (a *pair*, e.g.
``finetuned|base``). For each annotator the left/right placement of the two models is decided
by a stable hash of ``(annotator_id, item_id, pair)`` and the comparison order is shuffled with
a seed derived from ``annotator_id``, so the same annotator always sees the same sequence
(resumable) while different annotators see different ones.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODELS: tuple[str, ...] = ("base", "finetuned", "gemini")
#: Pairs compared, written ``(first, second)``; scores are reported from ``first``'s view.
PAIRS: tuple[tuple[str, str], ...] = (
    ("finetuned", "base"),
    ("finetuned", "gemini"),
    ("gemini", "base"),
)
CRITERIA: tuple[str, ...] = (
    "coherence",
    "feasibility",
    "flavor_harmony",
    "instruction_following",
)
CHOICES: tuple[str, ...] = ("A", "tie", "B")
CSV_COLUMNS: list[str] = [
    "timestamp_utc",
    "annotator_id",
    "item_id",
    "task",
    "pair",
    "model_a",
    "model_b",
    "choice",
    "winner_model",
    *[f"a_{c}" for c in CRITERIA],
    *[f"b_{c}" for c in CRITERIA],
    "comment",
]

_CSV_LOCK = threading.Lock()


@dataclass(frozen=True)
class Comparison:
    """One blind A/B comparison. ``model_a``/``model_b`` must never reach the UI."""

    item_id: str
    task: str
    instruction: str
    pair: str
    model_a: str
    model_b: str
    output_a: str
    output_b: str

    @property
    def key(self) -> tuple[str, str]:
        """Identity of the comparison independent of A/B placement."""
        return (self.item_id, self.pair)

    def public_view(self) -> dict[str, str]:
        """Fields safe to show to annotators (no model identities, no pair name)."""
        return {
            "task": self.task,
            "instruction": self.instruction,
            "response_a": self.output_a,
            "response_b": self.output_b,
        }


def pair_name(first: str, second: str) -> str:
    """Canonical pair label, e.g. ``"finetuned|base"``."""
    return f"{first}|{second}"


def stable_hash(*parts: str) -> int:
    """Process-independent 64-bit hash (Python's ``hash`` is salted per process)."""
    blob = "\x1f".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def load_generations(generations_dir: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load ``{model}.jsonl`` files that exist. Returns ``{model: {item_id: row}}``."""
    root = Path(generations_dir)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODELS:
        path = root / f"{model}.jsonl"
        if not path.is_file():
            continue
        rows: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    rows[str(row["id"])] = row
        if rows:
            out[model] = rows
    return out


def available_pairs(generations: dict[str, dict[str, dict[str, Any]]]) -> list[tuple[str, str]]:
    """Pairs from :data:`PAIRS` whose two models both have generations."""
    return [(a, b) for a, b in PAIRS if a in generations and b in generations]


def build_comparisons(
    generations: dict[str, dict[str, dict[str, Any]]], annotator_id: str
) -> list[Comparison]:
    """All comparisons for ``annotator_id`` with deterministic A/B order and item order."""
    comps: list[Comparison] = []
    for first, second in available_pairs(generations):
        pair = pair_name(first, second)
        shared = sorted(set(generations[first]) & set(generations[second]))
        for item_id in shared:
            row_1, row_2 = generations[first][item_id], generations[second][item_id]
            swap = stable_hash(annotator_id, item_id, pair) % 2 == 1
            (m_a, r_a), (m_b, r_b) = (
                ((second, row_2), (first, row_1)) if swap else ((first, row_1), (second, row_2))
            )
            comps.append(
                Comparison(
                    item_id=item_id,
                    task=str(row_1.get("task", "")),
                    instruction=str(row_1.get("user", "")),
                    pair=pair,
                    model_a=m_a,
                    model_b=m_b,
                    output_a=str(r_a.get("output", "")),
                    output_b=str(r_b.get("output", "")),
                )
            )
    rng = random.Random(stable_hash("order", annotator_id))
    rng.shuffle(comps)
    return comps


def winner_model(comp: Comparison, choice: str) -> str:
    """Map an A/B/tie choice to the true model name (or ``"tie"``)."""
    if choice == "A":
        return comp.model_a
    if choice == "B":
        return comp.model_b
    if choice == "tie":
        return "tie"
    raise ValueError(f"choice must be one of {CHOICES}, got {choice!r}")


def _rating(value: Any) -> int | str:
    """Validate an optional 1-5 rating; empty/None becomes ``""``."""
    if value is None or value == "":
        return ""
    r = int(value)
    if not 1 <= r <= 5:
        raise ValueError(f"ratings must be 1-5, got {value!r}")
    return r


def make_record(
    comp: Comparison,
    annotator_id: str,
    choice: str,
    ratings_a: dict[str, Any] | None = None,
    ratings_b: dict[str, Any] | None = None,
    comment: str = "",
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build one CSV row (includes the hidden model identities, for scoring only)."""
    annotator_id = annotator_id.strip()
    if not annotator_id:
        raise ValueError("annotator_id is required")
    ratings_a, ratings_b = ratings_a or {}, ratings_b or {}
    rec: dict[str, Any] = {
        "timestamp_utc": timestamp_utc or datetime.now(UTC).isoformat(timespec="seconds"),
        "annotator_id": annotator_id,
        "item_id": comp.item_id,
        "task": comp.task,
        "pair": comp.pair,
        "model_a": comp.model_a,
        "model_b": comp.model_b,
        "choice": choice,
        "winner_model": winner_model(comp, choice),
    }
    for c in CRITERIA:
        rec[f"a_{c}"] = _rating(ratings_a.get(c))
        rec[f"b_{c}"] = _rating(ratings_b.get(c))
    rec["comment"] = (comment or "").strip()
    return {k: rec[k] for k in CSV_COLUMNS}


def append_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append a row to the annotations CSV, creating it (with header) if absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CSV_LOCK:
        new = not path.is_file() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(record)


def read_annotations(path: str | Path) -> list[dict[str, str]]:
    """Read all rows of the annotations CSV (empty list if missing)."""
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def done_keys(path: str | Path, annotator_id: str) -> set[tuple[str, str]]:
    """``(item_id, pair)`` keys already labeled by ``annotator_id``."""
    aid = annotator_id.strip()
    return {
        (row["item_id"], row["pair"])
        for row in read_annotations(path)
        if row.get("annotator_id", "").strip() == aid
    }


def remaining(comparisons: list[Comparison], done: set[tuple[str, str]]) -> list[Comparison]:
    """Comparisons not yet labeled, preserving order (resume support)."""
    return [c for c in comparisons if c.key not in done]
