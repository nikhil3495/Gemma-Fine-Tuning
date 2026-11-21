"""Tests for the blind pairwise annotation tool and the human-preference scorer (no gradio)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from annotation.core import (
    CSV_COLUMNS,
    Comparison,
    append_record,
    available_pairs,
    build_comparisons,
    done_keys,
    load_generations,
    make_record,
    read_annotations,
    remaining,
)

FIXTURES = Path(__file__).parent / "fixtures" / "generations"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_scorer() -> ModuleType:
    path = REPO_ROOT / "scripts" / "score_human_prefs.py"
    spec = importlib.util.spec_from_file_location("score_human_prefs", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


scorer = _load_scorer()


@pytest.fixture
def gens() -> dict[str, Any]:
    return load_generations(FIXTURES)


def test_load_skips_missing_models(gens: dict[str, Any]) -> None:
    assert set(gens) == {"base", "finetuned"}  # gemini file absent
    assert available_pairs(gens) == [("finetuned", "base")]


def test_only_shared_items_are_compared(gens: dict[str, Any]) -> None:
    comps = build_comparisons(gens, "alice")
    assert sorted(c.item_id for c in comps) == ["p1", "r1", "r2"]  # p2 only in finetuned
    assert all(c.pair == "finetuned|base" for c in comps)
    assert all({c.model_a, c.model_b} == {"finetuned", "base"} for c in comps)


def test_order_is_deterministic_per_annotator(gens: dict[str, Any]) -> None:
    assert build_comparisons(gens, "alice") == build_comparisons(gens, "alice")


def test_both_positions_occur_and_order_is_shuffled() -> None:
    many = {
        m: {f"i{k}": {"id": f"i{k}", "task": "recipe", "user": "u", "output": m} for k in range(40)}
        for m in ("base", "finetuned", "gemini")
    }
    comps = build_comparisons(many, "bob")
    assert len(comps) == 120
    ft_base = [c for c in comps if c.pair == "finetuned|base"]
    assert any(c.model_a == "finetuned" for c in ft_base)
    assert any(c.model_b == "finetuned" for c in ft_base)
    # outputs follow their model after swapping
    assert all(c.output_a == c.model_a and c.output_b == c.model_b for c in comps)
    # item order is shuffled and differs between annotators
    keys = [c.key for c in comps]
    assert keys != sorted(keys)
    assert keys != [c.key for c in build_comparisons(many, "carol")]


def test_public_view_is_blind(gens: dict[str, Any]) -> None:
    for comp in build_comparisons(gens, "alice"):
        view = comp.public_view()
        assert set(view) == {"task", "instruction", "response_a", "response_b"}
        blob = json.dumps(view).lower()
        for name in ("finetuned", "base", "gemini", "|"):
            assert name not in blob


def _comp() -> Comparison:
    return Comparison("r1", "recipe", "u", "finetuned|base", "base", "finetuned", "x", "y")


def test_make_record_maps_choice_to_model() -> None:
    comp = _comp()
    assert make_record(comp, "a", "A")["winner_model"] == "base"
    assert make_record(comp, "a", "B")["winner_model"] == "finetuned"
    assert make_record(comp, "a", "tie")["winner_model"] == "tie"
    rec = make_record(comp, " a ", "B", {"coherence": "4"}, {"feasibility": 5}, " ok ")
    assert list(rec) == CSV_COLUMNS
    assert rec["annotator_id"] == "a" and rec["comment"] == "ok"
    assert rec["a_coherence"] == 4 and rec["b_feasibility"] == 5 and rec["a_feasibility"] == ""
    with pytest.raises(ValueError):
        make_record(comp, "a", "C")
    with pytest.raises(ValueError):
        make_record(comp, "a", "A", {"coherence": 9})
    with pytest.raises(ValueError):
        make_record(comp, "  ", "A")


def test_csv_append_and_resume(gens: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "sub" / "annotations.csv"
    comps = build_comparisons(gens, "alice")
    append_record(out, make_record(comps[0], "alice", "A"))
    append_record(out, make_record(comps[1], "alice", "tie"))
    append_record(out, make_record(comps[0], "bob", "B"))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == CSV_COLUMNS  # header written once
    assert len(lines) == 4
    assert len(read_annotations(out)) == 3

    done = done_keys(out, "alice")
    assert done == {comps[0].key, comps[1].key}
    assert remaining(build_comparisons(gens, "alice"), done) == comps[2:]
    assert len(remaining(build_comparisons(gens, "bob"), done_keys(out, "bob"))) == 2
    assert done_keys(tmp_path / "missing.csv", "alice") == set()


def _write_synthetic(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})


def _row(ann: str, item: str, task: str, winner: str) -> dict[str, str]:
    return {
        "annotator_id": ann,
        "item_id": item,
        "task": task,
        "pair": "finetuned|base",
        "model_a": "finetuned",
        "model_b": "base",
        "winner_model": winner,
    }


def test_scoring_known_rates_and_perfect_agreement(tmp_path: Path) -> None:
    # 10 items: finetuned wins 6, ties 2, loses 2; two annotators label identically.
    winners = ["finetuned"] * 6 + ["tie"] * 2 + ["base"] * 2
    tasks = ["recipe", "pairing"] * 5
    rows = [
        _row(ann, f"i{k}", tasks[k], w) for ann in ("alice", "bob") for k, w in enumerate(winners)
    ]
    ann = tmp_path / "annotations.csv"
    _write_synthetic(ann, rows)
    out_json, out_csv = tmp_path / "hp.json", tmp_path / "hp.csv"
    payload = scorer.score(ann, out_json, out_csv, n_boot=200)

    assert payload["status"] == "ok"
    overall = payload["preferences"]["finetuned|base"]["all"]
    assert overall["n"] == 20
    assert overall["win"]["mean"] == pytest.approx(0.6)
    assert overall["tie"]["mean"] == pytest.approx(0.2)
    assert overall["loss"]["mean"] == pytest.approx(0.2)
    assert overall["win"]["lo"] <= 0.6 <= overall["win"]["hi"]
    assert set(payload["preferences"]["finetuned|base"]) == {"all", "recipe", "pairing"}

    agr = payload["agreement"]
    assert agr["n_items_multi_rated"] == 10
    assert agr["mean_pairwise_cohen_kappa"] == pytest.approx(1.0)
    assert agr["fleiss_kappa"] == pytest.approx(1.0)

    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["status"] == "ok" and "git_sha" in saved["provenance"]
    with out_csv.open(encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert {r["task"] for r in table} == {"all", "recipe", "pairing"}


def test_agreement_three_raters_unequal() -> None:
    rows = [
        _row("a", "i1", "recipe", "finetuned"),
        _row("b", "i1", "recipe", "finetuned"),
        _row("c", "i1", "recipe", "base"),
        _row("a", "i2", "recipe", "base"),
        _row("b", "i2", "recipe", "tie"),
    ]
    agr = scorer.agreement(rows)
    assert agr["n_annotators"] == 3
    assert agr["fleiss_kappa"] is None
    assert agr["method"].startswith("mean_pairwise_cohen_kappa")
    assert agr["mean_pairwise_cohen_kappa"] is not None


def test_dedupe_keeps_latest() -> None:
    rows = [_row("a", "i1", "recipe", "base"), _row("a", "i1", "recipe", "finetuned")]
    assert [r["winner_model"] for r in scorer.dedupe(rows)] == ["finetuned"]


@pytest.mark.parametrize("make_file", [False, True])
def test_pending_annotation(tmp_path: Path, make_file: bool) -> None:
    ann = tmp_path / "annotations.csv"
    if make_file:
        _write_synthetic(ann, [])  # header only
    out_json, out_csv = tmp_path / "hp.json", tmp_path / "hp.csv"
    code = scorer.main(
        ["--annotations", str(ann), "--out-json", str(out_json), "--out-csv", str(out_csv)]
    )
    assert code == 0
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["status"] == "pending annotation"
    assert saved["launch_command"] == scorer.LAUNCH_COMMAND
    assert "annotation.app" in saved["launch_command"]
    assert "provenance" in saved
    assert not out_csv.exists()
