import json
import shutil
import sys
from pathlib import Path

from recipe_gemma.serve.demo import Demo

ROOT = Path(__file__).resolve().parents[1]


def _make_root(tmp: Path) -> Path:
    art = tmp / "data" / "artifacts"
    art.mkdir(parents=True)
    for f in ("flavor_graph.json.gz", "cooccur_train.json.gz"):
        shutil.copy(ROOT / "data" / "artifacts" / f, art / f)
    gen = tmp / "results" / "generations"
    gen.mkdir(parents=True)
    rows = {
        "base": [
            {"id": "r1", "task": "recipe", "user": "Create a recipe using: tomato, basil.",
             "output": "BASE tomato", "meta": {"requested": ["tomato", "basil"]}},
            {"id": "r2", "task": "recipe", "user": "Create a recipe using: chicken, rice.",
             "output": "BASE chicken", "meta": {"requested": ["chicken", "rice"]}},
            {"id": "anchor:garlic", "task": "pairing", "user": "Suggest 5 for garlic",
             "output": "1. Onion", "meta": {"anchor": "garlic"}},
        ]
    }  # fmt: skip
    rows["finetuned"] = [{**r, "output": r["output"].replace("BASE", "FT")} for r in rows["base"]]
    for k, rs in rows.items():
        (gen / f"{k}.jsonl").write_text("\n".join(json.dumps(r) for r in rs), encoding="utf-8")
    return tmp


def test_pairings_real_artifacts(tmp_path: Path) -> None:
    demo = Demo(_make_root(tmp_path))
    note, rows = demo.pairings("Tomatoes")
    assert "tomato" in note and len(rows) == 8
    assert all(r[2] > 0 for r in rows)  # positive NPMI only
    note, rows = demo.pairings("zzzz unknown")
    assert rows == [] and "No corpus evidence" in note


def test_closest_cached_and_pairing_cache(tmp_path: Path) -> None:
    demo = Demo(_make_root(tmp_path))
    hit = demo.closest_cached(["chicken", "rice", "pea"])
    assert hit["id"] == "r2" and abs(hit["overlap"] - 2 / 3) < 1e-9
    assert demo.cached_pairing_outputs("garlic") == {"base": "1. Onion", "finetuned": "1. Onion"}
    assert demo.recipe_examples()[0].startswith("Create a recipe")


def test_live_load_failure_falls_back(tmp_path: Path) -> None:
    demo = Demo(_make_root(tmp_path))
    assert demo.try_load_live("definitely-not-a-preset", None) is False
    assert demo.live_error


def test_space_staging_contains_runtime_files(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import deploy_space

    out = deploy_space.stage(tmp_path)
    for rel in ("app.py", "requirements.txt", "README.md", "src/recipe_gemma/serve/demo.py",
                "data/artifacts/flavor_graph.json.gz"):  # fmt: skip
        assert (out / rel).exists(), rel
    assert "sdk: gradio" in (out / "README.md").read_text(encoding="utf-8")
