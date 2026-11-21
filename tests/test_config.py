from pathlib import Path

from recipe_gemma.utils.config import config_hash, deep_merge, load_config, provenance


def test_deep_merge_nested() -> None:
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    out = deep_merge(base, {"b": {"d": 4}, "e": 5})
    assert out == {"a": 1, "b": {"c": 2, "d": 4}, "e": 5}
    assert base["b"]["d"] == 3  # base not mutated


def test_config_hash_ignores_private_keys_and_order() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1, "_config_path": "x"})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_load_config_extends(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("x: 1\ny: {z: 2}\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ny: {w: 3}\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["x"] == 1 and cfg["y"] == {"z": 2, "w": 3}


def test_provenance_fields() -> None:
    meta = provenance({"a": 1}, model_preset="tiny")
    for key in ("git_sha", "timestamp_utc", "config_hash", "model_preset"):
        assert key in meta
