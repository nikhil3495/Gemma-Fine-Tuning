"""End-to-end training/inference on the tiny random Keras LM (offline, CPU, no secrets)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from recipe_gemma.train.lm import TinyLM, load_lm  # noqa: E402
from recipe_gemma.train.trainer import BatchStream, train  # noqa: E402
from recipe_gemma.utils.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_encode_masks_prompt_and_shifts_labels() -> None:
    lm = TinyLM()
    x, y, sw = lm.encode(["ab"], ["cd"], seq_len=8)
    ids = x["token_ids"][0]
    # [BOS a b c d EOS PAD PAD] -> labels are next tokens, weight only where label is response
    assert ids[0] == lm.tok.bos_id and list(ids[1:5]) == [97, 98, 99, 100]
    assert list(y[0][:5]) == [97, 98, 99, 100, lm.tok.eos_id]
    assert list(sw[0][:6]) == [0, 0, 1, 1, 1, 0]
    assert x["padding_mask"][0].sum() == 6


def test_batch_stream_covers_epoch_without_repeats() -> None:
    n = 10
    x = {"token_ids": np.arange(n)[:, None]}
    s = BatchStream(x, np.arange(n), np.ones(n), batch_size=2, steps=5, seed=0)
    seen = np.concatenate([s[i][1] for i in range(len(s))])
    assert sorted(seen.tolist()) == list(range(n))
    s.on_epoch_end()  # next "epoch" continues with a new permutation
    seen2 = np.concatenate([s[i][1] for i in range(len(s))])
    assert sorted(seen2.tolist()) == list(range(n))


def test_lora_freezes_base_and_roundtrips(tmp_path: Path) -> None:
    lm = TinyLM(seed=1)
    total = lm.total_params()
    lm.enable_lora(2)
    assert 0 < lm.trainable_params() < total
    layer = lm._lora_layers()[0]
    layer.lora_kernel_b.assign(np.ones(layer.lora_kernel_b.shape) * 0.5)
    path = lm.save_adapter(tmp_path)
    lm2 = TinyLM(seed=1)
    lm2.load_adapter(path)
    np.testing.assert_allclose(
        np.asarray(lm2._lora_layers()[0].lora_kernel_b), np.asarray(layer.lora_kernel_b)
    )


def test_score_and_generate() -> None:
    lm = load_lm({"preset": "tiny-random"})
    s = lm.score(["hello"], ["world"], seq_len=32)
    assert s.shape == (1, 2) and s[0, 1] == 6  # 5 bytes + EOS
    # random model ~ uniform over 259 tokens -> per-token NLL near log(259)
    assert abs(s[0, 0] / s[0, 1] - math.log(259)) < 1.5
    lm.set_sampler("top_k", 0.7, 5, 1.0, 0)
    out = lm.generate(["hi"], max_length=12)
    assert isinstance(out[0], str)


def test_train_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(ROOT / "configs" / "train_tiny.yaml")
    cfg["out"] = {
        "dir": str(tmp_path / "ckpt"),
        "results": str(tmp_path / "res.json"),
        "log_csv": str(tmp_path / "log.csv"),
        "tensorboard": None,
    }
    res = train(cfg)
    assert res["training"]["steps_run"] == 4
    assert res["model"]["trainable_params"] < res["model"]["total_params"]
    assert (tmp_path / "ckpt" / "adapter.lora.npz").exists()
    saved = json.loads((tmp_path / "res.json").read_text(encoding="utf-8"))
    assert saved["meta"]["config_hash"] and saved["meta"]["model_preset"] == "tiny-random"
    assert math.isfinite(saved["val_perplexity"]["fine_tuned"]["ppl"])
    assert (tmp_path / "log.csv").read_text(encoding="utf-8").startswith("epoch")
