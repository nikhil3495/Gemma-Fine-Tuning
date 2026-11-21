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


def _tiny_cfg(tmp_path: Path, max_steps: int = 4) -> dict:
    cfg = load_config(ROOT / "configs" / "train_tiny.yaml")
    cfg["train"]["max_steps"] = max_steps
    cfg["out"] = {
        "dir": str(tmp_path / "ckpt"),
        "results": str(tmp_path / "res.json"),
        "log_csv": str(tmp_path / "log.csv"),
        "tensorboard": None,
    }
    return cfg


def test_gradient_accumulation_updates_every_k_microbatches() -> None:
    from recipe_gemma.train.trainer import make_optimizer

    lm = TinyLM(seed=3)
    lm.enable_lora(2)
    x, y, sw = lm.encode(["ab"] * 4, ["cdef"] * 4, seq_len=16)
    opt = make_optimizer({"lr": 1e-2, "grad_accum_steps": 4, "weight_decay": 0.0}, 10)
    lm.model.compile(
        optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    b = lm._lora_layers()[0].lora_kernel_b  # LoRA B starts at zero and gets a gradient
    before = np.asarray(b).copy()
    lm.model.fit({k: v[:3] for k, v in x.items()}, y[:3], sample_weight=sw[:3],
                 batch_size=1, epochs=1, verbose=0, shuffle=False)  # fmt: skip
    np.testing.assert_array_equal(np.asarray(b), before)  # 3 micro-batches: no update yet
    lm.model.fit({k: v[3:] for k, v in x.items()}, y[3:], sample_weight=sw[3:],
                 batch_size=1, epochs=1, verbose=0, shuffle=False)  # fmt: skip
    assert not np.array_equal(np.asarray(b), before)  # 4th micro-batch applies the update
    assert int(opt.iterations) == 1  # one optimizer step


def test_checkpoints_and_resume(tmp_path: Path) -> None:
    first = train(_tiny_cfg(tmp_path, max_steps=2))  # "interrupted" after step 2
    ckpt = tmp_path / "ckpt" / "ckpt"
    assert (ckpt / "step_00002" / "adapter.lora.npz").exists() and (ckpt / "best").exists()
    state = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
    assert state["step"] == 2 and state["latest"] == "step_00002"

    res = train(_tiny_cfg(tmp_path, max_steps=4), resume=True)
    assert res["training"]["resumed_from_step"] == 2
    assert res["training"]["steps_run"] == 4
    assert res["training"]["history"]["step"] == [2, 4]
    assert res["val_perplexity"]["base"] == first["val_perplexity"]["base"]  # reused
    assert (ckpt / "step_00004").exists()
    rows = [r for r in (tmp_path / "log.csv").read_text(encoding="utf-8").splitlines() if r]
    assert len(rows) == 3  # header + one row per validation, appended across the resume
