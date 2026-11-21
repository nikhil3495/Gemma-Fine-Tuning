"""LoRA fine-tuning loop (Keras 3, JAX backend).

python -m recipe_gemma.train --config configs/train_smoke.yaml
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras

from recipe_gemma.train.lm import CausalLM, load_lm
from recipe_gemma.utils.config import (
    REPO_ROOT,
    config_hash,
    provenance,
    read_jsonl,
    set_seed,
    write_json,
)

log = logging.getLogger(__name__)
NO_DECAY = ["bias", "norm", "scale", "gamma", "beta"]


class BatchStream(keras.utils.PyDataset):
    """Infinite, seeded, epoch-shuffled mini-batch stream over in-memory arrays.

    ``len()`` is the number of steps between validations; the global position persists across
    Keras "epochs", so ``epochs * steps_per_eval`` = total optimizer steps with true data epochs.
    """

    def __init__(
        self,
        x: dict[str, np.ndarray],
        y: np.ndarray,
        sw: np.ndarray,
        batch_size: int,
        steps: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.x, self.y, self.sw = x, y, sw
        self.n = len(y)
        self.batch_size = batch_size
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.offset = 0
        self.data_epoch = 0
        self.perm = self.rng.permutation(self.n)

    def __len__(self) -> int:
        return self.steps

    def _indices(self, global_step: int) -> np.ndarray:
        per_epoch = max(1, self.n // self.batch_size)
        epoch, pos = divmod(global_step, per_epoch)
        while epoch > self.data_epoch:  # deterministic reshuffle each data epoch
            self.perm = self.rng.permutation(self.n)
            self.data_epoch += 1
        return self.perm[pos * self.batch_size : (pos + 1) * self.batch_size]

    def __getitem__(self, idx: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        ix = self._indices(self.offset + idx)
        return {k: v[ix] for k, v in self.x.items()}, self.y[ix], self.sw[ix]

    def on_epoch_end(self) -> None:
        self.offset += self.steps


class StepLogger(keras.callbacks.Callback):
    """Prints a one-line progress record every ``every`` steps (easy to tail/monitor)."""

    def __init__(self, every: int = 10) -> None:
        super().__init__()
        self.every = every
        self.step = 0
        self.t0 = time.time()

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        self.step += 1
        if self.step % self.every == 0:
            loss = (logs or {}).get("loss", float("nan"))
            rate = (time.time() - self.t0) / self.step
            print(f"[train] step={self.step} loss={loss:.4f} sec_per_step={rate:.2f}", flush=True)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        print(
            f"[eval] step={self.step} loss={logs.get('loss', float('nan')):.4f} "
            f"val_loss={logs.get('val_loss', float('nan')):.4f}",
            flush=True,
        )


def make_optimizer(tcfg: dict[str, Any], total_steps: int) -> keras.optimizers.Optimizer:
    """AdamW + linear warmup + cosine decay; no weight decay on norms/biases."""
    warmup = int(tcfg.get("warmup_steps", 0))
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0 if warmup else float(tcfg["lr"]),
        decay_steps=max(1, total_steps - warmup),
        alpha=float(tcfg.get("min_lr_ratio", 0.1)),
        warmup_target=float(tcfg["lr"]) if warmup else None,
        warmup_steps=warmup,
    )
    opt = keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        clipnorm=tcfg.get("grad_clip"),
    )
    opt.exclude_from_weight_decay(var_names=NO_DECAY)
    return opt


def filter_overlength(
    lm: CausalLM, rows: list[dict[str, Any]], seq_len: int
) -> tuple[list[dict[str, Any]], int]:
    """Drop examples whose prompt+response does not fit in ``seq_len`` (keeps full targets,
    so the model learns to finish recipes and emit the end-of-turn token)."""
    kept = [r for r in rows if lm.token_length(r["prompt"] + r["response"]) + 3 <= seq_len]
    return kept, len(rows) - len(kept)


def perplexity(lm: CausalLM, rows: list[dict[str, Any]], seq_len: int, batch: int) -> dict:
    """Token-level perplexity of responses given prompts."""
    if not rows:
        return {"ppl": math.nan, "nll_per_token": math.nan, "tokens": 0, "n": 0}
    s = lm.score([r["prompt"] for r in rows], [r["response"] for r in rows], seq_len, batch)
    nll, n = float(s[:, 0].sum()), float(s[:, 1].sum())
    return {"ppl": math.exp(nll / n), "nll_per_token": nll / n, "tokens": int(n), "n": len(rows)}


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run LoRA fine-tuning; returns (and writes) the training results record."""
    seed = int(cfg["seed"])
    set_seed(seed)
    tcfg, dcfg = cfg["train"], cfg["data"]
    out_dir = REPO_ROOT / cfg["out"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if tcfg.get("mixed_precision"):
        keras.mixed_precision.set_global_policy(tcfg["mixed_precision"])

    train_rows = read_jsonl(REPO_ROOT / dcfg["train"])
    val_rows = read_jsonl(REPO_ROOT / dcfg["val"])
    if dcfg.get("max_train"):
        train_rows = train_rows[: dcfg["max_train"]]
    if dcfg.get("max_val"):
        val_rows = val_rows[: dcfg["max_val"]]

    t_load = time.time()
    lm = load_lm(cfg["model"])
    load_sec = time.time() - t_load
    seq_len = int(tcfg["seq_len"])

    dropped = {"train": 0, "val": 0}
    if dcfg.get("drop_overlength", True):
        train_rows, dropped["train"] = filter_overlength(lm, train_rows, seq_len)
        val_rows, dropped["val"] = filter_overlength(lm, val_rows, seq_len)
    log.info("train=%d val=%d dropped=%s", len(train_rows), len(val_rows), dropped)
    if not train_rows or not val_rows:
        raise ValueError(
            f"No examples left after seq_len={seq_len} filtering (dropped={dropped}); "
            "increase train.seq_len or set data.drop_overlength=false."
        )

    # Baseline (pre-LoRA) validation perplexity, measured with the same code path.
    base_val = perplexity(lm, val_rows, seq_len, int(tcfg.get("eval_batch_size", 2)))
    log.info("base val ppl=%.3f", base_val["ppl"])

    lm.enable_lora(int(cfg["lora"]["rank"]))
    total_params, trainable = lm.total_params(), lm.trainable_params()

    x, y, sw = lm.encode(
        [r["prompt"] for r in train_rows], [r["response"] for r in train_rows], seq_len
    )
    xv, yv, swv = lm.encode(
        [r["prompt"] for r in val_rows], [r["response"] for r in val_rows], seq_len
    )

    max_steps, eval_every = int(tcfg["max_steps"]), int(tcfg["eval_every"])
    epochs = math.ceil(max_steps / eval_every)
    stream = BatchStream(x, y, sw, int(tcfg["batch_size"]), eval_every, seed)

    model = lm.model
    if hasattr(lm, "_detach"):
        lm._detach()
    model.compile(
        optimizer=make_optimizer(tcfg, epochs * eval_every),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="mean_with_sample_weight"
        ),
        weighted_metrics=["accuracy"],
    )

    log_csv = REPO_ROOT / cfg["out"]["log_csv"]
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    callbacks: list[keras.callbacks.Callback] = [
        StepLogger(int(tcfg.get("log_every", 10))),
        keras.callbacks.CSVLogger(str(log_csv)),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(tcfg.get("early_stopping_patience", 3)),
            restore_best_weights=True,
        ),
    ]
    if cfg["out"].get("tensorboard"):
        try:
            import tensorflow  # noqa: F401  (Keras' TensorBoard callback writes via TF)

            callbacks.append(
                keras.callbacks.TensorBoard(str(REPO_ROOT / cfg["out"]["tensorboard"]))
            )
        except ImportError:
            log.warning("tensorflow not installed; TensorBoard logging skipped")

    t0 = time.time()
    hist = model.fit(
        stream,
        validation_data=(xv, yv, swv),
        validation_batch_size=int(tcfg.get("eval_batch_size", 2)),
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )
    train_sec = time.time() - t0
    if hasattr(lm, "_attach"):
        lm._attach()
    steps_run = len(hist.history["loss"]) * eval_every

    adapter_path = lm.save_adapter(out_dir)
    tuned_val = perplexity(lm, val_rows, seq_len, int(tcfg.get("eval_batch_size", 2)))
    log.info("fine-tuned val ppl=%.3f", tuned_val["ppl"])

    best_epoch = int(np.argmin(hist.history["val_loss"]))
    result = {
        "meta": provenance(cfg, model_preset=cfg["model"]["preset"], run_name=cfg["run_name"]),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "data": {
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "dropped_overlength": dropped,
            "train_tasks": _count(r["task"] for r in train_rows),
        },
        "model": {
            "preset": cfg["model"]["preset"],
            "total_params": total_params,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total_params,
            "lora_rank": int(cfg["lora"]["rank"]),
            "load_sec": round(load_sec, 1),
        },
        "training": {
            "steps_run": steps_run,
            "max_steps": max_steps,
            "early_stopped": steps_run < epochs * eval_every,
            "best_step": (best_epoch + 1) * eval_every,
            "best_val_loss": float(hist.history["val_loss"][best_epoch]),
            "final_train_loss": float(hist.history["loss"][-1]),
            "train_seconds": round(train_sec, 1),
            "sec_per_step": round(train_sec / max(1, steps_run), 2),
            "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
            "eval_every": eval_every,
        },
        "val_perplexity": {"base": base_val, "fine_tuned": tuned_val},
        "adapter": {"path": _rel(adapter_path)},
    }
    write_json(REPO_ROOT / cfg["out"]["results"], result)
    log.info("wrote %s", cfg["out"]["results"])
    return result


def _rel(path: Any) -> str:
    """Repo-relative POSIX path when possible (absolute otherwise)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


__all__ = ["BatchStream", "config_hash", "filter_overlength", "make_optimizer", "train"]
