"""LoRA fine-tuning loop (Keras 3, JAX backend), memory-safe and resumable.

python -m recipe_gemma.train --config configs/train_smoke.yaml [--resume] [--dry-run]

* Gradient accumulation (``train.grad_accum_steps``) via the Keras optimizer: micro-batches of
  ``batch_size`` are averaged into one optimizer step (effective batch = batch * accum).
* Every ``eval_every`` optimizer steps: validation, LoRA-adapter checkpoint
  (``<out>/ckpt/step_NNNNN``), best-adapter copy (``<out>/ckpt/best``) and ``state.json``.
  ``--resume`` restarts from the latest checkpoint (data-stream position and LR schedule are
  restored, and the optimizer state is reloaded when the checkpoint has it). EarlyStopping
  patience restarts on resume.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
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
from recipe_gemma.utils.memory import available_gb, free_memory, peak_rss_gb, rss_gb

log = logging.getLogger(__name__)
NO_DECAY = ["bias", "norm", "scale", "gamma", "beta"]


class BatchStream(keras.utils.PyDataset):
    """Infinite, seeded, epoch-shuffled mini-batch stream over in-memory arrays.

    ``len()`` is the number of micro-batches between validations; the global position persists
    across Keras "epochs" (and can be restored via ``offset`` when resuming).
    """

    def __init__(
        self,
        x: dict[str, np.ndarray],
        y: np.ndarray,
        sw: np.ndarray,
        batch_size: int,
        steps: int,
        seed: int,
        offset: int = 0,
    ) -> None:
        super().__init__()
        self.x, self.y, self.sw = x, y, sw
        self.n = len(y)
        self.batch_size = batch_size
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.offset = offset
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
    """One-line progress records (optimizer steps) that are easy to tail/monitor."""

    def __init__(self, every: int, accum: int, start_step: int = 0) -> None:
        super().__init__()
        self.every, self.accum = every, accum
        self.micro = start_step * accum
        self.start_micro = self.micro
        self.t0 = time.time()

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        self.micro += 1
        if self.micro % (self.every * self.accum) == 0:
            step = self.micro // self.accum
            loss = (logs or {}).get("loss", float("nan"))
            rate = (time.time() - self.t0) / max(1, (self.micro - self.start_micro) / self.accum)
            print(
                f"[train] step={step} loss={loss:.4f} sec_per_step={rate:.2f} "
                f"rss_gb={rss_gb():.2f} peak_gb={peak_rss_gb():.2f}",
                flush=True,
            )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        print(
            f"[eval] step={self.micro // self.accum} loss={logs.get('loss', float('nan')):.4f} "
            f"val_loss={logs.get('val_loss', float('nan')):.4f}",
            flush=True,
        )


class Checkpointer(keras.callbacks.Callback):
    """Save the LoRA adapter + resumable state after every validation."""

    def __init__(self, lm: CausalLM, ckpt_dir: Path, eval_every: int, state: dict) -> None:
        super().__init__()
        self.lm, self.ckpt_dir, self.eval_every, self.state = lm, ckpt_dir, eval_every, state
        self._t = time.time()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        step = (epoch + 1) * self.eval_every  # Keras epoch numbers continue after resume
        step_dir = self.ckpt_dir / f"step_{step:05d}"
        if step_dir.exists():
            shutil.rmtree(step_dir)
        self.lm.save_adapter(step_dir)
        opt_vars = self.model.optimizer.variables
        np.savez(step_dir / "optimizer.npz", *[np.asarray(v) for v in opt_vars])
        hist = self.state.setdefault("history", {})
        for k, v in logs.items():
            hist.setdefault(k, []).append(float(v))
        hist.setdefault("step", []).append(step)
        val = float(logs.get("val_loss", math.inf))
        best = self.state.get("best")
        if best is None or val < best["val_loss"]:
            best_dir = self.ckpt_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(step_dir, best_dir)
            self.state["best"] = {"step": step, "val_loss": val}
        now = time.time()
        self.state["train_seconds"] = self.state.get("train_seconds", 0.0) + (now - self._t)
        self._t = now
        self.state["step"] = step
        self.state["latest"] = step_dir.name
        self.state["peak_rss_gb"] = max(self.state.get("peak_rss_gb", 0.0), peak_rss_gb())
        write_json(self.ckpt_dir / "state.json", self.state)


def make_optimizer(tcfg: dict[str, Any], total_steps: int) -> keras.optimizers.Optimizer:
    """AdamW + linear warmup + cosine decay; no weight decay on norms/biases.

    ``total_steps``/``warmup_steps`` are optimizer steps; the Keras schedule is evaluated on
    micro-batch iterations, so both are scaled by ``grad_accum_steps``.
    """
    accum = int(tcfg.get("grad_accum_steps", 1) or 1)
    warmup = int(tcfg.get("warmup_steps", 0)) * accum
    total = total_steps * accum
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0 if warmup else float(tcfg["lr"]),
        decay_steps=max(1, total - warmup),
        alpha=float(tcfg.get("min_lr_ratio", 0.1)),
        warmup_target=float(tcfg["lr"]) if warmup else None,
        warmup_steps=warmup,
    )
    opt = keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        clipnorm=tcfg.get("grad_clip"),
        gradient_accumulation_steps=accum if accum > 1 else None,
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


def _adapter_file(d: Path) -> Path:
    return next(p for p in sorted(d.iterdir()) if p.name.startswith("adapter.lora."))


def train(cfg: dict[str, Any], resume: bool = False, overwrite: bool = False) -> dict[str, Any]:
    """Run (or resume) LoRA fine-tuning; returns (and writes) the training results record."""
    seed = int(cfg["seed"])
    set_seed(seed)
    tcfg, dcfg = cfg["train"], cfg["data"]
    out_dir = REPO_ROOT / cfg["out"]["dir"]
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if tcfg.get("mixed_precision"):
        keras.mixed_precision.set_global_policy(tcfg["mixed_precision"])
    state_path = ckpt_dir / "state.json"
    state: dict[str, Any] = {}
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_hash") != config_hash(_resume_key(cfg)):
            log.warning("config changed since the checkpoint (memory settings may differ)")
        log.info("resuming from %s (step %s)", state.get("latest"), state.get("step"))
    elif state_path.exists() and not overwrite:
        raise FileExistsError(
            f"{ckpt_dir} holds checkpoints (step {json.loads(state_path.read_text())['step']}); "
            "pass --resume to continue or --overwrite to start over"
        )
    elif ckpt_dir.exists():  # explicit --overwrite (or no state): start fresh
        shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True)
    start_step = int(state.get("step", 0))
    log.info("available RAM before load: %.2f GB", available_gb())

    train_rows = read_jsonl(REPO_ROOT / dcfg["train"])
    val_rows = read_jsonl(REPO_ROOT / dcfg["val"])
    if dcfg.get("max_train"):
        train_rows = train_rows[: dcfg["max_train"]]

    t_load = time.time()
    lm = load_lm(cfg["model"])
    load_sec = time.time() - t_load
    seq_len = int(tcfg["seq_len"])
    eval_bs = int(tcfg.get("eval_batch_size", 1))

    n_before = {"train": len(train_rows), "val": len(val_rows)}
    dropped = {"train": 0, "val": 0}
    if dcfg.get("drop_overlength", True):
        train_rows, dropped["train"] = filter_overlength(lm, train_rows, seq_len)
        val_rows, dropped["val"] = filter_overlength(lm, val_rows, seq_len)
    if dcfg.get("max_val"):
        val_rows = val_rows[: dcfg["max_val"]]
    log.info("train=%d val=%d dropped=%s", len(train_rows), len(val_rows), dropped)
    if not train_rows or not val_rows:
        raise ValueError(
            f"No examples left after seq_len={seq_len} filtering (dropped={dropped}); "
            "increase train.seq_len or set data.drop_overlength=false."
        )

    # Baseline (pre-LoRA) validation perplexity, measured with the same code path.
    if "base_val" in state:
        base_val = state["base_val"]
    else:
        base_val = perplexity(lm, val_rows, seq_len, eval_bs)
        state["base_val"] = base_val
    log.info("base val ppl=%.3f (rss %.2f GB)", base_val["ppl"], rss_gb())
    free_memory()  # drop scoring buffers / compiled graphs before training

    rank = int(cfg["lora"]["rank"])
    lm.enable_lora(rank)
    if state.get("latest"):
        lm.load_adapter(_adapter_file(ckpt_dir / state["latest"]))
    total_params, trainable = lm.total_params(), lm.trainable_params()

    x, y, sw = lm.encode(
        [r["prompt"] for r in train_rows], [r["response"] for r in train_rows], seq_len
    )
    xv, yv, swv = lm.encode(
        [r["prompt"] for r in val_rows], [r["response"] for r in val_rows], seq_len
    )

    accum = int(tcfg.get("grad_accum_steps", 1) or 1)
    max_steps, eval_every = int(tcfg["max_steps"]), int(tcfg["eval_every"])
    if max_steps % eval_every:
        raise ValueError("train.max_steps must be a multiple of train.eval_every")
    epochs = max_steps // eval_every
    initial_epoch = start_step // eval_every
    micro_per_epoch = eval_every * accum
    stream = BatchStream(
        x, y, sw, int(tcfg["batch_size"]), micro_per_epoch, seed,
        offset=initial_epoch * micro_per_epoch,
    )  # fmt: skip

    model = lm.model
    if hasattr(lm, "_detach"):
        lm._detach()
    optimizer = make_optimizer(tcfg, max_steps)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="mean_with_sample_weight"
        ),
        weighted_metrics=["accuracy"],
    )
    if start_step:
        optimizer.build(model.trainable_variables)
        opt_file = ckpt_dir / state["latest"] / "optimizer.npz"
        if opt_file.exists():  # full optimizer state: Adam moments + iteration counter
            saved = np.load(opt_file)
            for i, var in enumerate(optimizer.variables):
                var.assign(saved[f"arr_{i}"])
        else:  # older adapter-only checkpoint: restore the LR-schedule position only
            log.warning("no optimizer state in checkpoint; Adam moments restart from zero")
            optimizer._iterations.assign(start_step * accum)

    log_csv = REPO_ROOT / cfg["out"]["log_csv"]
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    state["config_hash"] = config_hash(_resume_key(cfg))
    callbacks: list[keras.callbacks.Callback] = [
        StepLogger(int(tcfg.get("log_every", 10)), accum, start_step),
        keras.callbacks.CSVLogger(str(log_csv), append=bool(start_step)),
        Checkpointer(lm, ckpt_dir, eval_every, state),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(tcfg.get("early_stopping_patience", 3))
        ),
    ]
    if cfg["out"].get("tensorboard"):
        try:
            import tensorboard  # noqa: F401  (needed by the Keras TensorBoard callback)

            callbacks.append(
                keras.callbacks.TensorBoard(str(REPO_ROOT / cfg["out"]["tensorboard"]))
            )
        except ImportError:
            log.warning("tensorboard not installed; TensorBoard logging skipped")

    if initial_epoch < epochs:
        model.fit(
            stream,
            validation_data=(xv, yv, swv),
            validation_batch_size=eval_bs,
            initial_epoch=initial_epoch,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )
    if hasattr(lm, "_attach"):
        lm._attach()

    # Restore the best checkpoint (by val loss), export it, and score it.
    best_dir = ckpt_dir / "best"
    lm.load_adapter(_adapter_file(best_dir))
    for f in best_dir.iterdir():
        shutil.copy(f, out_dir / f.name)
    adapter_path = _adapter_file(out_dir)
    free_memory()
    tuned_val = perplexity(lm, val_rows, seq_len, eval_bs)
    log.info("fine-tuned val ppl=%.3f", tuned_val["ppl"])

    hist = state["history"]
    steps_run = int(state["step"])
    train_sec = float(state.get("train_seconds", 0.0))
    peak = max(float(state.get("peak_rss_gb", 0.0)), peak_rss_gb())
    result = {
        "meta": provenance(cfg, model_preset=cfg["model"]["preset"], run_name=cfg["run_name"]),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "data": {
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "before_length_filter": n_before,
            "dropped_overlength": dropped,
            "train_tasks": _count(r["task"] for r in train_rows),
        },
        "model": {
            "preset": cfg["model"]["preset"],
            "total_params": total_params,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total_params,
            "lora_rank": rank,
            "load_sec": round(load_sec, 1),
        },
        "training": {
            "steps_run": steps_run,
            "max_steps": max_steps,
            "early_stopped": steps_run < max_steps,
            "resumed_from_step": start_step or None,
            "batch_size": int(tcfg["batch_size"]),
            "grad_accum_steps": accum,
            "effective_batch": int(tcfg["batch_size"]) * accum,
            "best_step": state["best"]["step"],
            "best_val_loss": state["best"]["val_loss"],
            "final_train_loss": hist["loss"][-1],
            "train_seconds": round(train_sec, 1),
            # cumulative wall-clock over all sessions / all steps (includes any host stalls)
            "sec_per_step": round(train_sec / max(1, steps_run), 2),
            "history": hist,
            "eval_every": eval_every,
        },
        "memory": {
            "peak_rss_gb": round(peak, 2),
            "xla_allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        "val_perplexity": {"base": base_val, "fine_tuned": tuned_val},
        "adapter": {"path": _rel(adapter_path), "best_step": state["best"]["step"]},
    }
    write_json(REPO_ROOT / cfg["out"]["results"], result)
    log.info("wrote %s (peak RSS %.2f GB)", cfg["out"]["results"], peak)
    return result


def _resume_key(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config subset that must match for a meaningful resume."""
    return {"seed": cfg["seed"], "data": cfg["data"], "model": cfg["model"], "lora": cfg["lora"]}


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


__all__ = ["BatchStream", "Checkpointer", "filter_overlength", "make_optimizer", "train"]
