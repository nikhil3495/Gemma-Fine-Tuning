"""A thin, backend-agnostic causal-LM wrapper used by training, evaluation and serving.

Two implementations share one interface:

* :class:`GemmaLM` wraps a ``keras_hub`` Gemma (3) causal LM loaded from a preset
  (Kaggle ``gemma3_instruct_270m`` or ``hf://...``).
* :class:`TinyLM` is a tiny randomly-initialised pure-Keras causal transformer with a byte
  tokenizer. It needs no downloads, no secrets and no TensorFlow, so CI exercises the exact
  same training / scoring / generation / LoRA code paths on CPU.

All tokenization happens eagerly in Python and training runs on NumPy arrays. (keras_hub's
tf.data tokenization path requires ``tensorflow-text``, which has no Windows wheels.)
"""

from __future__ import annotations

import json
import os
import types
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras

Arrays = tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]


class CausalLM(ABC):
    """Common interface for the models in this project."""

    name: str = "causal-lm"

    @property
    @abstractmethod
    def model(self) -> keras.Model:
        """Keras model mapping ``{"token_ids", "padding_mask"}`` -> logits ``[B, T, V]``."""

    @abstractmethod
    def encode(self, prompts: list[str], responses: list[str], seq_len: int) -> Arrays:
        """Tokenize prompt/response pairs -> ``(x, y, sample_weight)``.

        ``y`` is ``token_ids`` shifted left by one; ``sample_weight`` is 1 only on response
        tokens (the prompt is masked out of the loss)."""

    @abstractmethod
    def token_length(self, text: str) -> int:
        """Number of tokens in ``text`` (no special tokens)."""

    @abstractmethod
    def generate(self, prompts: list[str], max_length: int) -> list[str]:
        """Generate completions (prompt stripped). ``max_length`` counts prompt + output tokens."""

    @abstractmethod
    def enable_lora(self, rank: int) -> None:
        """Freeze the base weights and add LoRA adapters of the given rank."""

    @abstractmethod
    def save_adapter(self, out_dir: str | os.PathLike[str]) -> Path:
        """Save the LoRA adapter weights only. Returns the written file."""

    @abstractmethod
    def load_adapter(self, path: str | os.PathLike[str]) -> None:
        """Load LoRA adapter weights saved with :meth:`save_adapter`."""

    @abstractmethod
    def set_sampler(self, sampler: str, temperature: float, k: int, p: float, seed: int) -> None:
        """Configure decoding."""

    # ------------------------------------------------------------------ shared helpers
    def trainable_params(self) -> int:
        """Number of trainable parameters."""
        return int(sum(np.prod(w.shape) for w in self.model.trainable_weights))

    def total_params(self) -> int:
        """Total parameter count."""
        return int(self.model.count_params())

    def nll(self, x: dict[str, np.ndarray], y: np.ndarray, sw: np.ndarray) -> np.ndarray:
        """Per-example ``(sum NLL over weighted tokens, n weighted tokens)`` -> ``[B, 2]``."""
        logits = self.model(x, training=False)
        logits = keras.ops.cast(logits, "float32")
        logp = keras.ops.log_softmax(logits, axis=-1)
        tok = keras.ops.take_along_axis(logp, keras.ops.expand_dims(y, -1), axis=-1)[..., 0]
        w = keras.ops.cast(sw, "float32")
        nll = -keras.ops.sum(tok * w, axis=1)
        n = keras.ops.sum(w, axis=1)
        return np.stack([np.asarray(nll), np.asarray(n)], axis=1)

    def score(
        self, prompts: list[str], responses: list[str], seq_len: int, batch_size: int = 2
    ) -> np.ndarray:
        """``[N, 2]`` array of (NLL sum, token count) for each response given its prompt."""
        out = []
        for i in range(0, len(prompts), batch_size):
            x, y, sw = self.encode(
                prompts[i : i + batch_size], responses[i : i + batch_size], seq_len
            )
            out.append(self.nll(x, y, sw))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 2))


# =========================================================================== Gemma
class GemmaLM(CausalLM):
    """keras_hub Gemma / Gemma 3 causal LM."""

    def __init__(self, preset: str, dtype: str | None = None) -> None:
        import keras_hub

        kwargs: dict[str, Any] = {}
        if dtype:
            kwargs["dtype"] = dtype
        cls = keras_hub.models.CausalLM
        self.lm = cls.from_preset(preset, **kwargs)
        self.name = preset
        self._pre = self.lm.preprocessor
        self._lora_rank: int | None = None

    @property
    def model(self) -> keras.Model:
        return self.lm

    def _detach(self) -> None:
        """Run as a plain Keras model on token arrays.

        keras_hub's ``PipelineModel`` overrides ``fit``/``evaluate``/``predict`` to rebuild a
        grain/tf.data preprocessing pipeline, which rejects ``PyDataset`` inputs (and needs
        tensorflow-text). We bind the backend trainer's plain methods onto the instance, so
        internal ``self.evaluate`` calls made by ``fit`` use them too."""
        self.lm.preprocessor = None
        for name in ("fit", "evaluate", "predict"):
            base = next(
                c
                for c in type(self.lm).__mro__
                if c.__module__.startswith("keras.src") and name in c.__dict__
            )
            setattr(self.lm, name, types.MethodType(base.__dict__[name], self.lm))

    def _attach(self) -> None:
        self.lm.preprocessor = self._pre
        for name in ("fit", "evaluate", "predict"):
            self.lm.__dict__.pop(name, None)

    def encode(self, prompts: list[str], responses: list[str], seq_len: int) -> Arrays:
        self._pre.sequence_length = seq_len
        x, y, sw = self._pre({"prompts": list(prompts), "responses": list(responses)})
        x = {k: np.asarray(v) for k, v in x.items()}
        return x, np.asarray(y), np.asarray(sw).astype("float32")

    def token_length(self, text: str) -> int:
        return len(self._pre.tokenizer(text))

    def nll(self, x: dict[str, np.ndarray], y: np.ndarray, sw: np.ndarray) -> np.ndarray:
        self._detach()
        try:
            return super().nll(x, y, sw)
        finally:
            self._attach()

    def generate(self, prompts: list[str], max_length: int) -> list[str]:
        self._attach()
        out = self.lm.generate(list(prompts), max_length=max_length, strip_prompt=True)
        return [str(o) for o in out]

    def set_sampler(self, sampler: str, temperature: float, k: int, p: float, seed: int) -> None:
        import keras_hub

        samplers = {
            "greedy": lambda: keras_hub.samplers.GreedySampler(),
            "top_k": lambda: keras_hub.samplers.TopKSampler(
                k=k, temperature=temperature, seed=seed
            ),
            "top_p": lambda: keras_hub.samplers.TopPSampler(
                p=p, temperature=temperature, seed=seed
            ),
        }
        self.lm.compile(sampler=samplers[sampler]())

    def enable_lora(self, rank: int) -> None:
        self.lm.backbone.enable_lora(rank=rank)
        self._lora_rank = rank

    def save_adapter(self, out_dir: str | os.PathLike[str]) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "adapter.lora.h5"
        self.lm.backbone.save_lora_weights(str(path))
        return path

    def load_adapter(self, path: str | os.PathLike[str]) -> None:
        self.lm.backbone.load_lora_weights(str(path))


# =========================================================================== Tiny (CI)
class ByteTokenizer:
    """UTF-8 byte tokenizer: ids 0..255 are bytes; 256=PAD, 257=BOS, 258=EOS."""

    pad_id, bos_id, eos_id, vocab_size = 256, 257, 258, 259

    def encode(self, text: str) -> list[int]:
        """Text -> byte ids."""
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        """Byte ids -> text (special ids dropped)."""
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="ignore")


class TinyLM(CausalLM):
    """Tiny random causal LM for CI (no downloads, no secrets)."""

    def __init__(self, seed: int = 0) -> None:
        keras.utils.set_random_seed(seed)
        self.tok = ByteTokenizer()
        self._model = self._build()
        self.name = "tiny-random"
        self._lora_rank: int | None = None
        self._sampler = ("greedy", 1.0, 1, 1.0, seed)

    @staticmethod
    def _build() -> keras.Model:
        ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        pm = keras.Input(shape=(None,), dtype="int32", name="padding_mask")
        dim, heads, vocab = 32, 2, ByteTokenizer.vocab_size
        tok = keras.layers.Embedding(vocab, dim, name="token_embedding")(ids)
        h = TinyPositional(512, dim, name="position_embedding")(tok)
        a = keras.layers.MultiHeadAttention(heads, dim // heads, name="attention")(
            h, h, use_causal_mask=True
        )
        h = keras.layers.LayerNormalization(name="ln1")(h + a)
        f = keras.layers.Dense(dim * 2, activation="gelu", name="ffw_up")(h)
        f = keras.layers.Dense(dim, name="ffw_down")(f)
        h = keras.layers.LayerNormalization(name="ln2")(h + f)
        logits = keras.layers.Dense(vocab, name="lm_head")(h)
        return keras.Model({"token_ids": ids, "padding_mask": pm}, logits, name="tiny_causal_lm")

    @property
    def model(self) -> keras.Model:
        return self._model

    def encode(self, prompts: list[str], responses: list[str], seq_len: int) -> Arrays:
        b = len(prompts)
        # One extra position so labels can be the next token.
        full = np.full((b, seq_len + 1), self.tok.pad_id, dtype="int32")
        weight = np.zeros((b, seq_len + 1), dtype="float32")
        for i, (p, r) in enumerate(zip(prompts, responses, strict=True)):
            pi = [self.tok.bos_id, *self.tok.encode(p)]
            ri = [*self.tok.encode(r), self.tok.eos_id]
            ids = (pi + ri)[: seq_len + 1]
            full[i, : len(ids)] = ids
            weight[i, len(pi) : len(ids)] = 1.0  # weight on positions predicting response
        x_ids = full[:, :-1]
        y = full[:, 1:]
        sw = weight[:, 1:]
        mask = (x_ids != self.tok.pad_id).astype("int32")
        return {"token_ids": x_ids, "padding_mask": mask}, y, sw

    def token_length(self, text: str) -> int:
        return len(self.tok.encode(text))

    def set_sampler(self, sampler: str, temperature: float, k: int, p: float, seed: int) -> None:
        self._sampler = (sampler, temperature, k, p, seed)

    def generate(self, prompts: list[str], max_length: int) -> list[str]:
        sampler, temp, k, _p, seed = self._sampler
        rng = np.random.default_rng(seed)
        outs = []
        for prompt in prompts:
            ids = [self.tok.bos_id, *self.tok.encode(prompt)]
            start = len(ids)
            while len(ids) < max_length:
                arr = np.array([ids], dtype="int32")
                logits = np.asarray(
                    self._model({"token_ids": arr, "padding_mask": np.ones_like(arr)})
                )[0, -1]
                if sampler == "greedy":
                    nxt = int(np.argmax(logits))
                else:
                    z = logits / max(temp, 1e-6)
                    top = np.argsort(z)[-k:]
                    pz = np.exp(z[top] - z[top].max())
                    nxt = int(rng.choice(top, p=pz / pz.sum()))
                if nxt == self.tok.eos_id:
                    break
                ids.append(nxt)
            outs.append(self.tok.decode(ids[start:]))
        return outs

    def _lora_layers(self) -> list[keras.layers.Layer]:
        layers = []
        for layer in self._model._flatten_layers(include_self=False):
            if layer.name in {"query", "value"} and hasattr(layer, "enable_lora"):
                layers.append(layer)
        return layers

    def enable_lora(self, rank: int) -> None:
        for layer in self._model._flatten_layers(include_self=False):
            layer.trainable = False
        for layer in self._lora_layers():
            layer.trainable = True
            layer.enable_lora(rank)
        self._lora_rank = rank

    def save_adapter(self, out_dir: str | os.PathLike[str]) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for i, layer in enumerate(self._lora_layers()):
            arrays[f"{i}/lora_kernel_a"] = np.asarray(layer.lora_kernel_a)
            arrays[f"{i}/lora_kernel_b"] = np.asarray(layer.lora_kernel_b)
        path = out / "adapter.lora.npz"
        np.savez(path, **arrays)
        (out / "adapter_meta.json").write_text(json.dumps({"rank": self._lora_rank}), "utf-8")
        return path

    def load_adapter(self, path: str | os.PathLike[str]) -> None:
        data = np.load(path)
        meta = json.loads((Path(path).parent / "adapter_meta.json").read_text("utf-8"))
        if self._lora_rank is None:
            self.enable_lora(meta["rank"])
        for i, layer in enumerate(self._lora_layers()):
            layer.lora_kernel_a.assign(data[f"{i}/lora_kernel_a"])
            layer.lora_kernel_b.assign(data[f"{i}/lora_kernel_b"])


@keras.saving.register_keras_serializable(package="recipe_gemma")
class TinyPositional(keras.layers.Layer):
    """Learned absolute position embeddings added to the input."""

    def __init__(self, max_len: int, dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_len, self.dim = max_len, dim

    def build(self, input_shape: Any) -> None:
        self.table = self.add_weight(shape=(self.max_len, self.dim), name="table")

    def call(self, x: Any) -> Any:
        t = keras.ops.shape(x)[1]
        return x + self.table[None, :t, :]

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Shape-preserving (avoids symbolic broadcasting of the sliced table)."""
        return input_shape

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "max_len": self.max_len, "dim": self.dim}


def load_lm(model_cfg: dict[str, Any]) -> CausalLM:
    """Factory: ``{"preset": "tiny-random" | <keras_hub preset> | "hf://..."}``."""
    preset = model_cfg["preset"]
    if preset == "tiny-random":
        return TinyLM(seed=int(model_cfg.get("seed", 0)))
    return GemmaLM(preset, dtype=model_cfg.get("dtype"))
