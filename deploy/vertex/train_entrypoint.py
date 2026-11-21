"""Vertex AI training-container entrypoint.

1. resolve Kaggle / HF credentials from Secret Manager references (see resolve_secrets.py)
2. python -m recipe_gemma.data.build --config $DATA_CONFIG   (default configs/data.yaml)
3. python -m recipe_gemma.train --config $CONFIG            (default configs/train_full.yaml)
4. copy checkpoints/ and results/ to $OUTPUT_URI, else $AIP_MODEL_DIR (set by Vertex when the
   job has a baseOutputDirectory), using google-cloud-storage.

Outputs are uploaded even if training fails (partial logs are useful), then the exit code of
the failed step is returned so the Vertex job is marked FAILED.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP = Path(os.environ.get("APP_DIR", "/app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolve_secrets import resolve_secret_env  # noqa: E402


def run(cmd: list[str]) -> int:
    print(f"[entrypoint] $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=APP)


def upload_dir(local: Path, gcs_uri: str) -> int:
    """Recursively upload ``local`` to ``gs://bucket/prefix/<local.name>/``."""
    if not local.exists():
        print(f"[entrypoint] skip upload, {local} does not exist", flush=True)
        return 0
    from google.cloud import storage

    bucket_name, _, prefix = gcs_uri.removeprefix("gs://").partition("/")
    bucket = storage.Client().bucket(bucket_name)
    n = 0
    for f in local.rglob("*"):
        if f.is_file():
            key = "/".join(
                p for p in (prefix.rstrip("/"), local.name, f.relative_to(local).as_posix()) if p
            )
            bucket.blob(key).upload_from_filename(str(f))
            n += 1
    print(
        f"[entrypoint] uploaded {n} files from {local} to gs://{bucket_name}/{prefix}", flush=True
    )
    return n


def main() -> int:
    os.environ.setdefault("KERAS_BACKEND", "jax")
    resolve_secret_env()
    data_cfg = os.environ.get("DATA_CONFIG", "configs/data.yaml")
    train_cfg = os.environ.get("CONFIG", "configs/train_full.yaml")

    rc = 0
    if os.environ.get("SKIP_DATA_BUILD") != "1":
        rc = run([sys.executable, "-m", "recipe_gemma.data.build", "--config", data_cfg])
    if rc == 0:
        rc = run([sys.executable, "-m", "recipe_gemma.train", "--config", train_cfg])

    dest = os.environ.get("OUTPUT_URI") or os.environ.get("AIP_MODEL_DIR")
    if dest and dest.startswith("gs://"):
        for sub in ("checkpoints", "results"):
            upload_dir(APP / sub, dest)
    else:
        print(
            "[entrypoint] no gs:// OUTPUT_URI / AIP_MODEL_DIR; outputs stay in the container",
            flush=True,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
