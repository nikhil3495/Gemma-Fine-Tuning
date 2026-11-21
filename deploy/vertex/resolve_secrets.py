"""Resolve Secret Manager references into environment variables (inside Vertex containers).

Vertex AI CustomJob / Model container specs only accept plain ``name=value`` env vars, so
credentials are never passed inline. Instead the job passes *references*:

    KAGGLE_USERNAME_SECRET=projects/<p>/secrets/kaggle-username/versions/latest
    KAGGLE_KEY_SECRET=projects/<p>/secrets/kaggle-key/versions/latest
    HF_TOKEN_SECRET=projects/<p>/secrets/hf-token/versions/latest        (optional)

and :func:`resolve_secret_env` fetches each one with the job's service account (which needs
``roles/secretmanager.secretAccessor``) and sets ``KAGGLE_USERNAME`` etc. Values are never
printed or logged; only the variable names are.
"""

from __future__ import annotations

import os

SUFFIX = "_SECRET"


def resolve_secret_env(environ: dict[str, str] | None = None) -> list[str]:
    """Set ``NAME`` from every ``NAME_SECRET`` Secret Manager reference. Returns resolved names."""
    env = os.environ if environ is None else environ
    refs = {k[: -len(SUFFIX)]: v for k, v in env.items() if k.endswith(SUFFIX) and v}
    if not refs:
        return []
    from google.cloud import secretmanager  # installed in the Vertex images

    client = secretmanager.SecretManagerServiceClient()
    done = []
    for name, ref in sorted(refs.items()):
        if env.get(name):  # an explicit value already present wins (e.g. local docker run)
            continue
        resp = client.access_secret_version(request={"name": ref})
        env[name] = resp.payload.data.decode("utf-8").strip()
        done.append(name)
    print(f"[secrets] resolved from Secret Manager: {', '.join(done) or '(none)'}", flush=True)
    return done


if __name__ == "__main__":
    resolve_secret_env()
