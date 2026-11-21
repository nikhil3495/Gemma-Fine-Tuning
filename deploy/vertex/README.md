# Vertex AI (optional, guarded)

An **optional** path to run the full LoRA fine-tune and a short-lived serving test on Google
Cloud Vertex AI. Nothing in the main project depends on it: the default full run is the free
Kaggle/Colab notebooks in `notebooks/`.

> **Cost warning.** These scripts create **billable** resources: Cloud Build minutes,
> Artifact Registry storage, a `g2-standard-8` + 1x NVIDIA L4 training VM for up to 6 h
> (`scheduling.timeout` in `custom_job.yaml`), and, in `deploy_endpoint.sh`, an L4 serving
> replica for the 10-30 min the script runs. Check current pricing for your region before
> running. Vertex endpoints bill **per replica-hour while deployed, even when idle**, which is
> why `deploy_endpoint.sh` always tears its endpoint down.

## Files

| File | What it does |
|---|---|
| `guard.sh` | Spend guard, sourced first by every script. Also defines `vertex_log_resource` (appends created/deleted resource ids to `docs/DECISIONS.md`) and `vertex_stage_context` (a clean Docker build context: only `pyproject.toml`, `README.md`, `LICENSE`, `src/`, `configs/`, `deploy/vertex/`; no `.venv`, `data/`, `checkpoints/` or `.env`). |
| `Dockerfile` | Training image: `python:3.12-slim`, editable install of the repo with `[train,eval,gcp]`, plus `jax[cuda12]` (default) or `jax[tpu]` via `--build-arg ACCEL=tpu`. |
| `train_entrypoint.py` | Container entrypoint: resolve secrets, then `recipe_gemma.data.build`, then `recipe_gemma.train --config $CONFIG`, then upload `checkpoints/` and `results/` to `$OUTPUT_URI` or `$AIP_MODEL_DIR` with google-cloud-storage. |
| `resolve_secrets.py` | Turns `NAME_SECRET=projects/.../secrets/.../versions/latest` env vars into `NAME` env vars at runtime (Secret Manager). Values are never printed. |
| `custom_job.yaml` | `CustomJobSpec`: 1x `g2-standard-8` + 1x `NVIDIA_L4`, 100 GB SSD, 6 h timeout. Kaggle/HF credentials are **Secret Manager references**, never values. |
| `submit_training.sh` | Guard, then create the Artifact Registry repo if needed, build and push the image with Cloud Build, render `custom_job.yaml`, then `gcloud ai custom-jobs create`. |
| `serve/predictor.py` | FastAPI predictor for a Vertex custom container (`AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE`, `AIP_STORAGE_URI`). Loads the Gemma preset through `recipe_gemma.train.lm.load_lm`, loads `adapter.lora.h5` and wraps each `{"prompt": ...}` with `recipe_gemma.data.prompts.chat_prompt`. |
| `serve/Dockerfile` | Serving image (`ACCEL=cuda12` default, or `cpu`). |
| `deploy_endpoint.sh` | Guard, then `models upload`, `endpoints create`, `deploy-model` (min = max = 1 replica), one test `predict`, then **automatic** `undeploy-model` + `endpoints delete` + `models delete` from a `trap ... EXIT`. |

## Guard semantics

`guard.sh` exits (or, when sourced, returns) **non-zero** with a message listing every unmet
condition, unless **all** of these hold:

1. `gcloud auth list --filter=status:ACTIVE` shows an account (`gcloud auth login`),
2. `ALLOW_GCP_SPEND=1` (explicit opt-in to spending money),
3. `GCP_PROJECT` is set (every command passes `--project=$GCP_PROJECT`; the gcloud default
   project is ignored),
4. `GCP_REGION` is set (it must have L4 quota for the defaults, e.g. `us-central1`).

```bash
bash deploy/vertex/guard.sh && echo "guard OK"   # dry check, creates nothing
```

Every resource the scripts create or delete is appended to `docs/DECISIONS.md` as
`- **GCP <kind> <created|deleted> (<UTC time>):** <id> ...`. Lines appear only when a script
actually ran.

## One-time setup

```bash
pip install -e ".[gcp]"   # optional: google-cloud-aiplatform for Python-side use; the scripts only need gcloud
gcloud auth login
export GCP_PROJECT=<your-project> GCP_REGION=us-central1
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com --project=$GCP_PROJECT
gcloud storage buckets create gs://<bucket> --location=$GCP_REGION --project=$GCP_PROJECT

# Kaggle creds for the gated `gemma3_instruct_1b` preset (accept the Gemma license on Kaggle first).
# Values are read from your shell env, never written to a file in the repo.
printf '%s' "$KAGGLE_USERNAME" | gcloud secrets create kaggle-username --data-file=- --project=$GCP_PROJECT
printf '%s' "$KAGGLE_KEY"      | gcloud secrets create kaggle-key      --data-file=- --project=$GCP_PROJECT

# Let the job / endpoint service account read them (default: the Vertex AI custom-code service agent,
# service-<PROJECT_NUMBER>@gcp-sa-aiplatform-cc.iam.gserviceaccount.com, or pass SERVICE_ACCOUNT=...).
for s in kaggle-username kaggle-key; do
  gcloud secrets add-iam-policy-binding $s --project=$GCP_PROJECT \
    --member="serviceAccount:<SA_EMAIL>" --role=roles/secretmanager.secretAccessor
done
```

## Train

```bash
export ALLOW_GCP_SPEND=1 GCP_PROJECT=<your-project> GCP_REGION=us-central1
export GCS_OUTPUT=gs://<bucket>/recipe-gemma
bash deploy/vertex/submit_training.sh
# optional: TRAIN_CONFIG=configs/train_full.yaml ACCEL=cuda12 SERVICE_ACCOUNT=<sa-email> HF_TOKEN_SECRET_ID=hf-token
```

Outputs land in `$GCS_OUTPUT/model/checkpoints/full/adapter.lora.h5` and
`$GCS_OUTPUT/model/results/`. Copy them back with
`gcloud storage cp -r $GCS_OUTPUT/model/results ./results_vertex`.

The TPU build (`ACCEL=tpu`) produces a TPU image, but `custom_job.yaml` targets an L4; change
`machineSpec` to a TPU machine type with quota in your region before using it.

## Serve (short-lived test)

```bash
export ALLOW_GCP_SPEND=1 GCP_PROJECT=<your-project> GCP_REGION=us-central1
export ADAPTER_GCS_DIR=gs://<bucket>/recipe-gemma/model/checkpoints/full
bash deploy/vertex/deploy_endpoint.sh
# optional: SERVE_IMAGE_URI=<existing image> MACHINE_TYPE=... ACCELERATOR=... TEST_PROMPT="..." KEEP_MODEL=1
```

Request and response format:

```json
{"instances": [{"prompt": "Suggest 5 ingredients that pair well with miso, with a one-line reason each."}],
 "parameters": {"max_length": 512}}
{"predictions": [{"prompt": "...", "output": "Ingredients that pair well with miso: ..."}]}
```

Local smoke test of the serving image (no Vertex; needs Docker and, for the Kaggle preset,
Kaggle creds in your shell):

```bash
docker run --rm -p 8080:8080 -e KAGGLE_USERNAME -e KAGGLE_KEY \
  -e ADAPTER_PATH=/adapter/adapter.lora.h5 -v "$PWD/checkpoints/full:/adapter" <serve-image>
curl -s localhost:8080/health
curl -s -X POST localhost:8080/predict -H 'content-type: application/json' \
  -d '{"instances":[{"prompt":"Create a recipe using: tofu, ginger. Cuisine: Japanese"}]}'
```

## Teardown (manual, if a script was killed with SIGKILL or teardown failed)

```bash
R="--region=$GCP_REGION --project=$GCP_PROJECT"
gcloud ai endpoints list $R                                   # find recipe-gemma-*-ep
gcloud ai endpoints describe <ENDPOINT_ID> $R --format='value(deployedModels[].id)'
gcloud ai endpoints undeploy-model <ENDPOINT_ID> --deployed-model-id=<DEPLOYED_ID> $R --quiet
gcloud ai endpoints delete <ENDPOINT_ID> $R --quiet
gcloud ai models list $R --filter='display_name~^recipe-gemma'
gcloud ai models delete <MODEL_ID> $R --quiet
gcloud ai custom-jobs list $R --filter='state=JOB_STATE_RUNNING'
gcloud ai custom-jobs cancel <JOB_ID> $R
# storage that keeps billing at a low rate:
gcloud artifacts repositories delete recipe-gemma --location=$GCP_REGION --project=$GCP_PROJECT --quiet
gcloud storage rm -r gs://<bucket>/recipe-gemma
```

The `[gcp]` extra in `pyproject.toml` (`google-cloud-aiplatform`) is for Python-side use (for
example, calling an endpoint from a notebook). The shell scripts need only the `gcloud` CLI. The
images install `google-cloud-storage` and `google-cloud-secret-manager` themselves.
