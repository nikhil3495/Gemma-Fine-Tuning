#!/usr/bin/env bash
# Upload the LoRA model to Vertex AI, deploy it to a fresh endpoint, send one test prediction,
# then ALWAYS tear down (undeploy + delete endpoint + delete model) via `trap ... EXIT`, so no
# replica keeps billing after the script ends - even on error or Ctrl-C.
# BILLABLE while running: g2-standard-8 + L4 replica (deploys typically take 10-20 min).
#
# Required env (guard.sh): ALLOW_GCP_SPEND=1 GCP_PROJECT GCP_REGION + gcloud login
# Required env (this script):
#   ADAPTER_GCS_DIR   gs:// dir containing adapter.lora.h5, e.g. the training job output
#                     ${GCS_OUTPUT}/model/checkpoints/full
# Optional env:
#   SERVE_IMAGE_URI   skip the Cloud Build step and use this image
#   AR_REPO           recipe-gemma
#   MODEL_PRESET      gemma3_instruct_1b        MODEL_DTYPE  bfloat16
#   MACHINE_TYPE      g2-standard-8             ACCELERATOR  type=nvidia-l4,count=1 ("" = CPU)
#   SERVICE_ACCOUNT   endpoint runtime service account (secretAccessor + storage read)
#   KAGGLE_USERNAME_SECRET_ID / KAGGLE_KEY_SECRET_ID   kaggle-username / kaggle-key
#   TEST_PROMPT       a recipe request
#   KEEP_MODEL=1      keep the (non-billing) Model resource; endpoint is always deleted
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=guard.sh
source "$HERE/guard.sh" || exit 1

: "${ADAPTER_GCS_DIR:?set ADAPTER_GCS_DIR=gs://<bucket>/<prefix> holding adapter.lora.h5}"
AR_REPO="${AR_REPO:-recipe-gemma}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
NAME="recipe-gemma-${STAMP}"
MODEL_PRESET="${MODEL_PRESET:-gemma3_instruct_1b}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
ACCELERATOR="${ACCELERATOR-type=nvidia-l4,count=1}"
KAGGLE_USERNAME_SECRET_ID="${KAGGLE_USERNAME_SECRET_ID:-kaggle-username}"
KAGGLE_KEY_SECRET_ID="${KAGGLE_KEY_SECRET_ID:-kaggle-key}"
TEST_PROMPT="${TEST_PROMPT:-Create a recipe using: chicken, lemon, garlic, thyme. Cuisine: Mediterranean}"
G=(--project="$GCP_PROJECT")
R=(--region="$GCP_REGION")

MODEL_ID="" ENDPOINT_ID="" DEPLOYED_ID="" CTX="" WORK="$(mktemp -d)"

cleanup() {
  local rc=$?
  set +e
  trap - EXIT INT TERM
  echo "[teardown] starting (exit code $rc)" >&2
  if [[ -n "$ENDPOINT_ID" && -z "$DEPLOYED_ID" ]]; then
    # deploy-model may have been interrupted after creating the deployment
    DEPLOYED_ID="$(gcloud ai endpoints describe "$ENDPOINT_ID" "${R[@]}" "${G[@]}" \
      --format='value(deployedModels[0].id)' 2>/dev/null)"
  fi
  if [[ -n "$ENDPOINT_ID" && -n "$DEPLOYED_ID" ]]; then
    if gcloud ai endpoints undeploy-model "$ENDPOINT_ID" --deployed-model-id="$DEPLOYED_ID" \
      "${R[@]}" "${G[@]}" --quiet; then
      vertex_log_resource deleted deployed-model "$ENDPOINT_ID/deployedModels/$DEPLOYED_ID"
    else
      echo "[teardown] FAILED to undeploy $DEPLOYED_ID - undeploy manually (README)!" >&2
    fi
  fi
  if [[ -n "$ENDPOINT_ID" ]]; then
    if gcloud ai endpoints delete "$ENDPOINT_ID" "${R[@]}" "${G[@]}" --quiet; then
      vertex_log_resource deleted endpoint "$ENDPOINT_ID"
    else
      echo "[teardown] FAILED to delete endpoint $ENDPOINT_ID - delete manually (README)!" >&2
    fi
  fi
  if [[ -n "$MODEL_ID" && "${KEEP_MODEL:-}" != "1" ]]; then
    if gcloud ai models delete "$MODEL_ID" "${R[@]}" "${G[@]}" --quiet; then
      vertex_log_resource deleted model "$MODEL_ID"
    fi
  fi
  [[ -n "$CTX" ]] && rm -rf "$CTX"
  rm -rf "$WORK"
  echo "[teardown] done" >&2
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# 1) Serving image
if [[ -z "${SERVE_IMAGE_URI:-}" ]]; then
  if ! gcloud artifacts repositories describe "$AR_REPO" --location="$GCP_REGION" "${G[@]}" >/dev/null 2>&1; then
    gcloud artifacts repositories create "$AR_REPO" --repository-format=docker \
      --location="$GCP_REGION" --description="recipe-gemma images" "${G[@]}"
    vertex_log_resource created artifact-registry-repo \
      "projects/$GCP_PROJECT/locations/$GCP_REGION/repositories/$AR_REPO"
  fi
  SERVE_IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}/serve:${STAMP}"
  vertex_stage_context deploy/vertex/serve/Dockerfile CTX
  gcloud builds submit "$CTX" --tag="$SERVE_IMAGE_URI" --region="$GCP_REGION" "${G[@]}"
  vertex_log_resource created container-image "$SERVE_IMAGE_URI"
fi

# 2) Model upload (artifact-uri -> AIP_STORAGE_URI inside the container)
ENV_VARS="MODEL_PRESET=${MODEL_PRESET},MODEL_DTYPE=${MODEL_DTYPE}"
ENV_VARS+=",KAGGLE_USERNAME_SECRET=projects/${GCP_PROJECT}/secrets/${KAGGLE_USERNAME_SECRET_ID}/versions/latest"
ENV_VARS+=",KAGGLE_KEY_SECRET=projects/${GCP_PROJECT}/secrets/${KAGGLE_KEY_SECRET_ID}/versions/latest"
gcloud ai models upload "${R[@]}" "${G[@]}" \
  --display-name="$NAME" \
  --container-image-uri="$SERVE_IMAGE_URI" \
  --artifact-uri="$ADAPTER_GCS_DIR" \
  --container-ports=8080 \
  --container-health-route=/health \
  --container-predict-route=/predict \
  --container-env-vars="$ENV_VARS"
MODEL_ID="$(gcloud ai models list "${R[@]}" "${G[@]}" --filter="display_name=$NAME" \
  --format='value(name)' | head -n1)"
[[ -n "$MODEL_ID" ]] || { echo "model upload: could not find model $NAME" >&2; exit 1; }
vertex_log_resource created model "$MODEL_ID"

# 3) Endpoint
gcloud ai endpoints create "${R[@]}" "${G[@]}" --display-name="$NAME-ep"
ENDPOINT_ID="$(gcloud ai endpoints list "${R[@]}" "${G[@]}" --filter="display_name=$NAME-ep" \
  --format='value(name)' | head -n1)"
[[ -n "$ENDPOINT_ID" ]] || { echo "endpoint create: could not find $NAME-ep" >&2; exit 1; }
ENDPOINT_ID="${ENDPOINT_ID##*/}"
vertex_log_resource created endpoint "projects/$GCP_PROJECT/locations/$GCP_REGION/endpoints/$ENDPOINT_ID"

# 4) Deploy (exactly one replica)
DEPLOY_ARGS=(--model="${MODEL_ID##*/}" --display-name="$NAME" --machine-type="$MACHINE_TYPE"
  --min-replica-count=1 --max-replica-count=1 --traffic-split=0=100)
[[ -n "$ACCELERATOR" ]] && DEPLOY_ARGS+=(--accelerator="$ACCELERATOR")
[[ -n "${SERVICE_ACCOUNT:-}" ]] && DEPLOY_ARGS+=(--service-account="$SERVICE_ACCOUNT")
gcloud ai endpoints deploy-model "$ENDPOINT_ID" "${R[@]}" "${G[@]}" "${DEPLOY_ARGS[@]}"
DEPLOYED_ID="$(gcloud ai endpoints describe "$ENDPOINT_ID" "${R[@]}" "${G[@]}" \
  --format='value(deployedModels[0].id)')"
vertex_log_resource created deployed-model "$ENDPOINT_ID/deployedModels/$DEPLOYED_ID"

# 5) Test prediction
REQ="$WORK/request.json"
PYTHON="$(command -v python3 || command -v python)"
"$PYTHON" - "$TEST_PROMPT" >"$REQ" <<'PY'
import json, sys
print(json.dumps({"instances": [{"prompt": sys.argv[1]}], "parameters": {"max_length": 512}}))
PY
echo "[predict] request: $(cat "$REQ")" >&2
gcloud ai endpoints predict "$ENDPOINT_ID" "${R[@]}" "${G[@]}" --json-request="$REQ" \
  | tee "$WORK/response.txt"
echo "[predict] OK - tearing down now" >&2
# cleanup() runs on EXIT
