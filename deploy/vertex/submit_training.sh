#!/usr/bin/env bash
# Build + push the training image with Cloud Build, then submit a Vertex AI custom job.
# BILLABLE: Cloud Build minutes, Artifact Registry storage, g2-standard-8 + L4 for the job.
#
# Required env (checked by guard.sh): ALLOW_GCP_SPEND=1 GCP_PROJECT GCP_REGION + gcloud login
# Required env (this script):        GCS_OUTPUT=gs://<bucket>/recipe-gemma   (job outputs)
# Optional env:
#   TRAIN_CONFIG       configs/train_full.yaml
#   ACCEL              cuda12 | tpu                (Docker build arg; custom_job.yaml is L4)
#   AR_REPO            recipe-gemma                (Artifact Registry docker repo, created if missing)
#   IMAGE_TAG          UTC timestamp
#   SERVICE_ACCOUNT    job service account e-mail  (needs secretAccessor + storage write)
#   KAGGLE_USERNAME_SECRET_ID / KAGGLE_KEY_SECRET_ID   kaggle-username / kaggle-key
#   HF_TOKEN_SECRET_ID (optional; empty = not passed)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=guard.sh
source "$HERE/guard.sh" || exit 1

: "${GCS_OUTPUT:?set GCS_OUTPUT=gs://<bucket>/<prefix> for job outputs}"
[[ "$GCS_OUTPUT" == gs://* ]] || { echo "GCS_OUTPUT must start with gs://" >&2; exit 1; }
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_full.yaml}"
ACCEL="${ACCEL:-cuda12}"
AR_REPO="${AR_REPO:-recipe-gemma}"
IMAGE_TAG="${IMAGE_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}/train:${IMAGE_TAG}"
KAGGLE_USERNAME_SECRET_ID="${KAGGLE_USERNAME_SECRET_ID:-kaggle-username}"
KAGGLE_KEY_SECRET_ID="${KAGGLE_KEY_SECRET_ID:-kaggle-key}"
HF_TOKEN_SECRET_REF=""
if [[ -n "${HF_TOKEN_SECRET_ID:-}" ]]; then
  HF_TOKEN_SECRET_REF="projects/${GCP_PROJECT}/secrets/${HF_TOKEN_SECRET_ID}/versions/latest"
fi
JOB_NAME="recipe-gemma-train-${IMAGE_TAG}"
G=(--project="$GCP_PROJECT")

# The referenced secrets must exist (values are never read here).
for sid in "$KAGGLE_USERNAME_SECRET_ID" "$KAGGLE_KEY_SECRET_ID"; do
  if ! gcloud secrets describe "$sid" "${G[@]}" >/dev/null 2>&1; then
    echo "Secret '$sid' not found. Create it first, e.g.:" >&2
    echo "  printf '%s' \"\$KAGGLE_KEY\" | gcloud secrets create $sid --data-file=- --project=$GCP_PROJECT" >&2
    exit 1
  fi
done

# 1) Artifact Registry repo (idempotent)
if ! gcloud artifacts repositories describe "$AR_REPO" --location="$GCP_REGION" "${G[@]}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPO" --repository-format=docker \
    --location="$GCP_REGION" --description="recipe-gemma images" "${G[@]}"
  vertex_log_resource created artifact-registry-repo \
    "projects/$GCP_PROJECT/locations/$GCP_REGION/repositories/$AR_REPO"
fi

# 2) Build + push with Cloud Build from a clean staged context (no secrets, no data)
vertex_stage_context deploy/vertex/Dockerfile CTX
trap 'rm -rf "$CTX"' EXIT
echo "[submit] building $IMAGE_URI (ACCEL=$ACCEL) from $CTX" >&2
cat >"$CTX/cloudbuild.yaml" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: [build, --build-arg, ACCEL=${ACCEL}, -t, ${IMAGE_URI}, .]
images: [${IMAGE_URI}]
timeout: 3600s
EOF
gcloud builds submit "$CTX" --config="$CTX/cloudbuild.yaml" --region="$GCP_REGION" "${G[@]}"
vertex_log_resource created container-image "$IMAGE_URI"

# 3) Render the job spec and submit
SPEC="$CTX/custom_job.rendered.yaml"
sed -e "s#__IMAGE_URI__#${IMAGE_URI}#g" \
    -e "s#__TRAIN_CONFIG__#${TRAIN_CONFIG}#g" \
    -e "s#__GCP_PROJECT__#${GCP_PROJECT}#g" \
    -e "s#__KAGGLE_USERNAME_SECRET_ID__#${KAGGLE_USERNAME_SECRET_ID}#g" \
    -e "s#__KAGGLE_KEY_SECRET_ID__#${KAGGLE_KEY_SECRET_ID}#g" \
    -e "s#__HF_TOKEN_SECRET_REF__#\"${HF_TOKEN_SECRET_REF}\"#g" \
    -e "s#__GCS_OUTPUT__#${GCS_OUTPUT}#g" \
    "$HERE/custom_job.yaml" >"$SPEC"
if [[ -n "${SERVICE_ACCOUNT:-}" ]]; then
  sed -i "s#__SERVICE_ACCOUNT__#${SERVICE_ACCOUNT}#g" "$SPEC"
else
  sed -i '/__SERVICE_ACCOUNT__/d' "$SPEC"   # default Vertex AI custom-code service agent
  echo "[submit] SERVICE_ACCOUNT unset: the default Vertex service agent must be able to read the secrets" >&2
fi

JOB_ID="$(gcloud ai custom-jobs create --region="$GCP_REGION" --display-name="$JOB_NAME" \
  --config="$SPEC" --format='value(name)' "${G[@]}")"
vertex_log_resource created custom-job "$JOB_ID"

cat >&2 <<EOF

[submit] custom job submitted: $JOB_ID
  stream logs : gcloud ai custom-jobs stream-logs ${JOB_ID##*/} --region=$GCP_REGION --project=$GCP_PROJECT
  status      : gcloud ai custom-jobs describe ${JOB_ID##*/} --region=$GCP_REGION --project=$GCP_PROJECT --format='value(state)'
  cancel      : gcloud ai custom-jobs cancel ${JOB_ID##*/} --region=$GCP_REGION --project=$GCP_PROJECT
  outputs     : ${GCS_OUTPUT%/}/model/{checkpoints,results}/
EOF
