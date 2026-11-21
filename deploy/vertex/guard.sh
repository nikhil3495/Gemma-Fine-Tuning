#!/usr/bin/env bash
# Spend guard for every Vertex AI script in this directory.
#
# Usage (from another script):   source "$(dirname "${BASH_SOURCE[0]}")/guard.sh" || exit 1
# Usage (standalone check):      bash deploy/vertex/guard.sh && echo "guard OK"
#
# Refuses (non-zero) unless ALL of these hold:
#   1. `gcloud` is on PATH and `gcloud auth list --filter=status:ACTIVE` shows an account
#   2. ALLOW_GCP_SPEND=1            (explicit opt-in: these scripts create billable resources)
#   3. GCP_PROJECT is set           (passed as --project to every gcloud call)
#   4. GCP_REGION is set            (e.g. us-central1; must have L4 quota for the defaults)
#
# Also defines helpers used by the other scripts:
#   vertex_log_resource <created|deleted> <kind> <resource-id>   -> appends to docs/DECISIONS.md
#   vertex_stage_context <dockerfile> <out-var>                  -> clean Cloud Build context

VERTEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$VERTEX_DIR/../.." && pwd)"
DECISIONS_MD="$REPO_ROOT/docs/DECISIONS.md"

vertex_guard() {
  local problems=()
  if ! command -v gcloud >/dev/null 2>&1; then
    problems+=("gcloud CLI not found on PATH (install the Google Cloud SDK)")
  else
    local active
    active="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
    if [[ -z "$active" ]]; then
      problems+=("no ACTIVE gcloud account (run: gcloud auth login)")
    fi
  fi
  if [[ "${ALLOW_GCP_SPEND:-}" != "1" ]]; then
    problems+=("ALLOW_GCP_SPEND is not 1 (these scripts create BILLABLE resources; export ALLOW_GCP_SPEND=1 to opt in)")
  fi
  if [[ -z "${GCP_PROJECT:-}" ]]; then
    problems+=("GCP_PROJECT is not set (export GCP_PROJECT=<your-project-id>)")
  fi
  if [[ -z "${GCP_REGION:-}" ]]; then
    problems+=("GCP_REGION is not set (export GCP_REGION=us-central1)")
  fi

  if ((${#problems[@]})); then
    echo "[guard] REFUSING to touch Google Cloud:" >&2
    local p
    for p in "${problems[@]}"; do
      echo "[guard]   - $p" >&2
    done
    return 1
  fi
  echo "[guard] OK: account=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n1)" \
    "project=$GCP_PROJECT region=$GCP_REGION (ALLOW_GCP_SPEND=1)" >&2
  return 0
}

# Append one line per created/deleted cloud resource to docs/DECISIONS.md.
vertex_log_resource() {
  local action="$1" kind="$2" rid="$3"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local caller
  caller="$(basename "${BASH_SOURCE[1]:-$0}")"
  printf -- '- **GCP %s %s (%s):** `%s` (project `%s`, region `%s`, via `deploy/vertex/%s`).\n' \
    "$kind" "$action" "$ts" "$rid" "$GCP_PROJECT" "$GCP_REGION" "$caller" >>"$DECISIONS_MD"
  echo "[log] $action $kind $rid" >&2
}

# Build a minimal Docker build context (no .venv, data, checkpoints, .env or other secrets):
# only the files the image needs, with the chosen Dockerfile at the context root.
#   vertex_stage_context deploy/vertex/Dockerfile CTX ; gcloud builds submit "$CTX" --tag ...
vertex_stage_context() {
  local dockerfile="$1" __out="$2"
  local ctx
  ctx="$(mktemp -d)"
  cp "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/README.md" "$ctx/"
  [[ -f "$REPO_ROOT/LICENSE" ]] && cp "$REPO_ROOT/LICENSE" "$ctx/"
  cp -r "$REPO_ROOT/src" "$REPO_ROOT/configs" "$ctx/"
  mkdir -p "$ctx/deploy"
  cp -r "$VERTEX_DIR" "$ctx/deploy/vertex"
  find "$ctx" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  find "$ctx" -name '*.egg-info' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  cp "$REPO_ROOT/$dockerfile" "$ctx/Dockerfile"
  printf -v "$__out" '%s' "$ctx"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  # sourced: propagate failure to the caller (which runs with `|| exit 1` / set -e)
  vertex_guard || return 1
else
  vertex_guard
  exit $?
fi
