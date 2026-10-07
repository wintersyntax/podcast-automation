#!/usr/bin/env bash
# Build and deploy one immutable image, then tag the image Cloud Run accepted.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Shared Cloud Run discovery and Artifact Registry production-tag helpers.
source "$SCRIPT_DIR/sync_production_image_tags.sh"

usage() {
  cat <<'EOF'
Usage: scripts/deploy_production_image.sh <worker|human-review> [--dry-run]

Builds a commit-addressed image, resolves its digest, updates only the target
Cloud Run resource's image, confirms the configured image, and then moves the
Artifact Registry production tag to that confirmed image.

The working tree must be clean. --dry-run performs local preflight only and
does not contact Google Cloud.
EOF
}

require_clean_worktree() {
  if [[ -n "$(git -C "$REPOSITORY_ROOT" status --porcelain)" ]]; then
    fail "Refusing to deploy a dirty working tree. Commit or stash the changes first."
  fi
}

run_worker_summary_review_activation_gate() {
  [[ -f "$SCRIPT_DIR/summary_review_activation_preflight.py" ]] ||
    fail "Missing summary review activation preflight helper."
  command -v uv >/dev/null 2>&1 ||
    fail "uv must be installed and on PATH for summary review activation preflight."

  gcloud run jobs describe "$WORKER_JOB" \
    --region="$REGION" \
    --project="$PROJECT" \
    --format=json \
    | uv run --python 3.12 python "$SCRIPT_DIR/summary_review_activation_preflight.py"
}

run_worker_corpus_checkpoint_gate() {
  [[ -x "$SCRIPT_DIR/corpus-checkpoint-status" ]] ||
    fail "Missing executable corpus checkpoint status helper."
  "$SCRIPT_DIR/corpus-checkpoint-status" --gate
}

image_ref_for() {
  local image_name="$1"
  local commit_sha="$2"

  printf '%s\n' "${REGION}-docker.pkg.dev/${PROJECT}/${ARTIFACT_REPOSITORY}/${image_name}:${image_name}-${commit_sha}"
}

resolve_image_digest() {
  local image_ref="$1"
  local digest repository

  digest="$(gcloud artifacts docker images describe "$image_ref" \
    --project="$PROJECT" \
    --format='value(image_summary.digest)')"
  [[ "$digest" =~ ^sha256:[[:xdigit:]]{64}$ ]] || fail "Could not resolve an immutable digest for $image_ref"
  repository="$(image_repository "$image_ref")"
  printf '%s\n' "${repository}@${digest}"
}

deploy_target() {
  local target="$1"
  local image_ref="$2"

  case "$target" in
    worker)
      gcloud run jobs update "$WORKER_JOB" \
        --image="$image_ref" \
        --region="$REGION" \
        --project="$PROJECT"
      ;;
    human-review)
      gcloud run services update "$REVIEW_SERVICE" \
        --image="$image_ref" \
        --region="$REGION" \
        --project="$PROJECT"
      ;;
    *) fail "Unknown deployment target: $target" ;;
  esac
}

main() {
  local target="${1:-}"
  local dry_run="false"
  local image_name build_config commit_sha image_ref immutable_image active_image

  case "$target" in
    worker)
      image_name="podcast-worker"
      build_config="cloudbuild.worker.yaml"
      ;;
    human-review)
      image_name="podcast-human-review"
      build_config="cloudbuild.review.yaml"
      ;;
    -h|--help|'') usage; return 0 ;;
    *) fail "Unknown deployment target: $target"; return 1 ;;
  esac
  shift

  while (($#)); do
    case "$1" in
      --dry-run) dry_run="true" ;;
      -h|--help) usage; return 0 ;;
      *) fail "Unknown argument: $1"; return 1 ;;
    esac
    shift
  done

  require_clean_worktree
  commit_sha="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD)"
  image_ref="$(image_ref_for "$image_name" "$commit_sha")"

  echo "Target: $target"
  echo "Build image: $image_ref"

  if [[ "$dry_run" == "true" ]]; then
    echo "DRY RUN: would build the image, deploy its resolved digest, verify Cloud Run, and synchronize :production."
    return 0
  fi

  require_gcloud
  if [[ "$target" == "worker" ]]; then
    run_worker_summary_review_activation_gate
    run_worker_corpus_checkpoint_gate
  fi
  gcloud builds submit "$REPOSITORY_ROOT" \
    --config="$REPOSITORY_ROOT/$build_config" \
    --substitutions="_IMAGE=$image_ref"

  immutable_image="$(resolve_image_digest "$image_ref")"
  echo "Immutable image: $immutable_image"
  deploy_target "$target" "$immutable_image"

  active_image="$(active_image_for "$target")"
  [[ "$active_image" == "$immutable_image" ]] || fail "Cloud Run did not retain the expected image. Production tag was not changed."

  sync_target "$target"
  echo "DEPLOYMENT COMPLETE: $target"
}

main "$@"
