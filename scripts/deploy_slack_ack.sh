#!/usr/bin/env bash
# Deploy the dedicated Slack interaction acknowledgement function.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)" || {
  echo "ERROR: scripts/deploy_slack_ack.sh must run from this Git repository." >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/deploy_slack_ack.sh [--dry-run]

Deploys the existing podcast-slack-ack Gen2 HTTP function using its dedicated
build and runtime service accounts. Existing runtime and public-invoker
configuration is preserved.

--dry-run prints the intended configuration and deploy command without
contacting or changing Google Cloud.
EOF
}

main() {
  local dry_run="false"
  local -a deploy_command

  while (($#)); do
    case "$1" in
      --dry-run) dry_run="true" ;;
      -h|--help) usage; return 0 ;;
      *)
        echo "ERROR: Unknown argument: $1" >&2
        usage >&2
        return 1
        ;;
    esac
    shift
  done

  cd "$REPOSITORY_ROOT"
  [[ -f slack_ack/main.py ]] || {
    echo "ERROR: Expected Slack ACK source at $REPOSITORY_ROOT/slack_ack/main.py" >&2
    return 1
  }

  deploy_command=(
    gcloud functions deploy podcast-slack-ack
    --gen2
    --project=YOUR_GCP_PROJECT_ID
    --region=europe-west1
    --runtime=python312
    --source=slack_ack
    --entry-point=slack_interactions
    --trigger-http
    --build-service-account=projects/YOUR_GCP_PROJECT_ID/serviceAccounts/podcast-slack-ack-build@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com
    --service-account=podcast-slack-ack-runtime@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com
    --update-secrets=PODCAST_SLACK_SIGNING_SECRET=podcast-slack-signing-secret:latest
  )

  echo "Function: podcast-slack-ack (Gen2 HTTP)"
  echo "Project: YOUR_GCP_PROJECT_ID"
  echo "Region: europe-west1"
  echo "Source: slack_ack"
  echo "Entry point: slack_interactions"
  echo "Build service account: podcast-slack-ack-build@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com"
  echo "Runtime service account: podcast-slack-ack-runtime@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com"
  echo "Secret update: PODCAST_SLACK_SIGNING_SECRET=podcast-slack-signing-secret:latest"
  echo "Runtime, min-instances, and public-invoker settings: preserved"
  printf 'Deploy command:'
  printf ' %q' "${deploy_command[@]}"
  printf '\n'

  if [[ "$dry_run" == "true" ]]; then
    echo "DRY RUN: no Google Cloud command was run."
    return 0
  fi

  "${deploy_command[@]}"
}

main "$@"
