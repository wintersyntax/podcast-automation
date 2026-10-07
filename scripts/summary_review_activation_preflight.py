from __future__ import annotations

import json
import sys
from typing import Any


PRESET_ENV = "PODCAST_SUMMARY_REVIEW_PRESET"
API_KEY_ENV = "PODCAST_SUMMARY_REVIEW_API_KEY"
OTHER_OPENROUTER_API_KEY_ENVS = (
    "PODCAST_KNOWLEDGE_API_KEY",
    "PODCAST_TRANSCRIPT_REVIEW_API_KEY",
)


def _worker_env(job: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        containers = job["spec"]["template"]["spec"]["template"]["spec"]["containers"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cloud Run Worker configuration is missing the expected container structure") from exc

    if not isinstance(containers, list) or len(containers) != 1:
        raise ValueError("Cloud Run Worker configuration must contain exactly one container")

    container = containers[0]
    if not isinstance(container, dict):
        raise ValueError("Cloud Run Worker container configuration is malformed")

    env = container.get("env", [])
    if not isinstance(env, list):
        raise ValueError("Cloud Run Worker env configuration is malformed")

    entries: list[dict[str, Any]] = []
    for entry in env:
        if not isinstance(entry, dict):
            raise ValueError("Cloud Run Worker env entry is malformed")
        entries.append(entry)
    return entries


def _named_env_entry(entries: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    matches = [entry for entry in entries if entry.get("name") == name]
    if len(matches) > 1:
        raise ValueError(f"Cloud Run Worker has duplicate {name} entries")
    return matches[0] if matches else None


def validate_worker_summary_review_activation(
    job: dict[str, Any],
) -> tuple[str, str | None]:
    entries = _worker_env(job)
    preset_entry = _named_env_entry(entries, PRESET_ENV)

    if preset_entry is None:
        return ("INACTIVE", None)

    preset = preset_entry.get("value")
    if preset is None:
        raise ValueError(f"{PRESET_ENV} must use an explicit string value")
    if not isinstance(preset, str):
        raise ValueError(f"{PRESET_ENV} must be a string")

    preset = preset.strip()
    if not preset:
        return ("INACTIVE", None)

    key_entry = _named_env_entry(entries, API_KEY_ENV)
    if key_entry is None:
        raise ValueError(
            f"{PRESET_ENV} is active but {API_KEY_ENV} is missing"
        )

    if "value" in key_entry:
        raise ValueError(
            f"{API_KEY_ENV} must be bound through Secret Manager, not plaintext"
        )

    value_from = key_entry.get("valueFrom")
    if not isinstance(value_from, dict):
        raise ValueError(
            f"{API_KEY_ENV} must be bound through Secret Manager"
        )

    secret_ref = value_from.get("secretKeyRef")
    if not isinstance(secret_ref, dict):
        raise ValueError(
            f"{API_KEY_ENV} must be bound through Secret Manager"
        )

    secret_name = secret_ref.get("name")
    secret_key = secret_ref.get("key")
    if not isinstance(secret_name, str) or not secret_name.strip():
        raise ValueError(
            f"{API_KEY_ENV} Secret Manager binding is missing a secret name"
        )
    if not isinstance(secret_key, str) or not secret_key.strip():
        raise ValueError(
            f"{API_KEY_ENV} Secret Manager binding is missing a secret version"
        )

    reviewer_secret_name = secret_name.strip()
    for other_env in OTHER_OPENROUTER_API_KEY_ENVS:
        other_entry = _named_env_entry(entries, other_env)
        if other_entry is None:
            continue
        other_value_from = other_entry.get("valueFrom")
        if not isinstance(other_value_from, dict):
            continue
        other_secret_ref = other_value_from.get("secretKeyRef")
        if not isinstance(other_secret_ref, dict):
            continue
        other_secret_name = other_secret_ref.get("name")
        if (
            isinstance(other_secret_name, str)
            and other_secret_name.strip() == reviewer_secret_name
        ):
            raise ValueError(
                f"{API_KEY_ENV} must use a dedicated Secret Manager secret "
                f"distinct from {other_env}"
            )

    return ("ACTIVE", preset)


def main() -> int:
    try:
        job = json.load(sys.stdin)
        if not isinstance(job, dict):
            raise ValueError("Cloud Run Worker configuration must be a JSON object")
        state, preset = validate_worker_summary_review_activation(job)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"SUMMARY REVIEW ACTIVATION PREFLIGHT: RED - {exc}", file=sys.stderr)
        return 1

    if state == "ACTIVE":
        print(f"SUMMARY REVIEW ACTIVATION PREFLIGHT: ACTIVE - preset={preset}")
    else:
        print("SUMMARY REVIEW ACTIVATION PREFLIGHT: INACTIVE - no production reviewer selected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
