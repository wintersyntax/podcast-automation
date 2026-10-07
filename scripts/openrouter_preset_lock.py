#!/usr/bin/env python3
"""Check or explicitly update the transcript-reviewer OpenRouter preset lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from podcast_engine.compilation import REVIEW_POLICY_VERSION
from podcast_engine.preset_provenance import (
    LOCK_PATH,
    LOCK_SCHEMA_VERSION,
    config_sha256,
    fetch_current_transcript_reviewer,
    load_transcript_reviewer_lock,
    system_prompt_sha256,
    verify_transcript_reviewer,
)
from podcast_engine.review import REVIEW_PRESET


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_result(result) -> dict:
    return result.record()


def _write_lock(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _policy_generation(policy: object) -> int:
    """Return a comparable resolver-policy generation or fail closed."""

    match = re.fullmatch(r"resolver-policy-v([1-9][0-9]*)", policy) if isinstance(policy, str) else None
    if match is None:
        raise ValueError("Resolver policy version is not a comparable resolver-policy-vN generation")
    return int(match.group(1))


def _semantic_update_requires_forward_policy_bump(old_policy: object, current_policy: object) -> None:
    old_generation = _policy_generation(old_policy)
    current_generation = _policy_generation(current_policy)
    if current_generation <= old_generation:
        raise ValueError("Remote semantic preset change requires a strictly newer REVIEW_POLICY_VERSION generation")


def check() -> int:
    result = verify_transcript_reviewer(
        local_policy_version=REVIEW_POLICY_VERSION,
        preset_slug=REVIEW_PRESET,
        verified_at=_now(),
    )
    print(json.dumps(_safe_result(result), sort_keys=True))
    return 0 if result.verified else 1


def update() -> int:
    try:
        raw = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        old_lock = raw["presets"]["transcript_reviewer"]
        if not isinstance(old_lock, dict):
            raise ValueError("Transcript reviewer lock is missing")
        current, exact = fetch_current_transcript_reviewer(preset_slug=REVIEW_PRESET)
        new_config_digest = config_sha256(exact["config"])
        new_prompt_digest = system_prompt_sha256(exact["system_prompt"])
        semantic_change = (
            old_lock.get("config_sha256") != new_config_digest
            or old_lock.get("system_prompt_sha256") != new_prompt_digest
        )
        old_generation = _policy_generation(old_lock.get("local_policy_version"))
        current_generation = _policy_generation(REVIEW_POLICY_VERSION)

        if current_generation < old_generation:
            raise ValueError(
                "REVIEW_POLICY_VERSION cannot move backward relative to the committed preset lock"
            )

        if semantic_change:
            _semantic_update_requires_forward_policy_bump(
                old_lock.get("local_policy_version"),
                REVIEW_POLICY_VERSION,
            )

        new_lock = {
            "slug": REVIEW_PRESET,
            "local_policy_version": REVIEW_POLICY_VERSION,
            "preset_id": current["id"],
            "designated_version_id": exact["id"],
            "version": exact["version"],
            "config": exact["config"],
            "config_sha256": new_config_digest,
            "system_prompt_sha256": new_prompt_digest,
            "openrouter_updated_at": exact.get("updated_at"),
            "captured_at": _now(),
        }
        _write_lock(
            LOCK_PATH,
            {
                "schema_version": LOCK_SCHEMA_VERSION,
                "presets": {"transcript_reviewer": new_lock},
            },
        )
        print(json.dumps({"status": "updated", **{key: new_lock[key] for key in ("slug", "preset_id", "designated_version_id", "version", "config_sha256", "system_prompt_sha256")}}, sort_keys=True))
        return 0
    except (KeyError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "invalid", "reason": str(error)[:160]}, sort_keys=True), file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Verify the committed lock without writing files.")
    action.add_argument("--update", action="store_true", help="Explicitly capture the current designated preset version.")
    args = parser.parse_args()
    return check() if args.check else update()


if __name__ == "__main__":
    raise SystemExit(main())
