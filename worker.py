"""Compatibility entry point for one complete podcast processing run."""

import logging
import os

from podcast_engine.human_review import (
    RecompileAuditPersistenceError,
    complete_recompile_request,
)
from podcast_engine.pipeline import run_pipeline
from podcast_engine.worker_control import (
    RECOMPILE_EPISODE_KEY_ENV,
    RECOMPILE_REQUEST_ID_ENV,
    REVIEW_GENERATION_ENV,
)


LOGGER = logging.getLogger(__name__)


def _recompile_correlation_from_environment() -> dict[str, str] | None:
    values = {
        "request_id": os.environ.get(RECOMPILE_REQUEST_ID_ENV),
        "review_generation": os.environ.get(REVIEW_GENERATION_ENV),
        "episode_key": os.environ.get(RECOMPILE_EPISODE_KEY_ENV),
    }
    if not any(values.values()):
        return None
    if not all(isinstance(value, str) and value for value in values.values()):
        raise RuntimeError("Recompile worker correlation environment is incomplete")
    return values


def _require_completed_recompile_target(outcomes: list[dict], episode_key: str) -> None:
    """Require the correlated target to have completed exactly once.

    A successful correlated Cloud Run execution is reconciliation evidence that
    the exact target completed its full pipeline.  Do not let outcomes for
    other episodes, or an incomplete target outcome, satisfy that contract.
    """
    matches = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, dict)
        and isinstance(outcome.get("episode"), dict)
        and outcome["episode"].get("episode_key") == episode_key
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Correlated recompile target episode outcome was not found exactly once"
        )
    status = matches[0].get("status")
    if status not in {"completed", "waiting_for_summary_review_activation"}:
        raise RuntimeError(
            "Correlated recompile target episode did not complete the full pipeline"
        )


def main() -> list[dict]:
    correlation = _recompile_correlation_from_environment()
    if correlation:
        result = run_pipeline(
            target_episode_key=correlation["episode_key"],
        )
    else:
        result = run_pipeline()
    if correlation:
        _require_completed_recompile_target(result, correlation["episode_key"])
        try:
            complete_recompile_request(
                correlation["episode_key"],
                request_id=correlation["request_id"],
                review_generation=correlation["review_generation"],
            )
        except RecompileAuditPersistenceError:
            LOGGER.warning(
                "Recompile terminal audit persistence deferred after a successful pipeline",
                extra={
                    "episode_key": correlation["episode_key"],
                    "request_id": correlation["request_id"],
                },
                exc_info=True,
            )
    return result


if __name__ == "__main__":
    print(main())
