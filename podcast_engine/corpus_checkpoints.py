"""Versioned corpus-checkpoint policy and comparable-episode evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SCHEMA_VERSION = 1
EXPECTED_THRESHOLDS = (25, 50, 100)
ALLOWED_STATUSES = frozenset({"pending", "analysis_in_progress", "accepted"})


@dataclass(frozen=True)
class EpisodeEvidence:
    """Comparable-corpus evidence for one canonical episode."""

    episode_key: str
    comparable: bool
    level2_present: bool
    alignment_present: bool
    reason: str


@dataclass(frozen=True)
class CorpusEvidence:
    """Unique episode keys carrying current checkpoint evidence."""

    eligible_episode_keys: tuple[str, ...]
    level2_episode_keys: tuple[str, ...]
    alignment_episode_keys: tuple[str, ...]

    @property
    def eligible_count(self) -> int:
        return len(self.eligible_episode_keys)


@dataclass(frozen=True)
class CheckpointDecision:
    """Derived checkpoint state for one comparable-episode count."""

    blocking_threshold: int | None
    reached_unaccepted: tuple[int, ...]
    next_threshold: int | None


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_checkpoint_policy(policy: object) -> dict:
    if not isinstance(policy, dict):
        raise ValueError("Corpus checkpoint policy must be a JSON object")
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported corpus checkpoint schema version "
            f"{policy.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    if not _non_empty_string(policy.get("eligibility_policy")):
        raise ValueError("eligibility_policy must be a non-empty string")
    required_observers = policy.get("required_observers")
    if not isinstance(required_observers, dict) or not required_observers:
        raise ValueError("required_observers must be a non-empty object")

    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, list):
        raise ValueError("thresholds must be a list")
    counts = tuple(
        threshold.get("count") if isinstance(threshold, dict) else None
        for threshold in thresholds
    )
    if counts != EXPECTED_THRESHOLDS:
        raise ValueError(
            f"threshold counts must be exactly {EXPECTED_THRESHOLDS}; got {counts}"
        )

    for threshold in thresholds:
        status = threshold.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Unsupported checkpoint status: {status!r}")

        count = threshold["count"]
        accepted_at_count = threshold.get("accepted_at_count")
        accepted_commit = threshold.get("accepted_commit")
        evidence_ref = threshold.get("evidence_ref")

        if status == "accepted":
            if (
                isinstance(accepted_at_count, bool)
                or not isinstance(accepted_at_count, int)
                or accepted_at_count < count
                or not _non_empty_string(accepted_commit)
                or not _non_empty_string(evidence_ref)
            ):
                raise ValueError(
                    f"Accepted checkpoint {count} requires complete acceptance evidence"
                )
        elif any(
            value is not None
            for value in (accepted_at_count, accepted_commit, evidence_ref)
        ):
            raise ValueError(
                f"Unaccepted checkpoint {count} cannot contain acceptance evidence"
            )

    return policy


def load_checkpoint_policy(path: str | Path) -> dict:
    """Load and strictly validate a versioned checkpoint policy file."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate_checkpoint_policy(value)


def observer_contract_matches(
    report: dict,
    observer_name: str,
    expected: dict,
) -> bool:
    """Return whether one report observer contains every required policy field."""

    if not isinstance(report, dict) or not isinstance(expected, dict):
        return False
    observer = report.get(observer_name)
    if not isinstance(observer, dict):
        return False
    return all(observer.get(key) == value for key, value in expected.items())


def _state(episode: dict, name: str) -> object:
    status = episode.get("status")
    if not isinstance(status, dict):
        return None
    value = status.get(name)
    if not isinstance(value, dict):
        return None
    return value.get("state")


def _report_path(episode: dict) -> str | None:
    files = episode.get("files")
    if not isinstance(files, dict):
        return None
    compiled = files.get("compiled")
    if not isinstance(compiled, dict):
        return None
    value = compiled.get("report")
    return value if _non_empty_string(value) else None


def evaluate_episode(
    episode: dict,
    report: dict | None,
    policy: dict,
) -> EpisodeEvidence:
    """Evaluate whether one episode satisfies the current comparable contract."""

    _validate_checkpoint_policy(policy)
    if not isinstance(episode, dict):
        raise ValueError("Episode must be an object")

    raw_episode_key = episode.get("episode_key")
    episode_key = raw_episode_key if isinstance(raw_episode_key, str) else ""
    required_observers = policy["required_observers"]
    level2_present = observer_contract_matches(
        report or {},
        "episode_local_memory",
        required_observers.get("episode_local_memory", {}),
    )
    alignment_present = observer_contract_matches(
        report or {},
        "alignment_diagnostics",
        required_observers.get("alignment_diagnostics", {}),
    )

    if not _non_empty_string(raw_episode_key):
        reason = "missing_episode_key"
    elif _state(episode, "apple_transcript") != "ready":
        reason = "apple_not_ready"
    elif _state(episode, "whisper") != "ready":
        reason = "whisper_not_ready"
    elif _state(episode, "compiler") != "completed":
        reason = "compiler_not_completed"
    elif _report_path(episode) is None:
        reason = "missing_compiler_report"
    elif not isinstance(report, dict):
        reason = "missing_report_payload"
    elif not level2_present:
        reason = "missing_episode_local_memory_contract"
    elif not alignment_present:
        reason = "missing_alignment_diagnostics_contract"
    else:
        reason = "comparable"

    return EpisodeEvidence(
        episode_key=episode_key,
        comparable=reason == "comparable",
        level2_present=level2_present,
        alignment_present=alignment_present,
        reason=reason,
    )


def evaluate_corpus(
    episodes: list[dict],
    report_loader: Callable[[str], dict],
    policy: dict,
) -> CorpusEvidence:
    """Evaluate unique canonical episodes without hiding report-read failures."""

    _validate_checkpoint_policy(policy)
    if not isinstance(episodes, list):
        raise ValueError("episodes must be a list")

    seen: set[str] = set()
    eligible: list[str] = []
    level2: list[str] = []
    alignment: list[str] = []

    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("Episode must be an object")

        raw_episode_key = episode.get("episode_key")
        episode_key = raw_episode_key if isinstance(raw_episode_key, str) else ""
        if episode_key in seen:
            continue
        seen.add(episode_key)

        report_path = _report_path(episode)
        needs_report = (
            _non_empty_string(raw_episode_key)
            and _state(episode, "apple_transcript") == "ready"
            and _state(episode, "whisper") == "ready"
            and _state(episode, "compiler") == "completed"
            and report_path is not None
        )
        report = report_loader(report_path) if needs_report else None
        evidence = evaluate_episode(episode, report, policy)

        if evidence.level2_present:
            level2.append(episode_key)
        if evidence.alignment_present:
            alignment.append(episode_key)
        if evidence.comparable:
            eligible.append(episode_key)

    return CorpusEvidence(
        eligible_episode_keys=tuple(eligible),
        level2_episode_keys=tuple(level2),
        alignment_episode_keys=tuple(alignment),
    )


def evaluate_thresholds(policy: dict, eligible_count: int) -> CheckpointDecision:
    """Derive blocking and next-threshold state from live comparable count."""

    _validate_checkpoint_policy(policy)
    if isinstance(eligible_count, bool) or not isinstance(eligible_count, int):
        raise ValueError("eligible_count must be a non-negative integer")
    if eligible_count < 0:
        raise ValueError("eligible_count must be a non-negative integer")

    thresholds = policy["thresholds"]
    reached_unaccepted = tuple(
        threshold["count"]
        for threshold in thresholds
        if threshold["count"] <= eligible_count
        and threshold["status"] != "accepted"
    )
    blocking_threshold = reached_unaccepted[0] if reached_unaccepted else None
    next_threshold = next(
        (
            threshold["count"]
            for threshold in thresholds
            if threshold["count"] > eligible_count
        ),
        None,
    )
    return CheckpointDecision(
        blocking_threshold=blocking_threshold,
        reached_unaccepted=reached_unaccepted,
        next_threshold=next_threshold,
    )
