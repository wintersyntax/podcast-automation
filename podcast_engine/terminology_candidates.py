"""Non-authoritative terminology candidate collection and review storage.

The active compiler terminology policy lives in :mod:`compiler.terminology`.
This module deliberately owns only *candidate* vocabulary. Candidates may be
suggested by AI/external sources or observed in real episodes, but no candidate
is visible to compiler matching, resolver acceptance, review suppression, or
canonical transcript selection until a human deliberately promotes it through a
separate, versioned compiler-policy change.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable, Sequence

from compiler.terminology import (
    ALLOWED_TERMINOLOGY_CATEGORIES,
    TERMINOLOGY_REGISTRY,
    TERMINOLOGY_REGISTRY_SHA256,
)


CANDIDATE_REGISTRY_SCHEMA_VERSION = 1
CANDIDATE_SEED_SCHEMA_VERSION = 1
CANDIDATE_REGISTRY_PATH = "knowledge/terminology/candidates-v1.json"
CANDIDATE_AUTHORITY = "none"
CANDIDATE_COMPILER_VISIBILITY = "none"
MAX_WRITE_RETRIES = 3
MAX_OBSERVED_FORMS = 12
MAX_EPISODE_KEYS = 100
MAX_EVIDENCE_SAMPLES = 20
MAX_TERM_WORDS = 8
MAX_TERM_CHARS = 120

ALLOWED_CANDIDATE_CATEGORIES = tuple(
    sorted(set(ALLOWED_TERMINOLOGY_CATEGORIES) | {"proper_name"})
)
ALLOWED_CANDIDATE_PROVENANCE = (
    "ai-suggested",
    "corpus-confirmed",
    "human-reviewed",
    "external-curated",
)
ALLOWED_CANDIDATE_STATUSES = ("candidate", "rejected", "promoted")

_SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class CandidateObservation:
    """One non-authoritative observation of a possible terminology entry."""

    text: str
    suggested_category: str
    provenance: str
    episode_key: str | None = None
    observed_form: str | None = None
    evidence: dict[str, Any] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_candidate_text(value: str) -> str:
    """Return the stable case-insensitive candidate identity key."""

    if not isinstance(value, str):
        raise ValueError("candidate text must be a string")
    if "\n" in value or "\r" in value:
        raise ValueError("candidate text must be a single line")
    cleaned = _SPACE_PATTERN.sub(" ", value.strip())
    if not cleaned:
        raise ValueError("candidate text must not be blank")
    return cleaned.casefold()


def candidate_is_term_like(value: object) -> bool:
    """Apply a deliberately small phrase boundary for automatic collection."""

    if not isinstance(value, str):
        return False
    cleaned = _SPACE_PATTERN.sub(" ", value.strip())
    if not cleaned or len(cleaned) > MAX_TERM_CHARS:
        return False
    words = cleaned.split(" ")
    return 1 <= len(words) <= MAX_TERM_WORDS


def candidate_id_for(value: str) -> str:
    normalized = normalize_candidate_text(value)
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _active_keys() -> set[str]:
    keys: set[str] = set()
    for entry in TERMINOLOGY_REGISTRY:
        keys.add(normalize_candidate_text(entry.canonical))
        keys.update(normalize_candidate_text(alias) for alias in entry.aliases)
    return keys


ACTIVE_TERMINOLOGY_KEYS = frozenset(_active_keys())


def initial_candidate_registry(*, updated_at: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": CANDIDATE_REGISTRY_SCHEMA_VERSION,
        "updated_at": updated_at or _now_iso(),
        "authority": CANDIDATE_AUTHORITY,
        "compiler_visibility": CANDIDATE_COMPILER_VISIBILITY,
        "active_registry_sha256": TERMINOLOGY_REGISTRY_SHA256,
        "candidates": {},
    }


def _clean_string_list(values: object, *, field: str, maximum: int) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} values must be non-empty strings")
        cleaned = _SPACE_PATTERN.sub(" ", value.strip())
        key = cleaned.casefold()
        if key not in seen:
            output.append(cleaned)
            seen.add(key)
    if len(output) > maximum:
        raise ValueError(f"{field} exceeds the maximum of {maximum}")
    return output


def validate_candidate_registry(payload: object) -> dict[str, Any]:
    """Validate the persisted candidate registry and fail closed on drift."""

    if not isinstance(payload, dict):
        raise ValueError("candidate registry must be a JSON object")
    if payload.get("schema_version") != CANDIDATE_REGISTRY_SCHEMA_VERSION:
        raise ValueError("candidate registry schema is unsupported")
    if payload.get("authority") != CANDIDATE_AUTHORITY:
        raise ValueError("candidate registry must remain non-authoritative")
    if payload.get("compiler_visibility") != CANDIDATE_COMPILER_VISIBILITY:
        raise ValueError("candidate registry must remain invisible to compiler decisions")
    if not isinstance(payload.get("updated_at"), str) or not payload["updated_at"]:
        raise ValueError("candidate registry updated_at is required")
    if not isinstance(payload.get("active_registry_sha256"), str):
        raise ValueError("candidate registry active_registry_sha256 is required")
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("candidate registry candidates must be an object")

    seen_normalized: set[str] = set()
    for candidate_id, entry in candidates.items():
        if not isinstance(candidate_id, str) or not candidate_id.startswith("sha256:"):
            raise ValueError("candidate IDs must be sha256 identities")
        if not isinstance(entry, dict):
            raise ValueError(f"candidate {candidate_id} must be an object")
        canonical = entry.get("canonical")
        if not candidate_is_term_like(canonical):
            raise ValueError(f"candidate {candidate_id} canonical text is invalid")
        normalized = normalize_candidate_text(canonical)
        if entry.get("normalized") != normalized:
            raise ValueError(f"candidate {candidate_id} normalized identity is invalid")
        if candidate_id_for(canonical) != candidate_id:
            raise ValueError(f"candidate {candidate_id} hash identity is invalid")
        if entry.get("candidate_id") != candidate_id:
            raise ValueError(f"candidate {candidate_id} embedded identity is invalid")
        if normalized in seen_normalized:
            raise ValueError("candidate registry contains duplicate normalized identities")
        seen_normalized.add(normalized)
        # Historical candidate entries may later become active terminology.
        # Retaining that curation history must not require a candidate-store
        # migration when compiler policy is promoted. New observations for an
        # already-active term are still skipped by merge_candidate_observation().
        status = entry.get("status")
        if status not in ALLOWED_CANDIDATE_STATUSES:
            raise ValueError(f"candidate {candidate_id} has unsupported status")
        if entry.get("authoritative") is not False or entry.get("compiler_visible") is not False:
            raise ValueError(f"candidate {candidate_id} must remain non-authoritative")
        if not isinstance(entry.get("observation_count"), int) or entry["observation_count"] < 1:
            raise ValueError(f"candidate {candidate_id} observation_count is invalid")
        if not isinstance(entry.get("first_seen_at"), str) or not entry["first_seen_at"]:
            raise ValueError(f"candidate {candidate_id} first_seen_at is required")
        if not isinstance(entry.get("last_seen_at"), str) or not entry["last_seen_at"]:
            raise ValueError(f"candidate {candidate_id} last_seen_at is required")

        suggested = entry.get("suggested_categories")
        if not isinstance(suggested, dict) or not suggested:
            raise ValueError(f"candidate {candidate_id} suggested_categories is required")
        for category, count in suggested.items():
            if category not in ALLOWED_CANDIDATE_CATEGORIES:
                raise ValueError(f"candidate {candidate_id} category {category!r} is unsupported")
            if not isinstance(count, int) or count < 1:
                raise ValueError(f"candidate {candidate_id} category count is invalid")

        provenance = entry.get("provenance_counts")
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError(f"candidate {candidate_id} provenance_counts is required")
        for source, count in provenance.items():
            if source not in ALLOWED_CANDIDATE_PROVENANCE:
                raise ValueError(f"candidate {candidate_id} provenance {source!r} is unsupported")
            if not isinstance(count, int) or count < 1:
                raise ValueError(f"candidate {candidate_id} provenance count is invalid")

        _clean_string_list(
            entry.get("observed_forms", []),
            field=f"candidate {candidate_id} observed_forms",
            maximum=MAX_OBSERVED_FORMS,
        )
        _clean_string_list(
            entry.get("episode_keys", []),
            field=f"candidate {candidate_id} episode_keys",
            maximum=MAX_EPISODE_KEYS,
        )
        samples = entry.get("evidence_samples", [])
        if not isinstance(samples, list) or len(samples) > MAX_EVIDENCE_SAMPLES:
            raise ValueError(f"candidate {candidate_id} evidence_samples is invalid")
        if not all(isinstance(sample, dict) for sample in samples):
            raise ValueError(f"candidate {candidate_id} evidence_samples must contain objects")

    return payload


def validate_candidate_observation(observation: CandidateObservation) -> CandidateObservation:
    if not isinstance(observation, CandidateObservation):
        raise ValueError("candidate observation must be CandidateObservation")
    if not candidate_is_term_like(observation.text):
        raise ValueError("candidate observation is not a bounded terminology phrase")
    if observation.suggested_category not in ALLOWED_CANDIDATE_CATEGORIES:
        raise ValueError("candidate observation has unsupported category")
    if observation.provenance not in ALLOWED_CANDIDATE_PROVENANCE:
        raise ValueError("candidate observation has unsupported provenance")
    if observation.episode_key is not None and (
        not isinstance(observation.episode_key, str) or not observation.episode_key.strip()
    ):
        raise ValueError("candidate observation episode_key must be non-empty when supplied")
    if observation.observed_form is not None and not candidate_is_term_like(
        observation.observed_form
    ):
        raise ValueError("candidate observed_form is not a bounded terminology phrase")
    if observation.evidence is not None and not isinstance(observation.evidence, dict):
        raise ValueError("candidate evidence must be an object when supplied")
    return observation


def _append_unique_bounded(values: list[str], value: str, maximum: int) -> None:
    key = value.casefold()
    if any(existing.casefold() == key for existing in values):
        return
    if len(values) < maximum:
        values.append(value)


def merge_candidate_observation(
    registry: dict[str, Any],
    observation: CandidateObservation,
    *,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Merge one observation into a validated mutable registry object.

    Active terminology collisions are deliberately ignored rather than copied
    into the candidate store.  The returned result makes that outcome explicit.
    """

    validate_candidate_registry(registry)
    validate_candidate_observation(observation)
    normalized = normalize_candidate_text(observation.text)
    if normalized in ACTIVE_TERMINOLOGY_KEYS:
        return {
            "status": "already_active",
            "canonical": observation.text.strip(),
            "candidate_id": None,
        }

    candidate_id = candidate_id_for(observation.text)
    now = observed_at or _now_iso()
    candidates = registry["candidates"]
    entry = candidates.get(candidate_id)
    if entry is None:
        canonical = _SPACE_PATTERN.sub(" ", observation.text.strip())
        entry = {
            "candidate_id": candidate_id,
            "canonical": canonical,
            "normalized": normalized,
            "status": "candidate",
            "authoritative": False,
            "compiler_visible": False,
            "first_seen_at": now,
            "last_seen_at": now,
            "observation_count": 0,
            "suggested_categories": {},
            "provenance_counts": {},
            "observed_forms": [],
            "episode_keys": [],
            "evidence_samples": [],
        }
        candidates[candidate_id] = entry

    entry["last_seen_at"] = now
    entry["observation_count"] += 1
    categories = entry["suggested_categories"]
    categories[observation.suggested_category] = categories.get(
        observation.suggested_category, 0
    ) + 1
    provenance = entry["provenance_counts"]
    provenance[observation.provenance] = provenance.get(observation.provenance, 0) + 1

    observed_form = observation.observed_form or observation.text
    _append_unique_bounded(
        entry["observed_forms"],
        _SPACE_PATTERN.sub(" ", observed_form.strip()),
        MAX_OBSERVED_FORMS,
    )
    if observation.episode_key:
        _append_unique_bounded(
            entry["episode_keys"], observation.episode_key.strip(), MAX_EPISODE_KEYS
        )
    if observation.evidence and len(entry["evidence_samples"]) < MAX_EVIDENCE_SAMPLES:
        sample = deepcopy(observation.evidence)
        sample.setdefault("provenance", observation.provenance)
        sample.setdefault("observed_at", now)
        if observation.episode_key:
            sample.setdefault("episode_key", observation.episode_key.strip())
        entry["evidence_samples"].append(sample)

    registry["updated_at"] = now
    registry["active_registry_sha256"] = TERMINOLOGY_REGISTRY_SHA256
    validate_candidate_registry(registry)
    return {
        "status": "recorded",
        "canonical": entry["canonical"],
        "candidate_id": candidate_id,
        "observation_count": entry["observation_count"],
    }


def _entry_has_seed_id(entry: dict[str, Any] | None, seed_id: str) -> bool:
    if not isinstance(entry, dict):
        return False
    return any(
        isinstance(sample, dict)
        and sample.get("seed_id") == seed_id
        and sample.get("provenance") == "ai-suggested"
        for sample in entry.get("evidence_samples", [])
    )


def merge_seed_candidate_observation(
    registry: dict[str, Any],
    observation: CandidateObservation,
    *,
    seed_id: str,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Merge one AI-seed observation exactly once per seed identity.

    Re-running ``seed-ai`` must not inflate observation/category/provenance
    counts. The seed identity is already persisted in the evidence sample and
    therefore doubles as the idempotency key.
    """

    if not isinstance(seed_id, str) or not seed_id.strip():
        raise ValueError("terminology candidate seed_id is required")
    validate_candidate_registry(registry)
    validate_candidate_observation(observation)
    normalized = normalize_candidate_text(observation.text)
    if normalized in ACTIVE_TERMINOLOGY_KEYS:
        return {
            "status": "already_active",
            "canonical": observation.text.strip(),
            "candidate_id": None,
        }
    candidate_id = candidate_id_for(observation.text)
    if _entry_has_seed_id(registry["candidates"].get(candidate_id), seed_id):
        return {
            "status": "already_seeded",
            "canonical": observation.text.strip(),
            "candidate_id": candidate_id,
        }
    evidence = deepcopy(observation.evidence) if observation.evidence else {}
    evidence["seed_id"] = seed_id.strip()
    seeded = CandidateObservation(
        text=observation.text,
        suggested_category=observation.suggested_category,
        provenance=observation.provenance,
        episode_key=observation.episode_key,
        observed_form=observation.observed_form,
        evidence=evidence,
    )
    return merge_candidate_observation(registry, seeded, observed_at=observed_at)


def repair_duplicate_seed_observations(
    registry: dict[str, Any],
    observations: Iterable[CandidateObservation],
    *,
    seed_id: str,
) -> dict[str, int]:
    """Collapse duplicate observations from repeated runs of one AI seed.

    This repairs registries created before ``seed-ai`` became idempotent. Only
    duplicate evidence samples carrying the exact seed identity are removed;
    corpus, Human Review, and external-curated observations are untouched.
    """

    if not isinstance(seed_id, str) or not seed_id.strip():
        raise ValueError("terminology candidate seed_id is required")
    validate_candidate_registry(registry)
    by_id = {candidate_id_for(obs.text): obs for obs in observations}
    repaired_candidates = 0
    removed_observations = 0
    for candidate_id, observation in by_id.items():
        entry = registry["candidates"].get(candidate_id)
        if not isinstance(entry, dict):
            continue
        matching_indexes = [
            index
            for index, sample in enumerate(entry.get("evidence_samples", []))
            if isinstance(sample, dict)
            and sample.get("seed_id") == seed_id
            and sample.get("provenance") == observation.provenance
        ]
        if len(matching_indexes) <= 1:
            continue
        extra = len(matching_indexes) - 1
        keep_index = matching_indexes[0]
        entry["evidence_samples"] = [
            sample
            for index, sample in enumerate(entry["evidence_samples"])
            if index == keep_index or index not in matching_indexes
        ]
        entry["observation_count"] -= extra
        category = observation.suggested_category
        entry["suggested_categories"][category] -= extra
        if entry["suggested_categories"][category] <= 0:
            del entry["suggested_categories"][category]
        provenance = observation.provenance
        entry["provenance_counts"][provenance] -= extra
        if entry["provenance_counts"][provenance] <= 0:
            del entry["provenance_counts"][provenance]
        repaired_candidates += 1
        removed_observations += extra
    validate_candidate_registry(registry)
    return {
        "repaired_candidates": repaired_candidates,
        "removed_observations": removed_observations,
    }


class CandidateRegistryStore:
    """Durable GCS candidate registry with optimistic concurrency.

    This object is intentionally outside the compiler. Candidate persistence is
    observability/curation state and must never become a prerequisite for a
    successful compile or Human Review decision.
    """

    def __init__(self, *, bucket: Any | None = None, path: str = CANDIDATE_REGISTRY_PATH):
        if bucket is None:
            from .storage import get_bucket

            bucket = get_bucket()
        self.bucket = bucket
        self.path = path

    @staticmethod
    def _precondition_failed_type():
        from google.api_core.exceptions import PreconditionFailed

        return PreconditionFailed

    def _read(self) -> tuple[dict[str, Any] | None, int]:
        blob = self.bucket.blob(self.path)
        if not blob.exists():
            return None, 0
        PreconditionFailed = self._precondition_failed_type()
        for _ in range(MAX_WRITE_RETRIES):
            blob.reload()
            generation = int(blob.generation)
            try:
                content = blob.download_as_text(
                    encoding="utf-8",
                    if_generation_match=generation,
                )
            except PreconditionFailed:
                continue
            try:
                value = json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"{self.path} is not valid JSON") from error
            validate_candidate_registry(value)
            return value, generation
        raise RuntimeError(f"{self.path} changed repeatedly while reading snapshot")

    def read_snapshot(self) -> dict[str, Any] | None:
        """Return the current registry without creating or mutating it.

        Read-only consumers such as shadow retrieval must not initialize durable
        state merely because they were invoked.
        """

        value, _ = self._read()
        return deepcopy(value) if value is not None else None

    def load(self) -> dict[str, Any]:
        value, _ = self._read()
        if value is not None:
            return value
        PreconditionFailed = self._precondition_failed_type()
        for _ in range(MAX_WRITE_RETRIES):
            value = initial_candidate_registry()
            try:
                self.bucket.blob(self.path).upload_from_string(
                    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    content_type="application/json",
                    if_generation_match=0,
                )
                return value
            except PreconditionFailed:
                value, _ = self._read()
                if value is not None:
                    return value
        raise RuntimeError(f"{self.path} changed repeatedly while initializing")

    def record(self, observation: CandidateObservation) -> dict[str, Any]:
        validate_candidate_observation(observation)
        PreconditionFailed = self._precondition_failed_type()
        for _ in range(MAX_WRITE_RETRIES):
            registry, generation = self._read()
            if registry is None:
                self.load()
                continue
            working = deepcopy(registry)
            result = merge_candidate_observation(working, observation)
            if result["status"] == "already_active":
                return result
            try:
                self.bucket.blob(self.path).upload_from_string(
                    json.dumps(working, ensure_ascii=False, indent=2) + "\n",
                    content_type="application/json",
                    if_generation_match=generation,
                )
                return result
            except PreconditionFailed:
                continue
        raise RuntimeError(f"{self.path} changed repeatedly while recording a candidate")

    def record_many(self, observations: Iterable[CandidateObservation]) -> dict[str, int]:
        batch = list(observations)
        for observation in batch:
            validate_candidate_observation(observation)
        counts = {"recorded": 0, "already_active": 0}
        if not batch:
            return counts
        PreconditionFailed = self._precondition_failed_type()
        for _ in range(MAX_WRITE_RETRIES):
            registry, generation = self._read()
            if registry is None:
                self.load()
                continue
            working = deepcopy(registry)
            counts = {"recorded": 0, "already_active": 0}
            for observation in batch:
                result = merge_candidate_observation(working, observation)
                counts[result["status"]] += 1
            if counts["recorded"] == 0:
                return counts
            try:
                self.bucket.blob(self.path).upload_from_string(
                    json.dumps(working, ensure_ascii=False, indent=2) + "\n",
                    content_type="application/json",
                    if_generation_match=generation,
                )
                return counts
            except PreconditionFailed:
                continue
        raise RuntimeError(f"{self.path} changed repeatedly while recording candidates")


    def record_seed(
        self, observations: Iterable[CandidateObservation], *, seed_id: str
    ) -> dict[str, int]:
        batch = list(observations)
        for observation in batch:
            validate_candidate_observation(observation)
        counts = {"recorded": 0, "already_active": 0, "already_seeded": 0}
        if not batch:
            return counts
        PreconditionFailed = self._precondition_failed_type()
        for _ in range(MAX_WRITE_RETRIES):
            registry, generation = self._read()
            if registry is None:
                self.load()
                continue
            working = deepcopy(registry)
            counts = {"recorded": 0, "already_active": 0, "already_seeded": 0}
            for observation in batch:
                result = merge_seed_candidate_observation(
                    working, observation, seed_id=seed_id
                )
                counts[result["status"]] += 1
            if counts["recorded"] == 0:
                return counts
            try:
                self.bucket.blob(self.path).upload_from_string(
                    json.dumps(working, ensure_ascii=False, indent=2) + "\n",
                    content_type="application/json",
                    if_generation_match=generation,
                )
                return counts
            except PreconditionFailed:
                continue
        raise RuntimeError(f"{self.path} changed repeatedly while seeding candidates")

    def repair_seed_duplicates(
        self, observations: Iterable[CandidateObservation], *, seed_id: str
    ) -> dict[str, int]:
        batch = list(observations)
        PreconditionFailed = self._precondition_failed_type()
        for _ in range(MAX_WRITE_RETRIES):
            registry, generation = self._read()
            if registry is None:
                self.load()
                continue
            working = deepcopy(registry)
            result = repair_duplicate_seed_observations(
                working, batch, seed_id=seed_id
            )
            if result["removed_observations"] == 0:
                return result
            working["updated_at"] = _now_iso()
            try:
                self.bucket.blob(self.path).upload_from_string(
                    json.dumps(working, ensure_ascii=False, indent=2) + "\n",
                    content_type="application/json",
                    if_generation_match=generation,
                )
                return result
            except PreconditionFailed:
                continue
        raise RuntimeError(f"{self.path} changed repeatedly while repairing seed duplicates")


def compiler_candidate_observations(
    *, episode_key: str, report: dict[str, Any]
) -> list[CandidateObservation]:
    """Extract only strong Level-2 same-episode confirmations as candidates."""

    memory = report.get("episode_local_memory")
    if not isinstance(memory, dict) or memory.get("mode") != "shadow":
        return []
    observations: list[CandidateObservation] = []
    for signal in memory.get("signals", []):
        if not isinstance(signal, dict):
            continue
        category = signal.get("category")
        if category not in ALLOWED_CANDIDATE_CATEGORIES:
            continue
        difference_id = signal.get("difference_id")
        for form in signal.get("forms", []):
            if not isinstance(form, dict) or form.get("episode_confirmed") is not True:
                continue
            text = form.get("text")
            if not candidate_is_term_like(text):
                continue
            observations.append(
                CandidateObservation(
                    text=text,
                    suggested_category=category,
                    provenance="corpus-confirmed",
                    episode_key=episode_key,
                    observed_form=text,
                    evidence={
                        "source": form.get("source"),
                        "difference_id": difference_id,
                        "signal": signal.get("signal"),
                        "evidence_contract": memory.get("evidence_contract"),
                    },
                )
            )
    return observations


def human_review_candidate_observation(
    *,
    episode_key: str,
    review_item: dict[str, Any],
    chosen_text: str,
) -> CandidateObservation | None:
    """Return a bounded Human Review candidate, or None when it is not term-like."""

    category = review_item.get("category") or review_item.get("resolver_category")
    if category not in ALLOWED_CANDIDATE_CATEGORIES:
        return None
    if not candidate_is_term_like(chosen_text):
        return None
    return CandidateObservation(
        text=chosen_text,
        suggested_category=category,
        provenance="human-reviewed",
        episode_key=episode_key,
        observed_form=chosen_text,
        evidence={
            "difference_id": review_item.get("id"),
            "reason": review_item.get("reason"),
            "kind": review_item.get("kind"),
            "severity": review_item.get("severity"),
        },
    )


def human_review_candidate_observations(
    *, episode_key: str, review_record: dict[str, Any]
) -> list[CandidateObservation]:
    """Extract bounded terminology candidates from already-durable review truth."""

    observations: list[CandidateObservation] = []
    for decision in review_record.get("human_decisions", []):
        if not isinstance(decision, dict) or decision.get("reviewed_by") != "human":
            continue
        review_item = decision.get("review_item")
        chosen_text = decision.get("chosen_text")
        if not isinstance(review_item, dict) or not isinstance(chosen_text, str):
            continue
        observation = human_review_candidate_observation(
            episode_key=episode_key,
            review_item=review_item,
            chosen_text=chosen_text,
        )
        if observation is not None:
            observations.append(observation)
    return observations


def record_compiler_candidates_best_effort(
    *,
    episode_key: str,
    report: dict[str, Any],
    review_record: dict[str, Any] | None = None,
    store: CandidateRegistryStore | None = None,
) -> dict[str, int]:
    """Batch non-authoritative candidates after canonical compilation succeeds."""

    observations = compiler_candidate_observations(episode_key=episode_key, report=report)
    if isinstance(review_record, dict):
        observations.extend(
            human_review_candidate_observations(
                episode_key=episode_key, review_record=review_record
            )
        )
    if not observations:
        return {"recorded": 0, "already_active": 0, "errors": 0}
    try:
        counts = (store or CandidateRegistryStore()).record_many(observations)
        return {**counts, "errors": 0}
    except Exception as error:  # Candidate telemetry must never block canonical work.
        print(f"WARNING: terminology candidate collection failed: {error}")
        return {"recorded": 0, "already_active": 0, "errors": len(observations)}


def validate_seed_payload(payload: object) -> list[CandidateObservation]:
    """Validate an AI/external seed artifact without granting it authority."""

    if not isinstance(payload, dict) or payload.get("schema_version") != CANDIDATE_SEED_SCHEMA_VERSION:
        raise ValueError("terminology candidate seed schema is unsupported")
    if payload.get("authority") != "none":
        raise ValueError("terminology candidate seed must be non-authoritative")
    if payload.get("compiler_visibility") != "none":
        raise ValueError("terminology candidate seed must be compiler-invisible")
    seed_id = payload.get("seed_id")
    if not isinstance(seed_id, str) or not seed_id.strip():
        raise ValueError("terminology candidate seed_id is required")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("terminology candidate seed candidates must be a list")
    observations: list[CandidateObservation] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            raise ValueError(f"seed candidate {index} must be an object")
        provenance = item.get("provenance")
        observation = CandidateObservation(
            text=item.get("canonical"),
            suggested_category=item.get("suggested_category"),
            provenance=provenance,
            observed_form=item.get("canonical"),
            evidence={
                "seed_id": payload.get("seed_id"),
            },
        )
        validate_candidate_observation(observation)
        normalized = normalize_candidate_text(observation.text)
        if normalized in seen:
            raise ValueError(f"seed contains duplicate candidate {observation.text!r}")
        seen.add(normalized)
        observations.append(observation)
    return observations


def candidate_registry_stats(registry: dict[str, Any]) -> dict[str, Any]:
    validate_candidate_registry(registry)
    by_status: dict[str, int] = {}
    by_provenance: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for entry in registry["candidates"].values():
        by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
        for provenance in entry["provenance_counts"]:
            by_provenance[provenance] = by_provenance.get(provenance, 0) + 1
        if entry["suggested_categories"]:
            category = max(
                entry["suggested_categories"],
                key=lambda name: (entry["suggested_categories"][name], name),
            )
            by_category[category] = by_category.get(category, 0) + 1
    return {
        "candidate_count": len(registry["candidates"]),
        "by_status": dict(sorted(by_status.items())),
        "by_provenance": dict(sorted(by_provenance.items())),
        "by_primary_suggested_category": dict(sorted(by_category.items())),
        "authority": registry["authority"],
        "compiler_visibility": registry["compiler_visibility"],
    }
