"""Cloud-side Apple plus Whisper compilation and review stage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from compiler.review_policy import batch_recommendation, classify_review_anomaly
from compiler.transcript import (
    build_resolver_batch,
    build_triage_batch,
    clean_transcript,
    compile_transcripts,
    load_text,
    tokenize,
    write_outputs,
)

from .episode_contract import now_iso, paths_for
from .episode_generation import source_generation_fingerprint
from .human_review import (
    RecompileLifecycleInvariantError,
    link_recompile_result_generation,
    review_generation_fingerprint,
    review_queue_fingerprint,
    validated_human_resolutions,
)
from .worker_control import (
    RECOMPILE_EPISODE_KEY_ENV,
    RECOMPILE_REQUEST_ID_ENV,
    REVIEW_GENERATION_ENV,
)
from .preset_provenance import PresetProvenance, verify_transcript_reviewer
from .review import REVIEW_PRESET, resolve_compiler_batch, triage_compiler_batch
from .observability import emit_event
from .review_alignment import align_transcripts
from .terminology_candidates import record_compiler_candidates_best_effort
from .terminology_retrieval import attach_terminology_retrieval_shadow_best_effort
from .storage import (
    download_gcs_bytes,
    download_file_from_gcs,
    get_bucket,
    upload_path_to_gcs,
)


RUNTIME_DIR = (
    Path(__file__).resolve().parent.parent
    / "var"
    / "compiler"
)

REVIEW_RECORD_SCHEMA_VERSION = 4

# Bump this whenever the end-to-end review policy changes, including
# deterministic compiler classification or resolver eligibility.  The resolver
# record is valid only for the exact policy that produced its review lifecycle,
# not merely for an unchanged remote prompt.
REVIEW_POLICY_VERSION = "resolver-policy-v5"


def _recompile_correlation_for_episode(episode_key: str) -> dict[str, str] | None:
    """Return this worker's exact recompile context, if it has one."""

    values = {
        "request_id": os.environ.get(RECOMPILE_REQUEST_ID_ENV),
        "review_generation": os.environ.get(REVIEW_GENERATION_ENV),
        "episode_key": os.environ.get(RECOMPILE_EPISODE_KEY_ENV),
    }
    if not any(values.values()):
        return None
    if not all(isinstance(value, str) and value for value in values.values()):
        raise RecompileLifecycleInvariantError(
            "Recompile worker correlation environment is incomplete"
        )
    if values["episode_key"] != episode_key:
        return None
    return values


def _local_copy(
    gcs_path: str,
    destination: Path,
    *,
    refresh: bool = False,
) -> Path:
    """Materialize one GCS object locally.

    Compiler inputs use refresh=True so a warm container cannot accidentally
    reuse stale Apple or Whisper source bytes.
    """

    if (
        refresh
        or not destination.exists()
    ):
        download_file_from_gcs(
            gcs_path,
            destination,
        )

    return destination


def _optional_local_copy(
    gcs_path: str | None,
    destination: Path,
    *,
    refresh: bool = False,
) -> Path | None:
    return (
        _local_copy(
            gcs_path,
            destination,
            refresh=refresh,
        )
        if gcs_path
        else None
    )


def _file_digest(
    path: Path | None,
) -> str | None:
    if path is None:
        return None

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def _review_source_fingerprint(
    *,
    apple_text: bytes,
    apple_metadata: bytes | None,
    whisper_text: bytes,
    whisper_metadata: bytes | None,
) -> str:
    """Fingerprint the source bytes that determine a review lifecycle.

    Compatibility alias: delegates to the shared TASK-076 episode
    source-generation identity (podcast_engine.episode_generation) so
    Human Review and the episode AI budget ledger key off one payload/hash
    definition. Already-persisted records keep matching byte-for-byte.
    """

    return source_generation_fingerprint(
        apple_text=apple_text,
        apple_metadata=apple_metadata,
        whisper_text=whisper_text,
        whisper_metadata=whisper_metadata,
    )


def _review_source_fingerprint_from_paths(
    *,
    apple_text_path: Path,
    apple_metadata_path: Path | None,
    whisper_text_path: Path,
    whisper_metadata_path: Path | None,
) -> str:
    """Build the source portion of the review contract from local bytes."""

    return _review_source_fingerprint(
        apple_text=apple_text_path.read_bytes(),
        apple_metadata=(
            apple_metadata_path.read_bytes()
            if apple_metadata_path is not None
            else None
        ),
        whisper_text=whisper_text_path.read_bytes(),
        whisper_metadata=(
            whisper_metadata_path.read_bytes()
            if whisper_metadata_path is not None
            else None
        ),
    )


def review_record_is_current(episode: dict, record: dict | None) -> bool:
    """Return whether a durable review gate remains valid for this episode.

    This deliberately reads only the canonical source objects. It does not
    invoke the compiler or reviewer, so a current review gate remains cheap to
    preserve on scheduled runs.
    """

    if not _review_record_matches(record):
        return False

    sources = episode.get("files", {}).get("sources", {})
    apple = sources.get("apple", {})
    whisper = sources.get("whisper", {})
    apple_text_path = apple.get("text")
    whisper_text_path = whisper.get("text")
    if not isinstance(apple_text_path, str) or not isinstance(whisper_text_path, str):
        return False

    source_fingerprint = _review_source_fingerprint(
        apple_text=download_gcs_bytes(apple_text_path),
        apple_metadata=(
            download_gcs_bytes(apple["metadata"])
            if isinstance(apple.get("metadata"), str)
            else None
        ),
        whisper_text=download_gcs_bytes(whisper_text_path),
        whisper_metadata=(
            download_gcs_bytes(whisper["metadata"])
            if isinstance(whisper.get("metadata"), str)
            else None
        ),
    )
    return _review_record_matches(record, source_fingerprint=source_fingerprint)


def _review_input_fingerprint(
    *,
    apple_text_path: Path,
    apple_metadata_path: Path | None,
    whisper_text_path: Path,
    whisper_metadata_path: Path | None,
    resolver_batch: dict,
    preset_provenance: PresetProvenance,
    triage_batch: dict | None = None,
) -> str:
    """Fingerprint the exact inputs that can affect AI adjudication.

    Source bytes are included so a changed transcript automatically invalidates
    previous decisions.

    Resolver and advisory-triage request contracts are included so
    classification, focus, schema, review-context, or triage-input changes
    invalidate the old record too.

    Exact approved remote preset provenance is included so a changed immutable
    version is a cache miss even if a developer accidentally left the local
    policy version unchanged.
    """

    resolver_contract = {
        "batch": resolver_batch.get(
            "batch"
        ),
        "response_schema": resolver_batch.get(
            "response_schema"
        ),
    }
    triage_contract = {
        "batch": (triage_batch or {}).get(
            "batch"
        ),
        "response_schema": (triage_batch or {}).get(
            "response_schema"
        ),
        "deferred_ids": (triage_batch or {}).get(
            "deferred_ids"
        ),
    }

    payload = {
        "policy_version": (
            REVIEW_POLICY_VERSION
        ),
        "preset": REVIEW_PRESET,
        "preset_provenance": preset_provenance.cache_identity(),
        "sources": {
            "apple_text": (
                _file_digest(
                    apple_text_path
                )
            ),
            "apple_metadata": (
                _file_digest(
                    apple_metadata_path
                )
            ),
            "whisper_text": (
                _file_digest(
                    whisper_text_path
                )
            ),
            "whisper_metadata": (
                _file_digest(
                    whisper_metadata_path
                )
            ),
        },
        "resolver_contract": (
            resolver_contract
        ),
        "triage_contract": (
            triage_contract
        ),
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode(
        "utf-8"
    )

    return (
        "sha256:"
        + hashlib.sha256(
            encoded
        ).hexdigest()
    )


def _load_review_record(
    gcs_path: str,
) -> dict | None:
    blob = get_bucket().blob(
        gcs_path
    )

    if not blob.exists():
        return None

    try:
        payload = json.loads(
            blob.download_as_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        return None

    return (
        payload
        if isinstance(
            payload,
            dict,
        )
        else None
    )


def _review_record_matches(
    record: dict | None,
    input_fingerprint: str | None = None,
    source_fingerprint: str | None = None,
    preset_provenance: PresetProvenance | None = None,
) -> bool:
    if not isinstance(
        record,
        dict,
    ):
        return False

    return (
        record.get(
            "schema_version"
        )
        == REVIEW_RECORD_SCHEMA_VERSION

        and record.get(
            "policy_version"
        )
        == REVIEW_POLICY_VERSION

        and record.get(
            "reviewer_preset"
        )
        == REVIEW_PRESET

        and (
            input_fingerprint is None
            or record.get("input_fingerprint") == input_fingerprint
        )

        and (
            source_fingerprint is None
            or record.get("source_fingerprint") == source_fingerprint
        )

        and isinstance(
            record.get(
                "accepted"
            ),
            list,
        )

        and (
            preset_provenance is None
            or (
                preset_provenance.verified
                and isinstance(record.get("preset_provenance"), dict)
                and all(
                    record["preset_provenance"].get(field) == value
                    for field, value in preset_provenance.cache_identity().items()
                )
            )
        )

        and isinstance(
            record.get(
                "resolver_review"
            ),
            list,
        )
    )


def _resolution_from_record(
    record: dict,
) -> dict[str, list[dict]]:
    return {
        "accepted": [
            dict(item)
            for item in record.get(
                "accepted",
                [],
            )
            if isinstance(
                item,
                dict,
            )
        ],
        "review": [
            dict(item)
            for item in record.get(
                "resolver_review",
                [],
            )
            if isinstance(
                item,
                dict,
            )
        ],
        "outcomes": [
            dict(item)
            for item in record.get(
                "resolver_outcomes",
                [],
            )
            if isinstance(item, dict)
        ],
    }


def _triage_from_record(record: dict) -> dict:
    triage = {
        "accepted": [],
        "recommendations": [
            dict(item)
            for item in record.get(
                "triage_recommendations",
                [],
            )
            if isinstance(item, dict)
        ],
        "outcomes": [
            dict(item)
            for item in record.get(
                "triage_outcomes",
                [],
            )
            if isinstance(item, dict)
        ],
    }
    completion_metadata = record.get("triage_completion_metadata")
    if isinstance(completion_metadata, dict):
        triage["completion_metadata"] = dict(completion_metadata)

    failure_metadata = _safe_triage_failure_metadata(
        record.get("triage_failure_metadata")
    )
    if failure_metadata:
        triage["failure_metadata"] = failure_metadata

    return triage


def _safe_triage_failure_metadata(
    value,
) -> dict | list[dict]:
    """Keep only bounded non-sensitive advisory-triage failure evidence."""

    if isinstance(value, list):
        safe_chunks: list[dict] = []

        for chunk in value:
            if not isinstance(chunk, dict):
                continue

            item_ids = chunk.get("item_ids")
            if (
                not isinstance(item_ids, list)
                or not item_ids
                or len(item_ids) > 100
                or any(
                    not isinstance(item_id, int)
                    or isinstance(item_id, bool)
                    for item_id in item_ids
                )
            ):
                continue

            safe_chunk = {
                "item_ids": list(item_ids),
            }

            error_type = chunk.get("error_type")
            if (
                isinstance(error_type, str)
                and 1 <= len(error_type) <= 100
            ):
                safe_chunk["error_type"] = error_type

            http_status = chunk.get("http_status")
            if (
                isinstance(http_status, int)
                and not isinstance(http_status, bool)
                and 100 <= http_status <= 599
            ):
                safe_chunk["http_status"] = http_status

            safe_envelope = _safe_triage_failure_metadata(
                {
                    field: chunk.get(field)
                    for field in (
                        "finish_reason",
                        "completion_id",
                        "served_model",
                        "served_provider",
                        "content_chars",
                        "content_sha256",
                    )
                }
            )
            if isinstance(safe_envelope, dict):
                safe_chunk.update(safe_envelope)

            safe_chunks.append(safe_chunk)

        return safe_chunks

    if not isinstance(value, dict):
        return {}

    safe: dict = {}

    http_status = value.get("http_status")
    if (
        isinstance(http_status, int)
        and not isinstance(http_status, bool)
        and 100 <= http_status <= 599
    ):
        safe["http_status"] = http_status

    for field in (
        "finish_reason",
        "completion_id",
        "served_model",
        "served_provider",
    ):
        field_value = value.get(field)
        if isinstance(field_value, str) and field_value:
            safe[field] = field_value

    content_chars = value.get("content_chars")
    if (
        isinstance(content_chars, int)
        and not isinstance(content_chars, bool)
        and content_chars >= 0
    ):
        safe["content_chars"] = content_chars

    content_sha256 = value.get("content_sha256")
    if (
        isinstance(content_sha256, str)
        and len(content_sha256) == 64
        and all(character in "0123456789abcdef" for character in content_sha256)
    ):
        safe["content_sha256"] = content_sha256

    return safe


def _triage_error_metadata(error: Exception) -> dict | None:
    """Keep parse diagnostics plus, for a rejected request, only its status."""

    diagnostics = getattr(error, "diagnostics", None)
    metadata = dict(diagnostics) if isinstance(diagnostics, dict) else {}
    status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        metadata["http_status"] = status
    return metadata or None


def triage_degradation(triage: dict) -> dict | None:
    """Summarize advisory-triage unavailability without any transcript text.

    TASK-123: a whole triage run failing silently (every card falling back to
    ``triage_unavailable``) disables batch approval and assisted agreement
    without anyone noticing. Callers use this bounded summary to make that
    degradation visible; it never changes review routing or authority.
    """

    outcomes = [
        outcome
        for outcome in (triage or {}).get("outcomes", [])
        if isinstance(outcome, dict)
    ]
    unavailable = sum(
        1 for outcome in outcomes if outcome.get("status") == "unavailable"
    )
    if not unavailable:
        return None
    statuses: set[int] = set()
    failure_metadata = (triage or {}).get("failure_metadata")
    chunks = failure_metadata if isinstance(failure_metadata, list) else [failure_metadata]
    for chunk in chunks:
        if isinstance(chunk, dict):
            status = chunk.get("http_status")
            if isinstance(status, int) and not isinstance(status, bool):
                statuses.add(status)
    return {
        "unavailable": unavailable,
        "total": len(outcomes),
        "http_statuses": sorted(statuses),
    }


def _emit_triage_degradation(episode_key: str, triage: dict) -> None:
    summary = triage_degradation(triage)
    if summary is None:
        return
    emit_event(
        "advisory_triage_degraded",
        (
            f"Advisory triage unavailable for {summary['unavailable']} of "
            f"{summary['total']} review cards"
        ),
        severity="ERROR",
        episode_key=episode_key,
        **summary,
    )


def _unavailable_triage(
    triage_batch: dict,
    *,
    failure_metadata: dict | None = None,
) -> dict:
    result = {
        "accepted": [],
        "recommendations": [],
        "outcomes": [
            {
                "id": item["id"],
                "status": "unavailable",
                "recommendation": "needs_audio",
                "reason": "triage_unavailable",
            }
            for item in triage_batch.get("batch", {}).get("diff_items", [])
            if isinstance(item, dict) and isinstance(item.get("id"), int)
        ],
    }

    safe_failure_metadata = _safe_triage_failure_metadata(
        failure_metadata
    )
    if safe_failure_metadata:
        result["failure_metadata"] = safe_failure_metadata

    return result


def _resolve_with_cache(
    *,
    episode_key: str,
    resolver_batch: dict,
    input_fingerprint: str,
    source_fingerprint: str,
    resolver_record_path: str,
    preset_provenance: PresetProvenance,
) -> tuple[
    dict[str, list[dict]],
    dict | None,
    bool,
]:
    """Reuse persisted decisions when the exact review input is unchanged."""

    record = _load_review_record(
        resolver_record_path
    )

    if _review_record_matches(
        record,
        input_fingerprint,
        source_fingerprint,
        preset_provenance,
    ):
        print(
            "Reusing persisted transcript reviewer decisions"
        )

        return (
            _resolution_from_record(
                record
            ),
            record,
            True,
        )

    if not preset_provenance.verified:
        print(
            "Transcript resolver provenance is "
            f"{preset_provenance.status} ({preset_provenance.reason or 'unknown'}); "
            "routing AI-eligible items to Human Review"
        )
        review = [
            {
                "id": item["id"],
                "reason": f"preset_provenance_{preset_provenance.status}",
            }
            for item in resolver_batch.get("batch", {}).get("diff_items", [])
            if isinstance(item, dict) and isinstance(item.get("id"), int)
        ]
        return (
            {
                "accepted": [],
                "review": review,
                "outcomes": [
                    {
                        "id": item["id"],
                        "status": "provenance_unverified",
                        "reason": preset_provenance.reason or preset_provenance.status,
                    }
                    for item in review
                ],
            },
            record,
            False,
        )

    print("No matching transcript reviewer record; running AI reviewer")

    return (
        resolve_compiler_batch(
            resolver_batch,
            preset_provenance,
            episode_key=episode_key,
            source_fingerprint=source_fingerprint,
        ),
        record,
        False,
    )


def _route_review_anomalies(
    result,
    resolver_batch: dict,
) -> dict:
    """Keep deterministic anomalies out of AI source-choice inference."""

    differences = {
        difference.id: difference
        for difference in result.differences
    }

    routed = dict(resolver_batch)
    batch = dict(
        resolver_batch.get(
            "batch",
            {},
        )
    )
    diff_items = list(
        batch.get(
            "diff_items",
            [],
        )
    )
    bypassed_items = [
        dict(item)
        for item in resolver_batch.get(
            "bypassed_items",
            [],
        )
        if isinstance(item, dict)
    ]

    # Anomaly classification must also cover review-required differences that
    # were already excluded from resolver inference by structural eligibility.
    # Preserve the original bypass reason; anomaly metadata is an independent
    # Human Review signal and never transcript authority.
    for item in bypassed_items:
        difference_id = item.get("id")
        difference = (
            differences.get(difference_id)
            if isinstance(difference_id, int)
            else None
        )
        anomaly = (
            classify_review_anomaly(difference)
            if difference is not None
            else None
        )
        if anomaly is not None:
            item["anomaly"] = dict(anomaly)

    kept_items: list[dict] = []
    for raw_item in diff_items:
        if not isinstance(raw_item, dict):
            continue

        item = dict(raw_item)
        difference_id = item.get("id")
        difference = (
            differences.get(difference_id)
            if isinstance(difference_id, int)
            else None
        )
        anomaly = (
            classify_review_anomaly(difference)
            if difference is not None
            else None
        )

        if anomaly is None:
            kept_items.append(item)
            continue

        bypassed_items.append(
            {
                "id": difference_id,
                "reason": (
                    "review_anomaly:"
                    f"{anomaly['kind']}"
                ),
                "anomaly": dict(anomaly),
            }
        )

    batch["diff_items"] = kept_items
    routed["batch"] = batch
    routed["bypassed_items"] = bypassed_items
    return routed


def build_review_merge_preview(
    difference,
    *,
    base_source: str,
    base_text: str,
    radius: int = 30,
) -> dict:
    """Build bounded, exact merge geometry from the canonical primary source."""

    if base_source not in {"apple", "whisper"}:
        raise ValueError("review preview base source must be Apple or Whisper")
    if not isinstance(base_text, str):
        raise ValueError("review preview base text must be text")

    tokens = tokenize(base_text)
    start_word = getattr(
        difference,
        f"{base_source}_start_word",
        None,
    )
    end_word = getattr(
        difference,
        f"{base_source}_end_word",
        None,
    )

    if not isinstance(start_word, int) or not isinstance(end_word, int):
        raise ValueError("review preview word geometry is unavailable")

    start_index = start_word - 1
    end_index = end_word

    if (
        start_index < 0
        or end_index < start_index
        or start_index > len(tokens)
        or end_index > len(tokens)
    ):
        raise ValueError("review preview word geometry is invalid")

    if start_index < end_index:
        span_start = tokens[start_index].start
        span_end = tokens[end_index - 1].end
    elif start_index < len(tokens):
        span_start = tokens[start_index].start
        span_end = span_start
    else:
        span_start = len(base_text)
        span_end = span_start

    span_text = base_text[span_start:span_end]
    expected = getattr(
        difference,
        f"{base_source}_text",
        None,
    )

    if not isinstance(expected, str) or span_text != expected:
        raise ValueError(
            "review preview geometry does not match the canonical source span"
        )

    context_start_index = max(
        0,
        start_index - radius,
    )
    context_end_index = min(
        len(tokens),
        max(end_index, start_index + 1) + radius,
    )

    context_start = (
        0
        if context_start_index == 0
        else tokens[context_start_index].start
    )
    context_end = (
        len(base_text)
        if context_end_index == len(tokens)
        else tokens[context_end_index - 1].end
    )

    context_text = base_text[
        context_start:context_end
    ]

    return {
        "base_source": base_source,
        "context_text": context_text,
        "span_start": span_start - context_start,
        "span_end": span_end - context_start,
        "span_text": span_text,
    }


def _difference_review_record(
    difference,
    reason: str,
    *,
    resolver_item: dict | None = None,
    alignment=None,
) -> dict:
    """Return one compact but useful human-review record."""

    item = {
        "id": difference.id,
        "reason": reason,
        "kind": difference.kind,
        "severity": difference.severity,
        "category": (
            difference.resolver_category
        ),
        "apple_text": (
            difference.apple_text
        ),
        "whisper_text": (
            difference.whisper_text
        ),
        "apple_context": (
            difference.apple_context
        ),
        "whisper_context": (
            difference.whisper_context
        ),
        "apple_timestamp": (
            difference.apple_start_timestamp
        ),
        "whisper_start_timestamp": (
            difference.whisper_start_timestamp
        ),
        "whisper_end_timestamp": (
            difference.whisper_end_timestamp
        ),
        "suggestion": {
            "source": getattr(difference, "selected_source", None) or None,
            "reason": getattr(difference, "selection_reason", None) or None,
            "automatic_resolution": False,
        },
        "source_only": getattr(difference, "source_only", None),
        "source_only_source": getattr(difference, "source_only_source", None),
        "risk_reasons": getattr(difference, "risk_reasons", None),
        "domain_terms": getattr(difference, "domain_terms", None),
        "citation_signal": getattr(difference, "citation_signal", None),
        "preservation_class": getattr(difference, "preservation_class", None),
        "merge_action": getattr(difference, "merge_action", None),
        "representation_modified": getattr(
            difference,
            "representation_modified",
            False,
        ),
        "generation_stale": getattr(
            difference,
            "generation_stale",
            False,
        ),
        "custom_edit": getattr(difference, "custom_edit", None),
        "third_asr": getattr(difference, "third_asr", None),
    }

    if isinstance(resolver_item, dict):
        item["focus"] = {
            "scope": resolver_item.get("focus_scope", "full"),
            "apple_text": resolver_item.get("focus_apple_text"),
            "whisper_text": resolver_item.get("focus_whisper_text"),
        }

    if alignment is not None:
        item["alignment"] = alignment.locate_apple_span(
            difference.apple_start_word - 1,
            difference.apple_end_word,
        )

    return item


def _human_review_items(
    result,
    resolver_batch: dict,
    resolution: dict,
    *,
    alignment=None,
    base_texts: dict[str, str] | None = None,
    triage: dict | None = None,
) -> list[dict]:
    """Return only differences that still need a person."""

    differences = {
        difference.id: difference
        for difference
        in result.differences
    }

    accepted_ids = {
        item.get("id")
        for item in resolution.get(
            "accepted",
            [],
        )
        if (
            isinstance(
                item,
                dict,
            )
            and isinstance(
                item.get("id"),
                int,
            )
        )
    }

    resolver_reasons = {
        item["id"]: item.get(
            "reason",
            "resolver_requires_human_review",
        )
        for item in resolution.get(
            "review",
            [],
        )
        if (
            isinstance(
                item,
                dict,
            )
            and isinstance(
                item.get("id"),
                int,
            )
        )
    }

    batch_ids = {
        item.get("id")
        for item in resolver_batch.get(
            "batch",
            {},
        ).get(
            "diff_items",
            [],
        )
        if (
            isinstance(
                item,
                dict,
            )
            and isinstance(
                item.get("id"),
                int,
            )
        )
    }

    bypass_reasons = {
        item["id"]: item.get(
            "reason",
            "resolver_structurally_ineligible",
        )
        for item in resolver_batch.get(
            "bypassed_items",
            [],
        )
        if (
            isinstance(item, dict)
            and isinstance(item.get("id"), int)
        )
    }
    bypass_anomalies = {
        item["id"]: dict(item["anomaly"])
        for item in resolver_batch.get(
            "bypassed_items",
            [],
        )
        if (
            isinstance(item, dict)
            and isinstance(item.get("id"), int)
            and isinstance(item.get("anomaly"), dict)
        )
    }

    deferred_reasons = {
        item["id"]: item.get(
            "reason",
            "resolver_batch_limit",
        )
        for item in resolver_batch.get(
            "deferred_items",
            [],
        )
        if (
            isinstance(item, dict)
            and isinstance(item.get("id"), int)
        )
    }

    reasons: dict[
        int,
        str,
    ] = {}

    # Every high-value item sent to the AI must either be accepted or surfaced.
    for difference_id in batch_ids:

        if difference_id in accepted_ids:
            continue

        reasons[
            difference_id
        ] = resolver_reasons.get(
            difference_id,
            "resolver_requires_human_review",
        )

    # Compiler-only review items, such as ambiguous repetitions or a
    # source-only Whisper span with suspicious acoustic confidence,
    # intentionally never enter the focused AI batch but still need a person.
    for difference in result.differences:

        if (
            getattr(
                difference,
                "review_required",
                difference.merge_action == "review_kept_primary",
            )
            and difference.id not in batch_ids
        ):
            reasons.setdefault(
                difference.id,
                bypass_reasons.get(
                    difference.id,
                    deferred_reasons.get(
                        difference.id,
                        "compiler_requires_human_review",
                    ),
                ),
            )

    resolver_items = {
        item.get("id"): item
        for item in resolver_batch.get("batch", {}).get("diff_items", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    triage_outcomes = {
        item["id"]: item
        for item in (triage or {}).get("outcomes", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    triage_recommendations = {
        item["id"]: item
        for item in (triage or {}).get("recommendations", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }

    review_items: list[dict] = []
    base_source = getattr(
        result,
        "recommended_source",
        None,
    )
    base_text = (
        base_texts.get(base_source)
        if isinstance(base_texts, dict)
        and base_source in {"apple", "whisper"}
        else None
    )

    for difference_id in sorted(reasons):
        if difference_id not in differences:
            continue

        difference = differences[difference_id]
        item = _difference_review_record(
            difference,
            reasons[difference_id],
            resolver_item=resolver_items.get(difference_id),
            alignment=alignment,
        )

        if difference_id in bypass_anomalies:
            item["anomaly"] = dict(
                bypass_anomalies[difference_id]
            )

        triage_outcome = triage_outcomes.get(difference_id)
        if isinstance(triage_outcome, dict):
            projection = {
                field: triage_outcome[field]
                for field in (
                    "status",
                    "recommendation",
                    "confidence",
                    "reason",
                )
                if field in triage_outcome
            }
            triage_recommendation = triage_recommendations.get(difference_id)
            if isinstance(triage_recommendation, dict):
                for field in ("source", "text"):
                    if field in triage_recommendation:
                        projection[field] = triage_recommendation[field]
            item["triage"] = projection

        recommendation = batch_recommendation(item)
        if recommendation is not None:
            item["batch_recommendation"] = recommendation

        if isinstance(base_text, str):
            item["merge_preview"] = build_review_merge_preview(
                difference,
                base_source=base_source,
                base_text=base_text,
            )

        review_items.append(item)

    return review_items


def _build_review_record(
    *,
    episode_key: str,
    input_fingerprint: str,
    source_fingerprint: str,
    resolution: dict,
    human_review: list[dict],
    resolver_batch: dict,
    source_paths: dict,
    preset_provenance: PresetProvenance,
    triage: dict | None = None,
    triage_batch: dict | None = None,
) -> dict:
    """Create one durable reviewer audit record."""

    record = {
        "schema_version": (
            REVIEW_RECORD_SCHEMA_VERSION
        ),
        "policy_version": (
            REVIEW_POLICY_VERSION
        ),
        "episode_key": (
            episode_key
        ),
        "input_fingerprint": (
            input_fingerprint
        ),
        "source_fingerprint": source_fingerprint,
        "reviewer_preset": (
            REVIEW_PRESET
        ),
        "preset_provenance": preset_provenance.record(),
        "reviewed_at": (
            now_iso()
        ),
        "inputs": {
            "apple": {
                "text": (
                    source_paths[
                        "apple"
                    ].get(
                        "text"
                    )
                ),
                "metadata": (
                    source_paths[
                        "apple"
                    ].get(
                        "metadata"
                    )
                ),
            },
            "whisper": {
                "text": (
                    source_paths[
                        "whisper"
                    ].get(
                        "text"
                    )
                ),
                "metadata": (
                    source_paths[
                        "whisper"
                    ].get(
                        "metadata"
                    )
                ),
            },
        },
        "resolver_item_ids": [
            item.get(
                "id"
            )
            for item
            in resolver_batch.get(
                "batch",
                {},
            ).get(
                "diff_items",
                [],
            )
            if isinstance(
                item,
                dict,
            )
        ],
        "resolver_bypassed": [
            dict(item)
            for item in resolver_batch.get(
                "bypassed_items",
                [],
            )
            if isinstance(item, dict)
        ],
        "resolver_deferred": [
            dict(item)
            for item in resolver_batch.get(
                "deferred_items",
                [],
            )
            if isinstance(item, dict)
        ],
        "triage_item_ids": [
            item.get("id")
            for item in (triage_batch or {}).get("batch", {}).get("diff_items", [])
            if isinstance(item, dict) and isinstance(item.get("id"), int)
        ],
        "triage_recommendations": [
            dict(item)
            for item in (triage or {}).get("recommendations", [])
            if isinstance(item, dict)
        ],
        "triage_outcomes": [
            dict(item)
            for item in (triage or {}).get("outcomes", [])
            if isinstance(item, dict)
        ],
        "triage_completion_metadata": (
            dict((triage or {}).get("completion_metadata", {}))
            if isinstance((triage or {}).get("completion_metadata", {}), dict)
            else {}
        ),
        "triage_failure_metadata": _safe_triage_failure_metadata(
            (triage or {}).get("failure_metadata")
        ),
        "resolver_outcomes": [
            dict(item)
            for item in resolution.get(
                "outcomes",
                [],
            )
            if isinstance(item, dict)
        ],
        "accepted": [
            dict(item)
            for item
            in resolution.get(
                "accepted",
                [],
            )
            if isinstance(
                item,
                dict,
            )
        ],
        "resolver_review": [
            dict(item)
            for item
            in resolution.get(
                "review",
                [],
            )
            if isinstance(
                item,
                dict,
            )
        ],
        "human_decisions": [],
        "human_review": [
            dict(item)
            for item
            in human_review
            if isinstance(
                item,
                dict,
            )
        ],
        "human_review_queue_fingerprint": review_queue_fingerprint(
            human_review
        ),
        "human_review_generation_fingerprint": review_generation_fingerprint(
            input_fingerprint, human_review
        ),
    }
    completion_metadata = resolution.get("completion_metadata")
    if isinstance(completion_metadata, dict):
        for field in ("completion_id", "served_model", "served_provider"):
            value = completion_metadata.get(field)
            if isinstance(value, str) and value:
                record["preset_provenance"][field] = value
    return record


def _save_review_record(
    gcs_path: str,
    record: dict,
) -> str:
    get_bucket().blob(
        gcs_path
    ).upload_from_string(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        content_type=(
            "application/json"
        ),
    )

    print(
        f"Saved resolver audit record: {gcs_path}"
    )

    return gcs_path


def _write_cloud_review_state(
    report_path: Path,
    *,
    result,
    review: list[dict],
    reviewer: dict | None = None,
) -> None:
    """Write human-review semantics into the canonical JSON report."""

    payload = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    # Broad high/medium diagnostic count, kept distinct from the actual queue.
    payload[
        "risk_candidates"
    ] = (
        result.high_risk + result.medium_risk
        if hasattr(result, "high_risk") and hasattr(result, "medium_risk")
        else result.review_required
    )

    # Actual number that should gate the pipeline.
    payload[
        "review_required"
    ] = len(
        review
    )

    payload[
        "review"
    ] = review

    if reviewer is not None:

        payload[
            "reviewer"
        ] = reviewer

    report_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def compile_episode_sources(
    episode: dict,
) -> dict:
    """Compile canonical sources and reuse stable AI review decisions."""

    episode_key = episode[
        "episode_key"
    ]

    source_paths = episode[
        "files"
    ][
        "sources"
    ]

    apple = source_paths[
        "apple"
    ]

    whisper = source_paths[
        "whisper"
    ]

    if (
        not apple.get(
            "text"
        )
        or not whisper.get(
            "text"
        )
    ):
        raise ValueError(
            "Apple and Whisper transcript sources are both required"
        )

    work_dir = (
        RUNTIME_DIR
        / episode_key
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Always refresh source bytes. Otherwise a warm container could calculate
    # the new fingerprint from an old local file.
    apple_text_path = (
        _local_copy(
            apple[
                "text"
            ],
            work_dir
            / "apple-transcript.txt",
            refresh=True,
        )
    )

    whisper_text_path = (
        _local_copy(
            whisper[
                "text"
            ],
            work_dir
            / "whisper-transcript.txt",
            refresh=True,
        )
    )

    apple_metadata_path = (
        _optional_local_copy(
            apple.get(
                "metadata"
            ),
            work_dir
            / "apple-transcript.json",
            refresh=True,
        )
    )

    whisper_metadata_path = (
        _optional_local_copy(
            whisper.get(
                "metadata"
            ),
            work_dir
            / "whisper-transcript.json",
            refresh=True,
        )
    )

    compiler_kwargs = {
        "apple_metadata": (
            apple_metadata_path
        ),
        "whisper_metadata": (
            whisper_metadata_path
        ),
    }

    apple_text = load_text(
        apple_text_path
    )

    whisper_text = load_text(
        whisper_text_path
    )

    review_base_texts = {
        "apple": clean_transcript(
            apple_text,
            remove_timestamps=True,
        ),
        "whisper": clean_transcript(
            whisper_text,
        ),
    }

    result = compile_transcripts(
        apple_text,
        whisper_text,
        **compiler_kwargs,
    )

    resolver_batch = (
        build_resolver_batch(
            result,
            episode_key,
        )
    )
    resolver_batch = _route_review_anomalies(
        result,
        resolver_batch,
    )

    # Alignment is evidence for locating audio in the local UI. It is never
    # passed back into the compiler as a resolution or a source preference.
    alignment = align_transcripts(
        apple_text,
        whisper_text,
    )

    canonical = paths_for(
        episode_key
    )

    existing_record = _load_review_record(canonical["resolver_record"])
    recompile_correlation = _recompile_correlation_for_episode(episode_key)

    def link_recompile_result(review_record: dict) -> bool:
        if recompile_correlation is None:
            return False
        prior_generation = (existing_record or {}).get(
            "human_review_generation_fingerprint"
        )
        if prior_generation != recompile_correlation["review_generation"]:
            raise RecompileLifecycleInvariantError(
                "Recompile request does not match the consumed review generation"
            )
        result_generation = review_record.get("human_review_generation_fingerprint")
        if not isinstance(result_generation, str) or not result_generation:
            raise RecompileLifecycleInvariantError(
                "Rebuilt resolver has no durable review generation"
            )
        requests = review_record.get("recompile_requests", [])
        if not isinstance(requests, list):
            raise RecompileLifecycleInvariantError(
                "Rebuilt resolver has no durable recompile requests"
            )
        return link_recompile_result_generation(
            requests,
            request_id=recompile_correlation["request_id"],
            review_generation=recompile_correlation["review_generation"],
            result_generation=result_generation,
        )

    provisional_review = _human_review_items(
        result,
        resolver_batch,
        {"accepted": [], "review": [], "outcomes": []},
        alignment=alignment,
        base_texts=review_base_texts,
    )
    existing_human_resolutions = validated_human_resolutions(
        existing_record or {},
        provisional_review,
        require_current_item_evidence=True,
    )
    human_decided_ids = {
        item.get("id")
        for item in existing_human_resolutions
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    anomaly_ids = {
        item["id"]
        for item in resolver_batch.get("bypassed_items", [])
        if (
            isinstance(item, dict)
            and isinstance(item.get("id"), int)
            and isinstance(item.get("anomaly"), dict)
        )
    }
    triage_batch = build_triage_batch(
        result,
        exclude_ids=anomaly_ids | human_decided_ids,
    )
    resolver_item_ids = {
        item.get("id")
        for item in resolver_batch.get("batch", {}).get("diff_items", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    triage_item_ids = {
        item.get("id")
        for item in triage_batch.get("batch", {}).get("diff_items", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }

    if (resolver_item_ids - human_decided_ids) or triage_item_ids:
        # Provenance is an AI dependency.  Keep deterministic compilation and
        # existing valid Human Review truth independent of the Presets API
        # when no AI inference or AI-cache reuse remains necessary.
        preset_provenance = verify_transcript_reviewer(
            local_policy_version=REVIEW_POLICY_VERSION,
            preset_slug=REVIEW_PRESET,
            verified_at=now_iso(),
        )
    else:
        preset_provenance = PresetProvenance(
            status="not_required",
            reason="no_ai_resolver_work",
            slug=REVIEW_PRESET,
        )

    input_fingerprint = (
        _review_input_fingerprint(
            apple_text_path=(
                apple_text_path
            ),
            apple_metadata_path=(
                apple_metadata_path
            ),
            whisper_text_path=(
                whisper_text_path
            ),
            whisper_metadata_path=(
                whisper_metadata_path
            ),
            resolver_batch=(
                resolver_batch
            ),
            preset_provenance=preset_provenance,
            triage_batch=(
                triage_batch
            ),
        )
    )

    source_fingerprint = _review_source_fingerprint_from_paths(
        apple_text_path=apple_text_path,
        apple_metadata_path=apple_metadata_path,
        whisper_text_path=whisper_text_path,
        whisper_metadata_path=whisper_metadata_path,
    )

    (
        resolution,
        cached_record,
        cache_hit,
    ) = _resolve_with_cache(
        episode_key=episode_key,
        resolver_batch=(
            resolver_batch
        ),
        input_fingerprint=(
            input_fingerprint
        ),
        source_fingerprint=source_fingerprint,
        resolver_record_path=(
            canonical[
                "resolver_record"
            ]
        ),
        preset_provenance=preset_provenance,
    )

    if cache_hit:
        triage = _triage_from_record(cached_record or {})
    else:
        try:
            triage = triage_compiler_batch(
                triage_batch,
                preset_provenance,
                episode_key=episode_key,
                source_fingerprint=source_fingerprint,
            )
        except Exception as error:
            print(
                "Advisory transcript triage unavailable; "
                f"preserving Human Review ({type(error).__name__})"
            )
            triage = _unavailable_triage(
                triage_batch,
                failure_metadata=_triage_error_metadata(error),
            )

        _emit_triage_degradation(episode_key, triage)

    if resolution["accepted"]:

        result = (
            compile_transcripts(
                apple_text,
                whisper_text,
                resolver_resolutions=(
                    resolution[
                        "accepted"
                    ]
                ),
                **compiler_kwargs,
            )
        )

    review_before_human = (
        _human_review_items(
            result,
            resolver_batch,
            resolution,
            alignment=alignment,
            base_texts=review_base_texts,
            triage=triage,
        )
    )

    human_resolutions = validated_human_resolutions(
        cached_record or {},
        review_before_human,
        require_current_item_evidence=not cache_hit,
    )

    if human_resolutions:
        result = compile_transcripts(
            apple_text,
            whisper_text,
            resolver_resolutions=(
                resolution["accepted"] + human_resolutions
            ),
            **compiler_kwargs,
        )

    review = _human_review_items(
        result,
        resolver_batch,
        resolution,
        alignment=alignment,
        base_texts=review_base_texts,
        triage=triage,
    )

    decided_ids = {
        item.get("id")
        for item in human_resolutions
        if isinstance(item, dict)
    }
    preserved_human_decisions = [
        dict(decision)
        for decision in (cached_record or {}).get("human_decisions", [])
        if (
            isinstance(decision, dict)
            and decision.get("reviewed_by") == "human"
            and decision.get("id") in decided_ids
        )
    ]
    superseded_human_decisions = [
        dict(decision)
        for decision in (cached_record or {}).get("human_decisions", [])
        if (
            isinstance(decision, dict)
            and decision.get("reviewed_by") == "human"
            and decision.get("id") not in decided_ids
        )
    ]
    preserved_recompile_requests = [
        dict(request)
        for request in (cached_record or {}).get("recompile_requests", [])
        if isinstance(request, dict)
    ]
    review = [
        item for item in review
        if item.get("id") not in decided_ids
    ]

    if cache_hit:
        review_record = cached_record or {}
        existing_evidence = {
            item.get("id"): item.get("third_asr")
            for item in review_record.get("human_review", [])
            if isinstance(item, dict)
            and isinstance(item.get("third_asr"), dict)
        }
        for item in review:
            if item.get("id") in existing_evidence:
                item["third_asr"] = existing_evidence[item["id"]]
        queue_fingerprint = review_queue_fingerprint(review)
        changed_review = (
            review_record.get("human_review") != review
            or review_record.get("human_review_queue_fingerprint")
            != queue_fingerprint
        )
        review_record.setdefault("human_decisions", [])
        review_record["human_review"] = review
        review_record["human_review_queue_fingerprint"] = queue_fingerprint
        linked_recompile_result = link_recompile_result(review_record)
        if changed_review:
            review_record["human_review_updated_at"] = now_iso()
        if changed_review or linked_recompile_result:
            _save_review_record(canonical["resolver_record"], review_record)

    else:

        review_record = (
            _build_review_record(
                episode_key=(
                    episode_key
                ),
                input_fingerprint=(
                    input_fingerprint
                ),
                source_fingerprint=source_fingerprint,
                resolution=(
                    resolution
                ),
                human_review=(
                    review
                ),
                resolver_batch=(
                    resolver_batch
                ),
                source_paths=(
                    source_paths
                ),
                preset_provenance=preset_provenance,
                triage=triage,
                triage_batch=(
                    triage_batch
                ),
            )
        )
        review_record["human_decisions"] = preserved_human_decisions
        review_record["human_review_generation_fingerprint"] = (
            review_generation_fingerprint(
                input_fingerprint,
                list(review)
                + [
                    decision["review_item"]
                    for decision in preserved_human_decisions
                    if isinstance(decision.get("review_item"), dict)
                ],
            )
        )
        if preserved_recompile_requests:
            review_record["recompile_requests"] = preserved_recompile_requests
        link_recompile_result(review_record)
        if superseded_human_decisions:
            review_record["superseded_human_decisions"] = superseded_human_decisions

        _save_review_record(
            canonical[
                "resolver_record"
            ],
            review_record,
        )

    # Phase A terminology retrieval runs only after every canonical compiler,
    # resolver, and Human Review decision is complete. Its output is report-only
    # shadow telemetry and cannot affect the already-finalized decision path.
    terminology_retrieval = attach_terminology_retrieval_shadow_best_effort(result)
    if terminology_retrieval.get("retrieved_difference_count"):
        print(
            "Terminology retrieval shadow: "
            f"{terminology_retrieval['retrieved_difference_count']} diff(s), "
            f"{terminology_retrieval['candidate_hit_count']} candidate hit(s)"
        )

    outputs = write_outputs(
        result,
        output_dir=work_dir,
        apple_path=(
            apple[
                "text"
            ]
        ),
        whisper_path=(
            whisper[
                "text"
            ]
        ),
    )

    reviewer_report = {
        "preset": (
            REVIEW_PRESET
        ),
        "policy_version": (
            REVIEW_POLICY_VERSION
        ),
        "preset_provenance": review_record.get("preset_provenance"),
        "input_fingerprint": (
            input_fingerprint
        ),
        "resolver_record": (
            canonical[
                "resolver_record"
            ]
        ),
        "cache_hit": (
            cache_hit
        ),
        "resolver_eligible": len(
            resolver_batch.get(
                "batch",
                {},
            ).get(
                "diff_items",
                [],
            )
        ),
        "resolver_bypassed_structurally_ineligible": len(
            resolver_batch.get(
                "bypassed_items",
                [],
            )
        ),
        "resolver_deferred_by_batch_limit": len(
            resolver_batch.get(
                "deferred_items",
                [],
            )
        ),
        "reviewed_at": (
            review_record.get(
                "reviewed_at"
            )
        ),
        "accepted": len(
            resolution.get(
                "accepted",
                [],
            )
        ),
        "resolver_review": len(
            resolution.get(
                "review",
                [],
            )
        ),
        "resolver_abstained": sum(
            item.get("status") == "abstained"
            for item in resolution.get("outcomes", [])
            if isinstance(item, dict)
        ),
        "resolver_rejected_by_python": sum(
            item.get("status") == "rejected_by_python"
            for item in resolution.get("outcomes", [])
            if isinstance(item, dict)
        ),
        "human_decisions": len(
            review_record.get("human_decisions", [])
        ),
    }

    _write_cloud_review_state(
        outputs[
            "json"
        ],
        result=result,
        review=review,
        reviewer=(
            reviewer_report
        ),
    )

    transcript_path = (
        upload_path_to_gcs(
            outputs[
                "transcript"
            ],
            canonical[
                "compiled_text"
            ],
        )
    )

    report_path = (
        upload_path_to_gcs(
            outputs[
                "json"
            ],
            canonical[
                "compiler_report"
            ],
        )
    )

    # Candidate terminology is deliberately downstream, non-authoritative
    # curation telemetry. A failure here must never invalidate a successful
    # canonical compile or alter the uploaded compiler report.
    candidate_payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    candidate_collection = record_compiler_candidates_best_effort(
        episode_key=episode_key,
        report=candidate_payload,
        review_record=review_record,
    )
    if any(candidate_collection.values()):
        print(f"Terminology candidate collection: {candidate_collection}")

    return {
        "transcript": (
            transcript_path
        ),
        "report": (
            report_path
        ),
        "resolver_record": (
            canonical[
                "resolver_record"
            ]
        ),
        "review_cache_hit": (
            cache_hit
        ),
        "review_required": len(
            review
        ),
        "review": review,
    }
