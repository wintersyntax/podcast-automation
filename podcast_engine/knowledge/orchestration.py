"""Build and persist versioned podcast knowledge notes through storage-backed orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..episode_contract import now_iso, paths_for, summary_review_artifact_paths
from ..episode_generation import source_generation_fingerprint_from_episode
from ..notifications import send_email_summary_review_stalled_notification
from ..rss import fetch_episode_description
from ..storage import download_file_from_gcs, get_bucket
from . import metadata, summary, summary_review
from .frontmatter import render
from .notes_v3 import render as note_render
from .notes_v3 import writer as note_writer
from .notes_v3.units import UnitReport
from .models import (
    KNOWLEDGE_SCHEMA_VERSION,
    METADATA_POLICY_VERSION,
    NOTE_WRITER_POLICY_VERSION,
    SUMMARY_POLICY_VERSION,
    SUMMARY_REVIEW_EVIDENCE_SCHEMA_VERSION,
    SUMMARY_REVIEW_POLICY_VERSION,
)
from .summary_review_contract import (
    accepted_final_markdown,
    validate_audit_result,
    validate_edit_result,
)
from .summary_review_evidence import build_review_context, validate_review_context
from .summary_review_storage import (
    load_summary_review_stalled_diagnostics,
    persist_review_attempt,
)
from .tags import TagRegistry, semantic_projection


RUNTIME_DIR = Path(__file__).resolve().parents[2] / "var" / "knowledge"


def _load_manifest(gcs_path: str) -> dict | None:
    blob = get_bucket().blob(gcs_path)
    if not blob.exists():
        return None
    try:
        payload = json.loads(blob.download_as_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def canonical_summary_generated_at(episode_key: str) -> str | None:
    """Return the canonical summary-generation instant from its GCS manifest.

    This deliberately returns no value instead of substituting a worker or
    delivery time. Cached summaries therefore retain their original instant.
    """

    manifest = _load_manifest(paths_for(episode_key)["summary_metadata"])
    summary_record = manifest.get("summary") if isinstance(manifest, dict) else None
    generated_at = (
        summary_record.get("generated_at")
        if isinstance(summary_record, dict)
        else None
    )
    return generated_at if isinstance(generated_at, str) and generated_at else None


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _notify_if_summary_review_stalled(
    *,
    bucket,
    episode: dict,
    transcript: str,
    draft: str,
    review_preset: str,
) -> None:
    episode_key = episode["episode_key"]
    try:
        diagnostics = load_summary_review_stalled_diagnostics(
            bucket=bucket,
            episode_key=episode_key,
        )
        if diagnostics.get("stalled") is not True:
            return

        podcastops_base_url = os.environ.get("PODCAST_REVIEW_URL", "").strip()
        if not podcastops_base_url:
            return
        podcastops_url = (
            podcastops_base_url.rstrip("/")
            + f"/episodes/{episode_key}/summary-review"
        )

        send_email_summary_review_stalled_notification(
            bucket=bucket,
            episode=episode,
            review_input_identity={
                "episode_key": episode_key,
                "transcript_sha256": _sha256_bytes(transcript.encode("utf-8")),
                "draft_sha256": _sha256_bytes(draft.encode("utf-8")),
                "review_policy_version": SUMMARY_REVIEW_POLICY_VERSION,
                "review_preset": review_preset,
            },
            diagnostics=diagnostics,
            podcastops_url=podcastops_url,
        )
    except Exception as error:
        warning = {
            "severity": "WARNING",
            "message": "Summary-review stalled notification failed",
            "event": "summary_review_stalled_notification_failed",
            "episode_key": episode_key,
            "error_type": type(error).__name__,
        }
        print(json.dumps(warning, ensure_ascii=False, separators=(",", ":")))


def _summary_matches(
    manifest: dict | None,
    fingerprint: str,
    preset: str,
    reviewer_preset: str | None = None,
    body_bytes: bytes | None = None,
) -> bool:
    record = manifest.get("summary") if isinstance(manifest, dict) else None
    review = manifest.get("summary_review") if isinstance(manifest, dict) else None
    return (
        isinstance(record, dict)
        and isinstance(review, dict)
        and manifest.get("schema_version") == KNOWLEDGE_SCHEMA_VERSION
        and record.get("policy_version") == SUMMARY_POLICY_VERSION
        and record.get("preset") == preset
        and record.get("input_fingerprint") == fingerprint
        and isinstance(record.get("generated_at"), str)
        and review.get("policy_version") == SUMMARY_REVIEW_POLICY_VERSION
        and review.get("preset") == reviewer_preset
        and review.get("status") in {"pass", "revised"}
        and isinstance(review.get("review_id"), str)
        and bool(review.get("review_id"))
        and isinstance(review.get("artifact_root"), str)
        and bool(review.get("artifact_root"))
        and isinstance(body_bytes, bytes)
        and review.get("final_sha256") == _sha256_bytes(body_bytes)
    )


def _metadata_matches(manifest: dict, fingerprint: str, preset: str) -> bool:
    record = manifest.get("metadata") if isinstance(manifest, dict) else None
    return (
        isinstance(record, dict)
        and record.get("policy_version") == METADATA_POLICY_VERSION
        and record.get("preset") == preset
        and record.get("input_fingerprint") == fingerprint
        and isinstance(record.get("generated_at"), str)
        and all(
            isinstance(record.get(field), list)
            for field in ("topics", "people", "tags", "tag_candidates")
        )
    )


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _upload_text_if_changed(path: Path, gcs_path: str, content_type: str) -> bool:
    """Upload text only when the canonical object has different bytes."""

    content = path.read_bytes()
    blob = get_bucket().blob(gcs_path)
    if blob.exists() and blob.download_as_bytes() == content:
        return False
    blob.upload_from_filename(str(path), content_type=content_type)
    return True


def _download_canonical_compiled_transcript(episode: dict, destination: Path) -> Path:
    """Use the canonical compiled object, never the legacy Whisper transcript."""

    gcs_path = paths_for(episode["episode_key"])["compiled_text"]
    return Path(download_file_from_gcs(gcs_path, destination))


def _load_json_blob(blob) -> dict | None:
    if not blob.exists():
        return None
    try:
        value = json.loads(blob.download_as_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _accepted_review_artifacts_match(
    bucket,
    episode_key: str,
    manifest: dict,
    body_bytes: bytes,
    transcript_bytes: bytes,
) -> bool:
    """Prove an accepted manifest pointer resolves to one exact V2 review chain."""

    review = manifest.get("summary_review") if isinstance(manifest, dict) else None
    summary_record = manifest.get("summary") if isinstance(manifest, dict) else None
    if not isinstance(review, dict) or not isinstance(summary_record, dict):
        return False

    review_id = review.get("review_id")
    if not isinstance(review_id, str) or not review_id:
        return False
    paths = summary_review_artifact_paths(episode_key, review_id)
    if review.get("artifact_root") != paths["root"]:
        return False

    required_keys = (
        "draft",
        "final",
        "review",
        "transcript_index",
        "draft_index",
        "risk_inventory",
    )
    blobs = {key: bucket.blob(paths[key]) for key in required_keys}
    if any(not blob.exists() for blob in blobs.values()):
        return False

    try:
        draft_bytes = blobs["draft"].download_as_bytes()
        final_bytes = blobs["final"].download_as_bytes()
        draft = draft_bytes.decode("utf-8")
        transcript = transcript_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if final_bytes != body_bytes:
        return False

    envelope = _load_json_blob(blobs["review"])
    transcript_index = _load_json_blob(blobs["transcript_index"])
    draft_index = _load_json_blob(blobs["draft_index"])
    risk_inventory = _load_json_blob(blobs["risk_inventory"])
    if any(
        value is None
        for value in (envelope, transcript_index, draft_index, risk_inventory)
    ):
        return False

    try:
        rebuilt_context = build_review_context(transcript, draft)
        validate_review_context(rebuilt_context, transcript, draft)
    except (TypeError, ValueError):
        return False
    if transcript_index != rebuilt_context["transcript_span_index"]:
        return False
    if draft_index != rebuilt_context["draft_block_index"]:
        return False
    if risk_inventory != rebuilt_context["risk_inventory"]:
        return False

    expected_evidence_context = {
        "transcript_span_algorithm": rebuilt_context["transcript_span_index"]["algorithm"],
        "transcript_span_index_sha256": rebuilt_context["transcript_span_index"]["index_sha256"],
        "draft_block_algorithm": rebuilt_context["draft_block_index"]["algorithm"],
        "draft_block_index_sha256": rebuilt_context["draft_block_index"]["index_sha256"],
        "risk_inventory_algorithm": rebuilt_context["risk_inventory"]["algorithm"],
        "risk_inventory_sha256": rebuilt_context["risk_inventory"]["index_sha256"],
    }

    if envelope.get("schema_version") != SUMMARY_REVIEW_EVIDENCE_SCHEMA_VERSION:
        return False
    if envelope.get("review_id") != review_id:
        return False
    if envelope.get("episode_key") != episode_key:
        return False
    if envelope.get("outcome") != "accepted":
        return False
    if envelope.get("summary_policy_version") != SUMMARY_POLICY_VERSION:
        return False
    if envelope.get("review_policy_version") != SUMMARY_REVIEW_POLICY_VERSION:
        return False
    if envelope.get("summary_preset") != summary_record.get("preset"):
        return False
    if envelope.get("review_preset") != review.get("preset"):
        return False
    if envelope.get("transcript_sha256") != _sha256_bytes(transcript_bytes):
        return False
    if envelope.get("draft_sha256") != _sha256_bytes(draft_bytes):
        return False
    if envelope.get("final_sha256") != _sha256_bytes(final_bytes):
        return False
    if review.get("final_sha256") != envelope.get("final_sha256"):
        return False
    if envelope.get("evidence_context") != expected_evidence_context:
        return False

    completion = envelope.get("completion_metadata")
    if not isinstance(completion, dict):
        return False
    attempts = completion.get("attempts")
    attempt_count = completion.get("attempt_count")
    if (
        not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or not 1 <= attempt_count <= 3
        or not isinstance(attempts, list)
        or not 1 <= len(attempts) <= attempt_count
    ):
        return False

    served_identities = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            return False
        model = attempt.get("served_model")
        provider = attempt.get("served_provider")
        if (
            not isinstance(model, str)
            or not model
            or model.casefold().startswith("google/gemini")
            or not isinstance(provider, str)
            or not provider
        ):
            return False
        served_identities.append((model, provider))
    if len(set(served_identities)) != 1:
        return False
    served_model, served_provider = served_identities[0]
    if envelope.get("review_model") != served_model:
        return False
    if envelope.get("review_provider") != served_provider:
        return False
    if completion.get("served_model") != served_model:
        return False
    if completion.get("served_provider") != served_provider:
        return False

    audit_result = envelope.get("audit_result")
    edit_result = envelope.get("edit_result")
    if not isinstance(audit_result, dict) or not isinstance(edit_result, dict):
        return False
    try:
        validated_audit = validate_audit_result(
            audit_result,
            transcript,
            rebuilt_context,
        )
        if validated_audit["status"] not in {"pass", "revised"}:
            return False
        validated_edit = validate_edit_result(
            edit_result,
            validated_audit,
            draft,
            transcript,
            rebuilt_context,
        )
        accepted = accepted_final_markdown(validated_audit, validated_edit, draft)
    except (TypeError, ValueError):
        return False
    return isinstance(accepted, str) and accepted.encode("utf-8") == final_bytes


def _note_writer_record_matches(manifest: dict | None, fingerprint: str, preset: str) -> bool:
    """The stored writer note is current for these inputs (body aside)."""

    record = manifest.get("summary") if isinstance(manifest, dict) else None
    metadata_record = manifest.get("metadata") if isinstance(manifest, dict) else None
    writer_record = manifest.get("note_writer") if isinstance(manifest, dict) else None
    return (
        isinstance(record, dict)
        and isinstance(metadata_record, dict)
        and isinstance(writer_record, dict)
        and isinstance(writer_record.get("note"), dict)
        and manifest.get("schema_version") == KNOWLEDGE_SCHEMA_VERSION
        and record.get("policy_version") == NOTE_WRITER_POLICY_VERSION
        and record.get("preset") == preset
        and record.get("input_fingerprint") == fingerprint
        and isinstance(record.get("generated_at"), str)
        and all(
            isinstance(metadata_record.get(field), list)
            for field in ("topics", "people", "tags", "tag_candidates")
        )
    )


def _render_version(manifest: dict) -> str:
    return manifest["summary"].get("render_version") or note_render.LEGACY_RENDER_VERSION


def _note_writer_matches(
    manifest: dict | None,
    fingerprint: str,
    preset: str,
    body_bytes: bytes | None,
) -> bool:
    record = manifest.get("summary") if isinstance(manifest, dict) else None
    metadata_record = manifest.get("metadata") if isinstance(manifest, dict) else None
    return (
        _note_writer_record_matches(manifest, fingerprint, preset)
        and _render_version(manifest) == note_render.RENDER_VERSION
        and isinstance(record, dict)
        and isinstance(metadata_record, dict)
        and isinstance(manifest.get("note_writer"), dict)
        and manifest.get("schema_version") == KNOWLEDGE_SCHEMA_VERSION
        and record.get("policy_version") == NOTE_WRITER_POLICY_VERSION
        and record.get("preset") == preset
        and record.get("input_fingerprint") == fingerprint
        and isinstance(record.get("generated_at"), str)
        and isinstance(body_bytes, bytes)
        and record.get("body_sha256") == _sha256_bytes(body_bytes)
        and all(
            isinstance(metadata_record.get(field), list)
            for field in ("topics", "people", "tags", "tag_candidates")
        )
    )


def _episode_spend(episode_key: str, source_fingerprint: str) -> dict:
    """Per-stage AI spend for this source generation, recorded with the note.

    Best effort: the ledger is the authority, and a failed read must never
    block publishing a note that has already been paid for.
    """

    from ..ai_budget import episode_spend_summary

    try:
        return episode_spend_summary(episode_key, source_fingerprint)
    except Exception as error:  # noqa: BLE001 -- audit-only record
        return {"unavailable": type(error).__name__}


def _build_single_pass_note(episode: dict, preset: str) -> str:
    """TASK-118: one writer call -> checks -> render -> existing contracts.

    Replaces summary + summary review + metadata. The writer returns topics,
    people and tag proposals together with the note, so the tag registry and
    frontmatter contracts are fed exactly as before. The body and the
    manifest are uploaded only after the note has passed structural
    validation and the deterministic checks. The checked structured note
    lives inside the manifest (``note_writer.note``), so no new GCS object
    -- and no change to the named-object IAM contract -- is needed.
    """

    episode_key = episode["episode_key"]
    paths = paths_for(episode_key)
    local_root = RUNTIME_DIR / episode_key
    compiled_path = _download_canonical_compiled_transcript(
        episode,
        local_root / "compiled-transcript.txt",
    )
    transcript_bytes = compiled_path.read_bytes()
    try:
        transcript = transcript_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Compiled transcript must be UTF-8 text") from error

    bucket = get_bucket()
    manifest = _load_manifest(paths["summary_metadata"])
    fingerprint = note_writer.input_fingerprint(episode, transcript_bytes, preset)
    body_path = local_root / "body.md"
    body_blob = bucket.blob(paths["summary_body"])
    cached_body_bytes = body_blob.download_as_bytes() if body_blob.exists() else None

    if _note_writer_matches(manifest, fingerprint, preset, cached_body_bytes):
        print("Knowledge note cache hit")
        try:
            body = cached_body_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Cached summary body must be UTF-8 text") from error
        _write_text(body_path, body)
        metadata_values = {
            field: list(manifest["metadata"][field])
            for field in ("topics", "people", "tags")
        }
    elif _note_writer_record_matches(manifest, fingerprint, preset):
        # Same checked note, newer renderer (or a changed body object):
        # re-render from the stored note -- no model call, no spend.
        print("Knowledge note re-render")
        unit_report = UnitReport()
        body = note_render.render_body(manifest["note_writer"]["note"], unit_report)
        _write_text(body_path, body)
        _upload_text_if_changed(body_path, paths["summary_body"], "text/markdown; charset=utf-8")
        manifest["summary"]["body_sha256"] = _sha256_bytes(body.encode("utf-8"))
        manifest["summary"]["render_version"] = note_render.RENDER_VERSION
        manifest["note_writer"]["units"] = {
            "unbraced_imperial": list(unit_report.unbraced_imperial),
            "unknown_units": list(unit_report.unknown_units),
        }
        metadata_values = {
            field: list(manifest["metadata"][field])
            for field in ("topics", "people", "tags")
        }
    else:
        tag_registry = TagRegistry(bucket=bucket)
        vocabulary = semantic_projection(tag_registry.load())
        source_fingerprint = source_generation_fingerprint_from_episode(episode)
        description = fetch_episode_description(episode.get("feed_url"), episode.get("guid"))
        result = note_writer.generate(
            episode,
            transcript,
            vocabulary,
            episode_key=episode_key,
            source_fingerprint=source_fingerprint,
            description=description,
        )
        generated_at = now_iso()
        body = result.body
        body_bytes = body.encode("utf-8")
        metadata_values = tag_registry.resolve_and_record(
            episode_key,
            metadata.normalize_metadata(result.note),
        )

        _write_text(body_path, body)
        _upload_text_if_changed(body_path, paths["summary_body"], "text/markdown; charset=utf-8")

        manifest = {
            "schema_version": KNOWLEDGE_SCHEMA_VERSION,
            "episode_key": episode_key,
            "summary": {
                "policy_version": NOTE_WRITER_POLICY_VERSION,
                "preset": preset,
                "input_fingerprint": fingerprint,
                "generated_at": generated_at,
                "body_sha256": _sha256_bytes(body_bytes),
                "render_version": note_render.RENDER_VERSION,
            },
            "note_writer": {
                **note_writer.audit_record(result),
                "episode_spend": _episode_spend(episode_key, source_fingerprint),
                "description_sha256": (
                    _sha256_bytes(description.encode("utf-8")) if description else None
                ),
                "note": result.note,
                "tag_vocabulary_sha256": _sha256_bytes(
                    json.dumps(
                        vocabulary, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ),
            },
            "metadata": {
                "policy_version": NOTE_WRITER_POLICY_VERSION,
                "preset": preset,
                "input_fingerprint": fingerprint,
                "generated_at": generated_at,
                "topics": metadata_values["topics"],
                "people": metadata_values["people"],
                "tags": metadata_values["tags"],
                "tag_candidates": metadata_values["tag_candidates"],
                "unknown_existing_tags": metadata_values["unknown_existing_tags"],
            },
        }

    return _publish_note(episode, local_root, paths, manifest, metadata_values, body)


def _publish_note(
    episode: dict,
    local_root: Path,
    paths: dict,
    manifest: dict,
    metadata_values: dict,
    body: str,
) -> str:
    """Render frontmatter + body and upload the final note and manifest."""

    final_path = _write_text(
        local_root / "summary.md",
        render(episode, metadata_values, manifest["summary"]["generated_at"], body),
    )
    manifest_path = _write_text(
        local_root / "metadata.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    _upload_text_if_changed(
        final_path,
        paths["summary"],
        "text/markdown; charset=utf-8",
    )
    _upload_text_if_changed(
        manifest_path,
        paths["summary_metadata"],
        "application/json",
    )
    return str(final_path)


def build_knowledge_note(episode: dict) -> str | None:
    """Build one cacheable summary body, metadata manifest, and final note.

    The caller receives the local final ``summary.md`` path so the existing
    pipeline can pass it unchanged into ``update_episode(markdown_file=...)``.

    When ``PODCAST_KNOWLEDGE_WRITER_PRESET`` is configured, the TASK-118
    single-pass writer produces the note. Otherwise the legacy chain runs:
    all three canonical artifacts are written only after the V2 reviewer
    chain is validated and its immutable evidence is persisted.
    """

    writer_preset = note_writer.note_writer_preset()
    if writer_preset is not None:
        return _build_single_pass_note(episode, writer_preset)

    reviewer_preset = summary_review.active_summary_review_preset()
    if reviewer_preset is None:
        return None

    episode_key = episode["episode_key"]
    paths = paths_for(episode_key)
    local_root = RUNTIME_DIR / episode_key
    compiled_path = _download_canonical_compiled_transcript(
        episode,
        local_root / "compiled-transcript.txt",
    )
    transcript_bytes = compiled_path.read_bytes()
    try:
        transcript = transcript_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Compiled transcript must be UTF-8 text") from error

    bucket = get_bucket()
    manifest = _load_manifest(paths["summary_metadata"])
    tag_registry = TagRegistry(bucket=bucket)
    vocabulary = semantic_projection(tag_registry.load())
    summary_fingerprint = summary.input_fingerprint(episode, transcript_bytes)
    # TASK-076 Task 7: the same immutable source-generation identity the
    # episode AI budget ledger keys on (podcast_engine.episode_generation),
    # resolved lazily below only when a stage actually needs to spend --
    # a full cache hit across summary/review and metadata never downloads
    # source bytes again or touches the budget ledger at all.
    source_fingerprint: str | None = None
    body_path = local_root / "body.md"
    body_blob = bucket.blob(paths["summary_body"])

    cached_body_bytes = body_blob.download_as_bytes() if body_blob.exists() else None
    cache_hit = (
        isinstance(cached_body_bytes, bytes)
        and _summary_matches(
            manifest,
            summary_fingerprint,
            summary.summary_preset(),
            reviewer_preset,
            cached_body_bytes,
        )
        and _accepted_review_artifacts_match(
            bucket,
            episode_key,
            manifest,
            cached_body_bytes,
            transcript_bytes,
        )
    )

    if cache_hit:
        print("Knowledge summary cache hit")
        try:
            body = cached_body_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Cached summary body must be UTF-8 text") from error
        _write_text(body_path, body)
    else:
        if source_fingerprint is None:
            source_fingerprint = source_generation_fingerprint_from_episode(episode)
        draft = summary.generate(
            episode,
            transcript,
            episode_key=episode_key,
            source_fingerprint=source_fingerprint,
        )
        review_context = build_review_context(transcript, draft)
        try:
            review_output = summary_review.generate(
                episode,
                transcript,
                draft,
                review_context,
                episode_key=episode_key,
                source_fingerprint=source_fingerprint,
            )
        except summary_review.SummaryReviewValidationError as error:
            persist_review_attempt(
                bucket=bucket,
                episode_key=episode_key,
                transcript=transcript,
                draft=draft,
                review_context=review_context,
                audit_result=None,
                edit_result=None,
                accepted_final=None,
                failure={"stage": error.stage, "code": error.code},
                summary_policy_version=SUMMARY_POLICY_VERSION,
                summary_preset=summary.summary_preset(),
                review_policy_version=SUMMARY_REVIEW_POLICY_VERSION,
                review_preset=reviewer_preset,
                completion_metadata=error.completion_metadata,
                reviewed_at=now_iso(),
            )
            _notify_if_summary_review_stalled(
                bucket=bucket,
                episode=episode,
                transcript=transcript,
                draft=draft,
                review_preset=reviewer_preset,
            )
            raise

        if review_output.get("review_context") is not review_context:
            raise RuntimeError("Summary reviewer did not preserve the Python-owned review context")

        reviewed_at = now_iso()
        audit_result = review_output.get("audit_result")
        edit_result = review_output.get("edit_result")
        accepted_final = review_output.get("accepted_final_markdown")
        completion_metadata = review_output.get("completion_metadata")
        terminal_failure = (
            isinstance(audit_result, dict) and audit_result.get("status") == "fail"
        )
        persisted_review = persist_review_attempt(
            bucket=bucket,
            episode_key=episode_key,
            transcript=transcript,
            draft=draft,
            review_context=review_context,
            audit_result=audit_result,
            edit_result=edit_result,
            accepted_final=accepted_final,
            failure=(
                {"stage": "audit", "code": "review_rejected"}
                if terminal_failure
                else None
            ),
            summary_policy_version=SUMMARY_POLICY_VERSION,
            summary_preset=summary.summary_preset(),
            review_policy_version=SUMMARY_REVIEW_POLICY_VERSION,
            review_preset=reviewer_preset,
            completion_metadata=completion_metadata,
            reviewed_at=reviewed_at,
        )
        if terminal_failure:
            _notify_if_summary_review_stalled(
                bucket=bucket,
                episode=episode,
                transcript=transcript,
                draft=draft,
                review_preset=reviewer_preset,
            )

        if not isinstance(accepted_final, str) or not accepted_final:
            raise RuntimeError("Summary reviewer rejected the draft")

        body = accepted_final
        _write_text(body_path, body)
        _upload_text_if_changed(
            body_path,
            paths["summary_body"],
            "text/markdown; charset=utf-8",
        )
        manifest = {
            "schema_version": KNOWLEDGE_SCHEMA_VERSION,
            "episode_key": episode_key,
            "summary": {
                "policy_version": SUMMARY_POLICY_VERSION,
                "preset": summary.summary_preset(),
                "input_fingerprint": summary_fingerprint,
                "generated_at": reviewed_at,
            },
            "summary_review": {
                "policy_version": SUMMARY_REVIEW_POLICY_VERSION,
                "preset": reviewer_preset,
                "review_id": persisted_review["review_id"],
                "artifact_root": persisted_review["artifact_root"],
                "status": audit_result["status"],
                "final_sha256": persisted_review["final_sha256"],
            },
        }

    body_bytes = body.encode("utf-8")
    metadata_fingerprint = metadata.input_fingerprint(
        episode,
        body_bytes,
        transcript_bytes,
        vocabulary,
    )
    if _metadata_matches(manifest, metadata_fingerprint, metadata.metadata_preset()):
        print("Knowledge metadata cache hit")
        metadata_values = {
            field: list(manifest["metadata"][field])
            for field in ("topics", "people", "tags")
        }
    else:
        if source_fingerprint is None:
            source_fingerprint = source_generation_fingerprint_from_episode(episode)
        extracted_metadata = metadata.generate(
            episode,
            body,
            transcript,
            vocabulary,
            episode_key=episode_key,
            source_fingerprint=source_fingerprint,
        )
        metadata_values = tag_registry.resolve_and_record(episode_key, extracted_metadata)
        manifest["metadata"] = {
            "policy_version": METADATA_POLICY_VERSION,
            "preset": metadata.metadata_preset(),
            "input_fingerprint": metadata_fingerprint,
            "generated_at": now_iso(),
            "topics": metadata_values["topics"],
            "people": metadata_values["people"],
            "tags": metadata_values["tags"],
            "tag_candidates": metadata_values["tag_candidates"],
            "unknown_existing_tags": metadata_values["unknown_existing_tags"],
        }

    return _publish_note(episode, local_root, paths, manifest, metadata_values, body)


__all__ = ["build_knowledge_note", "canonical_summary_generated_at"]
