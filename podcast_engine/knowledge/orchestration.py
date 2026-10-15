"""Build and persist versioned podcast knowledge notes through storage-backed orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..episode_contract import now_iso, paths_for
from ..episode_generation import source_generation_fingerprint_from_episode
from ..rss import fetch_episode_description
from ..storage import download_file_from_gcs, get_bucket
from .frontmatter import render
from .notes_v3 import render as note_render
from .notes_v3 import schema as note_schema
from .notes_v3 import writer as note_writer
from .notes_v3.units import UnitReport
from .models import (
    KNOWLEDGE_SCHEMA_VERSION,
    NOTE_WRITER_POLICY_VERSION,
    NOTE_WRITER_PRESET_ENV,
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

    The writer returns topics,
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
            note_schema.normalize_metadata(result.note),
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


def build_knowledge_note(episode: dict) -> str:
    """Build one cacheable note body, metadata manifest, and final note.

    The caller receives the local final ``summary.md`` path so the existing
    pipeline can pass it unchanged into ``update_episode(markdown_file=...)``.
    The TASK-118 single-pass writer is the only note path (TASK-146), so a
    Worker without ``PODCAST_KNOWLEDGE_WRITER_PRESET`` fails closed here
    instead of publishing nothing silently.
    """

    writer_preset = note_writer.note_writer_preset()
    if writer_preset is None:
        raise RuntimeError(
            f"{NOTE_WRITER_PRESET_ENV} is required to build knowledge notes"
        )
    return _build_single_pass_note(episode, writer_preset)


__all__ = ["build_knowledge_note", "canonical_summary_generated_at"]
