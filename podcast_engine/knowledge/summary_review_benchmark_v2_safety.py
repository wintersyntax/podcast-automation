"""Paid-operator safety boundary for Summary Review Benchmark v2.

This module is installed onto the canonical benchmark module after its audited
implementation snapshot is executed.  It deliberately composes the existing
state machine, evaluator, journal, and decision-lock code instead of forking
those semantics.
"""

from __future__ import annotations

import contextvars
import fcntl
import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from functools import wraps
from pathlib import Path

from .summary_review_benchmark_v2_journal import (
    _validate_state as _validate_execution_journal_state,
)


BENCHMARK_V2_EVIDENCE_MAX_AGE = timedelta(hours=24)
BENCHMARK_V2_AUTHORIZATION_FORMAT = 1
BENCHMARK_V2_BUDGET_USD = "6.00"
BENCHMARK_V2_PROVIDER_PRICING_SOURCE_API = (
    "https://openrouter.ai/api/v1/models/:author/:slug/endpoints"
)

_ACTIVE_REPOSITORY_ROOT: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "benchmark_v2_active_repository_root",
    default=None,
)
_ALLOW_PREFLIGHT_LOCKS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "benchmark_v2_allow_preflight_locks",
    default=False,
)
_ALLOW_PREFLIGHT_ARTIFACT_WRITE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "benchmark_v2_allow_preflight_artifact_write",
    default=False,
)
_EXPECTED_SERVED_PROVIDER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "benchmark_v2_expected_served_provider",
    default=None,
)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                _json_safe(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Benchmark v2 operator artifact must be canonical JSON") from error


def _parse_canonical_json_bytes(data: bytes, *, label: str) -> dict:
    if not isinstance(data, bytes) or not data:
        raise ValueError(f"Benchmark v2 {label} must be non-empty exact bytes")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Benchmark v2 {label} must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark v2 {label} must be a JSON object")
    if data != _canonical_json_bytes(payload):
        raise ValueError(f"Benchmark v2 {label} bytes must be canonical JSON")
    return payload


def _parse_utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"Benchmark v2 {label} captured_at_utc must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(
            f"Benchmark v2 {label} captured_at_utc must be canonical UTC"
        ) from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"Benchmark v2 {label} captured_at_utc must be UTC")
    return parsed.astimezone(timezone.utc)


def _select_raw_endpoint(
    *,
    model_id: str,
    routing_provider_slug: object,
    served_provider_identity: object,
    raw_response: object,
) -> dict:
    if (
        not isinstance(routing_provider_slug, str)
        or not routing_provider_slug.strip()
        or not isinstance(served_provider_identity, str)
        or not served_provider_identity.strip()
        or not isinstance(raw_response, dict)
        or not raw_response
    ):
        raise ValueError(
            f"Benchmark v2 provider/pricing capture entry is invalid for {model_id}"
        )
    data = raw_response.get("data")
    endpoints = data.get("endpoints") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or data.get("id") != model_id
        or not isinstance(endpoints, list)
    ):
        raise ValueError(f"Benchmark v2 raw endpoint response is invalid for {model_id}")
    matching_endpoints = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        provider_name = endpoint.get("provider_name")
        endpoint_name = endpoint.get("name")
        observed_identity = (
            provider_name if isinstance(provider_name, str) else endpoint_name
        )
        if observed_identity == served_provider_identity:
            matching_endpoints.append(endpoint)
    if not matching_endpoints:
        raise ValueError(
            f"Benchmark v2 raw capture requires exactly one selected endpoint for {model_id}"
        )
    if len(matching_endpoints) == 1:
        endpoint = matching_endpoints[0]
    else:
        route_matches = [
            candidate
            for candidate in matching_endpoints
            if candidate.get("tag") == routing_provider_slug
        ]
        if len(route_matches) != 1:
            raise ValueError(
                f"Benchmark v2 raw capture requires exactly one selected endpoint for {model_id}"
            )
        endpoint = route_matches[0]
    return endpoint


def _derive_provider_pricing_from_raw_endpoint(
    *,
    model_id: str,
    routing_provider_slug: object,
    served_provider_identity: object,
    raw_response: object,
) -> tuple[dict, dict]:
    endpoint = _select_raw_endpoint(
        model_id=model_id,
        routing_provider_slug=routing_provider_slug,
        served_provider_identity=served_provider_identity,
        raw_response=raw_response,
    )
    raw_pricing = endpoint.get("pricing")
    context_length = endpoint.get("context_length")
    prompt_price = raw_pricing.get("prompt") if isinstance(raw_pricing, dict) else None
    completion_price = (
        raw_pricing.get("completion") if isinstance(raw_pricing, dict) else None
    )
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length <= 0
        or not isinstance(prompt_price, str)
        or not isinstance(completion_price, str)
    ):
        raise ValueError(f"Benchmark v2 raw endpoint pricing is invalid for {model_id}")
    try:
        parsed_prompt_price = Decimal(prompt_price)
        parsed_completion_price = Decimal(completion_price)
    except InvalidOperation as error:
        raise ValueError(
            f"Benchmark v2 raw endpoint pricing is invalid for {model_id}"
        ) from error
    if (
        not parsed_prompt_price.is_finite()
        or parsed_prompt_price < 0
        or not parsed_completion_price.is_finite()
        or parsed_completion_price < 0
    ):
        raise ValueError(f"Benchmark v2 raw endpoint pricing is invalid for {model_id}")
    return (
        {
            "routing_provider_slug": routing_provider_slug,
            "compatible_served_providers": [served_provider_identity],
        },
        {
            "routing_provider_slug": routing_provider_slug,
            "available_provider_slugs": [routing_provider_slug],
            "prompt_usd_per_token": prompt_price,
            "completion_usd_per_token": completion_price,
            "context_length_tokens": context_length,
        },
    )


def validate_benchmark_v2_provider_pricing_snapshot(
    provider_lock: dict,
    pricing_evidence: dict,
    *,
    repository_root: Path | None = None,
    now_utc: datetime | None = None,
    max_age: timedelta = BENCHMARK_V2_EVIDENCE_MAX_AGE,
    require_present: bool = True,
    check_age: bool = True,
) -> dict:
    """Validate one coherent provider/pricing metadata snapshot.

    `require_present=False` exists for legacy unit-only bundle fixtures that carry
    no snapshot metadata at all.  Any partially present metadata still fails.
    Canonical authorization and paid execution always call this with the strict
    defaults, so there is no operator freshness bypass.
    """

    if not isinstance(provider_lock, dict) or not isinstance(pricing_evidence, dict):
        raise ValueError("Benchmark v2 provider/pricing snapshot payloads must be objects")
    if not isinstance(max_age, timedelta) or max_age <= timedelta(0):
        raise ValueError("Benchmark v2 provider/pricing snapshot max age is invalid")

    fields = ("captured_at_utc", "source_api", "metadata_snapshots")
    provider_has_any = any(field in provider_lock for field in fields)
    pricing_has_any = any(field in pricing_evidence for field in fields)
    if not provider_has_any and not pricing_has_any and not require_present:
        return {}

    if any(field not in provider_lock for field in fields) or any(
        field not in pricing_evidence for field in fields
    ):
        raise ValueError("Benchmark v2 provider/pricing snapshot metadata is incomplete")

    captured_at = provider_lock["captured_at_utc"]
    if pricing_evidence["captured_at_utc"] != captured_at:
        raise ValueError("Benchmark v2 provider/pricing snapshot timestamp mismatch")

    source_api = provider_lock["source_api"]
    if (
        not isinstance(source_api, str)
        or not source_api.strip()
        or pricing_evidence["source_api"] != source_api
    ):
        raise ValueError("Benchmark v2 provider/pricing snapshot source mismatch")

    metadata_snapshots = provider_lock["metadata_snapshots"]
    if (
        not isinstance(metadata_snapshots, dict)
        or not metadata_snapshots
        or pricing_evidence["metadata_snapshots"] != metadata_snapshots
    ):
        raise ValueError("Benchmark v2 provider/pricing metadata snapshots mismatch")
    _canonical_json_bytes(metadata_snapshots)

    if repository_root is not None:
        provider_models = provider_lock.get("models")
        pricing_models = pricing_evidence.get("models")
        if not isinstance(provider_models, dict) or not isinstance(pricing_models, dict):
            raise ValueError("Benchmark v2 provider/pricing snapshot models are invalid")
        if set(provider_models) != set(pricing_models):
            raise ValueError("Benchmark v2 provider/pricing snapshot model sets mismatch")
        if set(metadata_snapshots) != set(provider_models):
            raise ValueError(
                "Benchmark v2 provider/pricing snapshot artifact map is incomplete"
            )
        if source_api != BENCHMARK_V2_PROVIDER_PRICING_SOURCE_API:
            raise ValueError("Benchmark v2 provider/pricing snapshot source is not canonical")
        snapshots_root = _workspace_root(Path(repository_root)) / "metadata-snapshots"
        workspace = _workspace_root(Path(repository_root))
        if workspace.is_symlink() or snapshots_root.is_symlink():
            raise ValueError("Benchmark v2 snapshot artifact workspace must not be a symlink")
        for model_id in sorted(provider_models):
            digest = metadata_snapshots.get(model_id)
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            ):
                raise ValueError("Benchmark v2 snapshot artifact digest is invalid")
            artifact_path = snapshots_root / f"{digest.removeprefix('sha256:')}.json"
            if (
                artifact_path.resolve(strict=False).parent
                != snapshots_root.resolve(strict=False)
                or not artifact_path.is_file()
                or artifact_path.is_symlink()
            ):
                raise ValueError(
                    f"Benchmark v2 snapshot artifact is unavailable for {model_id}"
                )
            try:
                artifact_bytes = artifact_path.read_bytes()
            except OSError as error:
                raise ValueError(
                    f"Benchmark v2 snapshot artifact is unreadable for {model_id}"
                ) from error
            artifact = _parse_canonical_json_bytes(
                artifact_bytes,
                label=f"snapshot artifact for {model_id}",
            )
            if _sha256_bytes(artifact_bytes) != digest:
                raise ValueError(
                    f"Benchmark v2 snapshot artifact digest mismatch for {model_id}"
                )
            if (
                set(artifact)
                != {
                    "captured_at_utc",
                    "model_id",
                    "pricing",
                    "provider",
                    "raw_response",
                    "snapshot_format",
                    "source_api",
                }
                or artifact.get("captured_at_utc") != captured_at
                or artifact.get("model_id") != model_id
                or artifact.get("pricing") != pricing_models[model_id]
                or artifact.get("provider") != provider_models[model_id]
                or not isinstance(artifact.get("raw_response"), dict)
                or not artifact["raw_response"]
                or artifact.get("snapshot_format") != 1
                or artifact.get("source_api") != source_api
            ):
                raise ValueError(
                    f"Benchmark v2 snapshot artifact content mismatch for {model_id}"
                )
            provider_entry = artifact["provider"]
            compatible = (
                provider_entry.get("compatible_served_providers")
                if isinstance(provider_entry, dict)
                else None
            )
            if not isinstance(compatible, list) or len(compatible) != 1:
                raise ValueError(
                    f"Benchmark v2 snapshot artifact provider identity is invalid for {model_id}"
                )
            derived_provider, derived_pricing = _derive_provider_pricing_from_raw_endpoint(
                model_id=model_id,
                routing_provider_slug=provider_entry.get("routing_provider_slug"),
                served_provider_identity=compatible[0],
                raw_response=artifact["raw_response"],
            )
            if derived_provider != provider_models[model_id] or derived_pricing != pricing_models[model_id]:
                raise ValueError(
                    f"Benchmark v2 snapshot raw endpoint content mismatch for {model_id}"
                )

    captured_dt = _parse_utc_timestamp(captured_at, label="provider/pricing snapshot")
    if check_age:
        now = datetime.now(timezone.utc) if now_utc is None else now_utc
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("Benchmark v2 provider/pricing snapshot current time is invalid")
        now = now.astimezone(timezone.utc)
        if captured_dt > now:
            raise ValueError("Benchmark v2 provider/pricing snapshot timestamp is in the future")
        if now - captured_dt > max_age:
            raise ValueError("Benchmark v2 provider/pricing evidence is stale")

    return {
        "captured_at_utc": captured_at,
        "source_api": source_api,
        "metadata_snapshots": metadata_snapshots,
    }


def _workspace_root(repository_root: Path) -> Path:
    return Path(repository_root) / "var" / "benchmarks" / "summary-review-v2"


def _artifact_path(repository_root: Path, filename: str) -> Path:
    return _workspace_root(repository_root) / filename


def _require_workspace_artifact_path(repository_root: Path, path: Path) -> Path:
    root = _workspace_root(repository_root)
    target = Path(path)
    if target != root / target.name:
        raise ValueError("Benchmark v2 operator artifact path is not canonical")
    if target.is_symlink():
        raise ValueError("Benchmark v2 operator artifact path must not be a symlink")
    if root.is_symlink():
        raise ValueError("Benchmark v2 operator workspace must not be a symlink")
    if target.resolve(strict=False).parent != root.resolve(strict=False):
        raise ValueError("Benchmark v2 operator artifact path escapes canonical workspace")
    return target


def _write_immutable_json(repository_root: Path, path: Path, payload: dict) -> dict:
    target = _require_workspace_artifact_path(repository_root, path)
    expected_bytes = _canonical_json_bytes(payload)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError("Benchmark v2 operator workspace is unavailable") from error

    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise ValueError("Benchmark v2 operator artifact target is not a file")
        try:
            existing = target.read_bytes()
        except OSError as error:
            raise ValueError("Benchmark v2 operator artifact is unreadable") from error
        if existing != expected_bytes:
            raise ValueError("Benchmark v2 operator artifact is immutable and conflicts")
        return {"path": target, "payload": payload, "bytes": expected_bytes}

    try:
        with target.open("xb") as handle:
            handle.write(expected_bytes)
    except FileExistsError:
        try:
            existing = target.read_bytes()
        except OSError as error:
            raise ValueError("Benchmark v2 operator artifact is unreadable") from error
        if existing != expected_bytes:
            raise ValueError("Benchmark v2 operator artifact is immutable and conflicts")
    except OSError as error:
        raise ValueError("Benchmark v2 operator artifact could not be written") from error
    return {"path": target, "payload": payload, "bytes": expected_bytes}


def _read_canonical_artifact(repository_root: Path, filename: str) -> tuple[Path, bytes, dict]:
    path = _require_workspace_artifact_path(
        repository_root,
        _artifact_path(repository_root, filename),
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Benchmark v2 required {filename} artifact is unavailable")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Benchmark v2 required {filename} artifact is unreadable") from error
    return path, data, _parse_canonical_json_bytes(data, label=filename)


def _write_content_addressed_snapshot(
    repository_root: Path,
    *,
    digest: str,
    artifact_bytes: bytes,
) -> Path:
    workspace = _workspace_root(repository_root)
    snapshots_root = workspace / "metadata-snapshots"
    if workspace.is_symlink() or snapshots_root.is_symlink():
        raise ValueError("Benchmark v2 snapshot artifact workspace must not be a symlink")
    path = snapshots_root / f"{digest.removeprefix('sha256:')}.json"
    if path.resolve(strict=False).parent != snapshots_root.resolve(strict=False):
        raise ValueError("Benchmark v2 snapshot artifact path is not canonical")
    try:
        snapshots_root.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != artifact_bytes:
                raise ValueError("Benchmark v2 snapshot artifact is immutable and conflicts")
            return path
        with path.open("xb") as handle:
            handle.write(artifact_bytes)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != artifact_bytes:
            raise ValueError("Benchmark v2 snapshot artifact is immutable and conflicts")
    except OSError as error:
        raise ValueError("Benchmark v2 snapshot artifact could not be written") from error
    return path


def _bound_decision(binding: object) -> bool:
    return isinstance(binding, dict) and isinstance(binding.get("benchmark_binding"), dict)


def _extract_selected_provider_from_lock(lock_payload: dict, *, model_id: str) -> str:
    if not isinstance(lock_payload, dict) or lock_payload.get("decision") != "advance":
        raise ValueError("Benchmark v2 qualification lock does not advance a candidate")
    selected = lock_payload.get("selected_candidate")
    if not isinstance(selected, dict) or selected.get("candidate_id") != model_id:
        raise ValueError("Benchmark v2 Geoffrey candidate conflicts with qualification lock")
    provider = selected.get("observed_served_provider_identity")
    if not isinstance(provider, str) or not provider:
        raise ValueError("Benchmark v2 qualification winner provider identity is unavailable")
    return provider


def _selected_provider_from_response(response: object) -> str | None:
    if not isinstance(response, dict):
        return None
    metadata = response.get("openrouter_metadata")
    endpoints = (
        metadata.get("endpoints", {}).get("available", [])
        if isinstance(metadata, dict)
        else []
    )
    if not isinstance(endpoints, list):
        return None
    selected = [
        item
        for item in endpoints
        if isinstance(item, dict) and item.get("selected") is True
    ]
    if len(selected) != 1:
        return None
    provider = selected[0].get("provider")
    return provider if isinstance(provider, str) and provider else None


def _qualified_provider_from_journal(
    namespace: dict,
    *,
    lock_dir: Path,
    journal_path: Path,
    benchmark_binding: dict,
    ordered_case_keys: list[str],
    model_id: str,
) -> str:
    qualification_path = Path(lock_dir) / "qualification.lock.json"
    if not qualification_path.is_file() or qualification_path.is_symlink():
        raise ValueError("Benchmark v2 qualification lock is unavailable for Geoffrey")
    try:
        lock_bytes = qualification_path.read_bytes()
    except OSError as error:
        raise ValueError("Benchmark v2 qualification lock is unreadable") from error
    lock_payload = _parse_canonical_json_bytes(
        lock_bytes,
        label="qualification lock",
    )
    expected = _extract_selected_provider_from_lock(lock_payload, model_id=model_id)

    journal = namespace["load_or_initialize_execution_journal"](
        Path(journal_path),
        benchmark_binding=benchmark_binding,
        ordered_case_keys=ordered_case_keys,
    )
    completed = journal.get("completed_cases")
    if not isinstance(completed, list):
        raise ValueError("Benchmark v2 qualification provider evidence is invalid")
    providers = set()
    prefix = f"{model_id}::"
    for item in completed:
        if not isinstance(item, dict):
            continue
        case_key = item.get("case_key")
        if not isinstance(case_key, str) or not case_key.startswith(prefix):
            continue
        evidence = item.get("execution_evidence")
        provider = (
            evidence.get("served_provider_identity")
            if isinstance(evidence, dict)
            else None
        )
        if not isinstance(provider, str) or not provider:
            raise ValueError("Benchmark v2 qualification provider evidence is invalid")
        providers.add(provider)
    if providers != {expected}:
        raise ValueError(
            "Benchmark v2 qualification winner provider evidence is absent, ambiguous, or inconsistent"
        )
    return expected


def _extract_selected_provider_from_designated_tracks_lock(
    lock_payload: dict, *, model_id: str
) -> str:
    if not isinstance(lock_payload, dict):
        raise ValueError("Benchmark v2 designated-tracks lock payload is invalid")
    designated_tracks = lock_payload.get("designated_tracks")
    if not isinstance(designated_tracks, dict):
        raise ValueError("Benchmark v2 designated-tracks lock is missing tracks")
    track_entry = designated_tracks.get(model_id)
    if not isinstance(track_entry, dict) or track_entry.get("eligible") is not True:
        raise ValueError(
            "Benchmark v2 designated-tracks lock does not make this candidate eligible"
        )
    selected = track_entry.get("candidate_result")
    if not isinstance(selected, dict) or selected.get("candidate_id") != model_id:
        raise ValueError(
            "Benchmark v2 Geoffrey candidate conflicts with designated-tracks lock"
        )
    provider = selected.get("observed_served_provider_identity")
    if not isinstance(provider, str) or not provider:
        raise ValueError(
            "Benchmark v2 designated-tracks candidate provider identity is unavailable"
        )
    return provider


def _qualified_provider_from_designated_tracks_journal(
    namespace: dict,
    *,
    lock_dir: Path,
    journal_path: Path,
    benchmark_binding: dict,
    ordered_case_keys: list[str],
    model_id: str,
) -> str:
    designated_tracks_path = Path(lock_dir) / "qualification.designated-tracks.lock.json"
    if not designated_tracks_path.is_file() or designated_tracks_path.is_symlink():
        raise ValueError(
            "Benchmark v2 designated-tracks lock is unavailable for Geoffrey"
        )
    try:
        lock_bytes = designated_tracks_path.read_bytes()
    except OSError as error:
        raise ValueError("Benchmark v2 designated-tracks lock is unreadable") from error
    lock_payload = _parse_canonical_json_bytes(
        lock_bytes,
        label="designated-tracks lock",
    )
    expected = _extract_selected_provider_from_designated_tracks_lock(
        lock_payload, model_id=model_id
    )

    journal = namespace["load_or_initialize_execution_journal"](
        Path(journal_path),
        benchmark_binding=benchmark_binding,
        ordered_case_keys=ordered_case_keys,
    )
    completed = journal.get("completed_cases")
    if not isinstance(completed, list):
        raise ValueError("Benchmark v2 designated-tracks provider evidence is invalid")
    providers = set()
    prefix = f"{model_id}::"
    for item in completed:
        if not isinstance(item, dict):
            continue
        case_key = item.get("case_key")
        if not isinstance(case_key, str) or not case_key.startswith(prefix):
            continue
        evidence = item.get("execution_evidence")
        provider = (
            evidence.get("served_provider_identity")
            if isinstance(evidence, dict)
            else None
        )
        if not isinstance(provider, str) or not provider:
            raise ValueError("Benchmark v2 designated-tracks provider evidence is invalid")
        providers.add(provider)
    if providers != {expected}:
        raise ValueError(
            "Benchmark v2 designated-tracks candidate provider evidence is absent, ambiguous, or inconsistent"
        )
    return expected


@contextmanager
def _paid_runner_lock(repository_root: Path):
    root = _workspace_root(repository_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError("Benchmark v2 paid runner workspace is unavailable") from error
    path = root / "paid-runner.lock"
    if path.is_symlink():
        raise ValueError("Benchmark v2 paid runner lock must not be a symlink")

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    locked = False
    try:
        try:
            descriptor = os.open(path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Benchmark v2 paid runner lock must be a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(
                    "Benchmark v2 paid execution already has an active runner"
                ) from error
            locked = True
        except OSError as error:
            raise ValueError(
                "Benchmark v2 paid runner lock is unavailable"
            ) from error
        yield path
    finally:
        if descriptor is not None:
            if locked:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(descriptor)


def install_paid_operator_safety(namespace: dict) -> None:
    """Install the paid-operator safety contract onto the canonical module."""

    original_build_bundle = namespace["build_benchmark_v2_run_bundle"]
    original_preflight = namespace["run_benchmark_v2_paid_authorization_preflight"]
    original_paid_runner = namespace["run_benchmark_v2_paid_execution"]
    original_execute_case = namespace["execute_case"]
    original_validate_served_provider = namespace["validate_served_provider"]
    original_final_gate = namespace["run_geoffrey_final_gate_with_qualification_lock"]
    original_qualification_writer = namespace["write_qualification_decision_lock"]
    original_geoffrey_writer = namespace["write_geoffrey_final_decision_lock"]
    original_qualification_designated_tracks_writer = namespace[
        "write_qualification_designated_tracks_lock"
    ]
    original_geoffrey_track_writer = namespace[
        "write_geoffrey_final_decision_lock_for_track"
    ]
    original_final_gate_for_track = namespace[
        "run_geoffrey_final_gate_with_designated_track_lock"
    ]
    original_locked_dry_run = namespace.get("run_locked_qualification_geoffrey_dry_run")

    def validate_benchmark_v2_request_parameter_compatibility(
        *,
        config: dict,
        provider_lock: dict,
        pricing_evidence: dict,
        repository_root: Path,
    ) -> dict:
        """Prove actual request parameters fit every frozen locked endpoint."""

        repository_root = Path(repository_root)
        validate_benchmark_v2_provider_pricing_snapshot(
            provider_lock,
            pricing_evidence,
            repository_root=repository_root,
            check_age=False,
        )
        models = config.get("models") if isinstance(config, dict) else None
        provider_models = (
            provider_lock.get("models")
            if isinstance(provider_lock, dict)
            else None
        )
        metadata_snapshots = (
            provider_lock.get("metadata_snapshots")
            if isinstance(provider_lock, dict)
            else None
        )
        if (
            not isinstance(models, list)
            or not isinstance(provider_models, dict)
            or set(provider_models) != set(models)
            or not isinstance(metadata_snapshots, dict)
            or set(metadata_snapshots) != set(models)
        ):
            raise ValueError(
                "Benchmark v2 request parameter candidates do not match frozen evidence"
            )

        required_parameters = {
            "max_tokens",
            "response_format",
            "structured_outputs",
        }
        result = {}
        snapshots_root = _workspace_root(repository_root) / "metadata-snapshots"
        for model_id in models:
            temperature = namespace["_temperature_for_model"](config, model_id)
            digest = metadata_snapshots[model_id]
            artifact_path = snapshots_root / f"{digest.removeprefix('sha256:')}.json"
            artifact = _parse_canonical_json_bytes(
                artifact_path.read_bytes(),
                label=f"snapshot artifact for {model_id}",
            )
            provider_entry = provider_models[model_id]
            compatible_providers = provider_entry.get(
                "compatible_served_providers"
            )
            endpoint = _select_raw_endpoint(
                model_id=model_id,
                routing_provider_slug=provider_entry.get("routing_provider_slug"),
                served_provider_identity=compatible_providers[0],
                raw_response=artifact.get("raw_response"),
            )
            supported_parameters = endpoint.get("supported_parameters")
            if (
                not isinstance(supported_parameters, list)
                or any(
                    not isinstance(parameter, str) or not parameter
                    for parameter in supported_parameters
                )
                or len(set(supported_parameters)) != len(supported_parameters)
            ):
                raise ValueError(
                    f"Benchmark v2 supported parameters are invalid for {model_id}"
                )
            supported = set(supported_parameters)
            missing = sorted(required_parameters - supported)
            if missing:
                raise ValueError(
                    "Benchmark v2 locked endpoint does not support required "
                    f"parameters for {model_id}: {', '.join(missing)}"
                )
            if temperature is None:
                if "temperature" in supported:
                    raise ValueError(
                        "Benchmark v2 temperature exception is stale for "
                        f"{model_id}: locked endpoint now supports temperature"
                    )
                temperature_status = "omitted_unsupported"
            else:
                if "temperature" not in supported:
                    raise ValueError(
                        "Benchmark v2 locked endpoint does not support required "
                        f"temperature for {model_id}"
                    )
                temperature_status = "required_supported"
            result[model_id] = {
                "required_parameters": sorted(required_parameters),
                "temperature": temperature_status,
            }
        return result

    def install_benchmark_v2_provider_pricing_capture(
        *,
        repository_root: Path,
        capture_bytes: bytes,
        now_utc: datetime | None = None,
    ) -> dict:
        """Install or rotate exact provider/pricing evidence before paid state."""

        repository_root = Path(repository_root)
        capture = _parse_canonical_json_bytes(
            capture_bytes,
            label="provider/pricing capture",
        )
        if set(capture) != {
            "capture_format",
            "captured_at_utc",
            "source_api",
            "models",
        } or capture.get("capture_format") != 1:
            raise ValueError("Benchmark v2 provider/pricing capture shape is invalid")
        if capture.get("source_api") != BENCHMARK_V2_PROVIDER_PRICING_SOURCE_API:
            raise ValueError("Benchmark v2 provider/pricing capture source is not canonical")
        models = capture.get("models")
        if not isinstance(models, dict):
            raise ValueError("Benchmark v2 provider/pricing capture models are invalid")
        config, _config_bytes = namespace["load_canonical_benchmark_v2_config"](
            root=repository_root
        )
        canonical_models = config.get("models")
        if (
            not isinstance(canonical_models, list)
            or set(models) != set(canonical_models)
            or len(models) != len(canonical_models)
        ):
            raise ValueError(
                "Benchmark v2 provider/pricing capture candidates are not canonical"
            )

        common = {
            "captured_at_utc": capture.get("captured_at_utc"),
            "source_api": capture.get("source_api"),
        }
        provider_lock = {**common, "models": {}}
        pricing_evidence = {**common, "models": {}}
        snapshot_bytes_by_digest = {}
        metadata_snapshots = {}
        for model_id in canonical_models:
            entry = models.get(model_id)
            if not isinstance(entry, dict) or set(entry) != {
                "raw_response",
                "routing_provider_slug",
                "served_provider_identity",
            }:
                raise ValueError(
                    f"Benchmark v2 provider/pricing capture entry is invalid for {model_id}"
                )
            raw_response = entry.get("raw_response")
            provider, pricing = _derive_provider_pricing_from_raw_endpoint(
                model_id=model_id,
                routing_provider_slug=entry.get("routing_provider_slug"),
                served_provider_identity=entry.get("served_provider_identity"),
                raw_response=raw_response,
            )
            provider_lock["models"][model_id] = provider
            pricing_evidence["models"][model_id] = pricing
            snapshot_payload = {
                "captured_at_utc": common["captured_at_utc"],
                "model_id": model_id,
                "pricing": pricing,
                "provider": provider,
                "raw_response": raw_response,
                "snapshot_format": 1,
                "source_api": common["source_api"],
            }
            snapshot_bytes = _canonical_json_bytes(snapshot_payload)
            digest = _sha256_bytes(snapshot_bytes)
            metadata_snapshots[model_id] = digest
            snapshot_bytes_by_digest[digest] = snapshot_bytes
        provider_lock["metadata_snapshots"] = metadata_snapshots
        pricing_evidence["metadata_snapshots"] = metadata_snapshots
        snapshot = validate_benchmark_v2_provider_pricing_snapshot(
            provider_lock,
            pricing_evidence,
            now_utc=now_utc,
        )
        provider_bytes = _canonical_json_bytes(provider_lock)
        pricing_bytes = _canonical_json_bytes(pricing_evidence)
        workspace = _workspace_root(repository_root)

        with _paid_runner_lock(repository_root):
            paid_authorization_path = workspace / "paid-authorization.json"
            if paid_authorization_path.exists() or paid_authorization_path.is_symlink():
                raise ValueError(
                    "Benchmark v2 provider/pricing refresh is forbidden after paid authorization"
                )
            journal_path = workspace / "execution-state.json"
            if journal_path.exists() or journal_path.is_symlink():
                if journal_path.is_symlink() or not journal_path.is_file():
                    raise ValueError("Benchmark v2 paid state journal is invalid")
                journal = _parse_canonical_json_bytes(
                    journal_path.read_bytes(),
                    label="paid state journal",
                )
                if (
                    journal.get("spent_usd") != "0.00"
                    or journal.get("attempts") != []
                    or journal.get("completed_case_keys") != []
                    or journal.get("completed_cases") != []
                ):
                    raise ValueError(
                        "Benchmark v2 provider/pricing refresh is forbidden after paid state"
                    )
            paid_locks = workspace / "locks"
            if paid_locks.exists() or paid_locks.is_symlink():
                if paid_locks.is_symlink() or not paid_locks.is_dir():
                    raise ValueError("Benchmark v2 paid decision lock directory is invalid")
                try:
                    if any(paid_locks.iterdir()):
                        raise ValueError(
                            "Benchmark v2 provider/pricing refresh is forbidden after paid state"
                        )
                except OSError as error:
                    raise ValueError(
                        "Benchmark v2 paid decision lock directory is unreadable"
                    ) from error

            provider_path = workspace / "provider-lock.json"
            pricing_path = workspace / "pricing-evidence.json"
            current_provider = (
                provider_path.read_bytes()
                if provider_path.is_file() and not provider_path.is_symlink()
                else None
            )
            current_pricing = (
                pricing_path.read_bytes()
                if pricing_path.is_file() and not pricing_path.is_symlink()
                else None
            )
            if current_provider == provider_bytes and current_pricing == pricing_bytes:
                for digest, artifact_bytes in snapshot_bytes_by_digest.items():
                    _write_content_addressed_snapshot(
                        repository_root,
                        digest=digest,
                        artifact_bytes=artifact_bytes,
                    )
                validate_benchmark_v2_provider_pricing_snapshot(
                    provider_lock,
                    pricing_evidence,
                    repository_root=repository_root,
                    now_utc=now_utc,
                )
                return {
                    "status": "unchanged",
                    "capture_sha256": _sha256_bytes(capture_bytes),
                    "provider_lock_bytes": provider_bytes,
                    "pricing_evidence_bytes": pricing_bytes,
                    "provider_pricing_snapshot": snapshot,
                }

            archive_targets = [
                provider_path,
                pricing_path,
                workspace / "preflight-result.json",
                workspace / "preflight-authorization.json",
                journal_path,
                workspace / "preflight",
            ]
            existing_targets = [path for path in archive_targets if path.exists()]
            if existing_targets:
                archive_material = b"\0".join(
                    path.name.encode("utf-8")
                    + b"\0"
                    + (path.read_bytes() if path.is_file() and not path.is_symlink() else b"dir")
                    for path in existing_targets
                )
                archive_root = workspace / "history" / hashlib.sha256(
                    archive_material
                ).hexdigest()
                if workspace.is_symlink() or (workspace / "history").is_symlink():
                    raise ValueError("Benchmark v2 refresh history path is invalid")
                try:
                    archive_root.mkdir(parents=True, exist_ok=False)
                    for path in existing_targets:
                        if path.is_symlink():
                            raise ValueError("Benchmark v2 refresh source must not be a symlink")
                        os.replace(path, archive_root / path.name)
                except OSError as error:
                    raise ValueError("Benchmark v2 prior evidence could not be archived") from error

            for digest, artifact_bytes in snapshot_bytes_by_digest.items():
                _write_content_addressed_snapshot(
                    repository_root,
                    digest=digest,
                    artifact_bytes=artifact_bytes,
                )
            _write_immutable_json(repository_root, provider_path, provider_lock)
            _write_immutable_json(repository_root, pricing_path, pricing_evidence)
            validate_benchmark_v2_provider_pricing_snapshot(
                provider_lock,
                pricing_evidence,
                repository_root=repository_root,
                now_utc=now_utc,
            )
            return {
                "status": "installed",
                "capture_sha256": _sha256_bytes(capture_bytes),
                "provider_lock_bytes": provider_bytes,
                "pricing_evidence_bytes": pricing_bytes,
                "provider_pricing_snapshot": snapshot,
            }

    def reconcile_benchmark_v2_aborted_execution_epoch(
        *,
        repository_root: Path,
        evidence_bytes: bytes,
    ) -> dict:
        repository_root = Path(repository_root)
        evidence = _parse_canonical_json_bytes(
            evidence_bytes,
            label="aborted execution reconciliation evidence",
        )
        workspace = _workspace_root(repository_root)
        journal_path = workspace / "execution-state.json"

        with _paid_runner_lock(repository_root):
            if workspace.is_symlink() or (workspace / "history").is_symlink():
                raise ValueError(
                    "Benchmark v2 reconciliation workspace must not be a symlink"
                )
            if not journal_path.is_file() or journal_path.is_symlink():
                raise ValueError(
                    "Benchmark v2 reconciliation requires an execution journal"
                )

            decision_lock_dir = workspace / "locks"
            if decision_lock_dir.is_symlink():
                raise ValueError(
                    "Benchmark v2 reconciliation decision lock directory must not be a symlink"
                )
            if decision_lock_dir.exists():
                if not decision_lock_dir.is_dir():
                    raise ValueError(
                        "Benchmark v2 reconciliation decision lock path must be a directory"
                    )
                for filename in (
                    "qualification.lock.json",
                    "qualification.no_winner.lock.json",
                    "winner.lock.json",
                    "no_winner.lock.json",
                ):
                    if (decision_lock_dir / filename).exists():
                        raise ValueError(
                            "Benchmark v2 reconciliation refuses an existing decision lock"
                        )

            try:
                journal_bytes = journal_path.read_bytes()
            except OSError as error:
                raise ValueError(
                    "Benchmark v2 reconciliation execution journal is unreadable"
                ) from error

            journal = _parse_canonical_json_bytes(
                journal_bytes,
                label="reconciliation execution journal",
            )
            _validate_execution_journal_state(journal)
            attempts = journal.get("attempts")
            if (
                journal.get("spent_usd") != "0.00"
                or journal.get("completed_case_keys") != []
                or journal.get("completed_cases") != []
                or not isinstance(attempts, list)
                or len(attempts) != 1
                or not isinstance(attempts[0], dict)
                or attempts[0].get("status")
                not in {"reserved", "transport_failed"}
            ):
                raise ValueError(
                    "Benchmark v2 reconciliation requires one zero-spend unresolved attempt"
                )

            expected_fields = {
                "reconciliation_format",
                "kind",
                "request_sha256",
                "provider_usage_before",
                "provider_usage_after",
                "generation_count",
                "upstream_request_count",
                "observed_at_utc",
            }
            if (
                set(evidence) != expected_fields
                or evidence.get("reconciliation_format") != 1
                or evidence.get("kind")
                != "benchmark_v2_aborted_execution_reconciliation"
                or evidence.get("request_sha256")
                != attempts[0].get("request_sha256")
                or evidence.get("provider_usage_before")
                != evidence.get("provider_usage_after")
                or evidence.get("generation_count") != 0
                or evidence.get("upstream_request_count") != 0
            ):
                raise ValueError(
                    "Benchmark v2 reconciliation evidence does not prove zero provider execution"
                )

            archive_targets = [
                journal_path,
                workspace / "paid-authorization.json",
                workspace / "preflight-authorization.json",
                workspace / "preflight-result.json",
                workspace / "preflight",
            ]
            archive_material = (
                evidence_bytes + b"\0" + journal_bytes
            )
            archive_root = (
                workspace
                / "history"
                / (
                    "aborted-"
                    + hashlib.sha256(archive_material).hexdigest()
                )
            )

            try:
                archive_root.mkdir(parents=True, exist_ok=False)
                with (archive_root / "reconciliation-evidence.json").open("xb") as handle:
                    handle.write(evidence_bytes)
                for source in archive_targets:
                    if not source.exists():
                        continue
                    if source.is_symlink():
                        raise ValueError(
                            "Benchmark v2 reconciliation source must not be a symlink"
                        )
                    os.replace(source, archive_root / source.name)
            except OSError as error:
                raise ValueError(
                    "Benchmark v2 aborted execution epoch could not be archived"
                ) from error

            return {
                "status": "reconciled",
                "archive_path": archive_root,
                "reconciliation_sha256": _sha256_bytes(evidence_bytes),
            }

    def require_canonical_benchmark_v2_lock_dir(
        lock_dir: Path,
        *,
        repository_root: Path | None = None,
        allow_preflight: bool | None = None,
    ) -> Path:
        candidate = Path(lock_dir)
        active_root = _ACTIVE_REPOSITORY_ROOT.get()
        if repository_root is not None:
            root = Path(repository_root)
        elif active_root is not None:
            root = active_root
        else:
            root = Path(namespace["REPOSITORY_ROOT"])

        allow_pf = _ALLOW_PREFLIGHT_LOCKS.get() if allow_preflight is None else allow_preflight
        workspace = _workspace_root(root)
        allowed = [workspace / "locks"]
        if allow_pf:
            allowed.append(workspace / "preflight" / "locks")

        if candidate.is_symlink():
            raise ValueError("Benchmark v2 decision lock directory must not be a symlink")
        candidate_resolved = candidate.resolve(strict=False)
        if not any(candidate_resolved == path.resolve(strict=False) for path in allowed):
            raise ValueError(
                "Benchmark v2 bound decision lock directory must be canonical"
            )
        return candidate

    def require_canonical_benchmark_v2_geoffrey_track_lock_dir(
        lock_dir: Path,
        *,
        repository_root: Path | None = None,
        allow_preflight: bool | None = None,
    ) -> Path:
        candidate = Path(lock_dir)
        active_root = _ACTIVE_REPOSITORY_ROOT.get()
        if repository_root is not None:
            root = Path(repository_root)
        elif active_root is not None:
            root = active_root
        else:
            root = Path(namespace["REPOSITORY_ROOT"])

        allow_pf = _ALLOW_PREFLIGHT_LOCKS.get() if allow_preflight is None else allow_preflight
        workspace = _workspace_root(root)
        config, _config_bytes = namespace["load_canonical_benchmark_v2_config"](root=root)
        designated_candidate_ids = (
            config.get("geoffrey_designated_candidates")
            if isinstance(config, dict)
            else None
        )
        if not isinstance(designated_candidate_ids, list) or not designated_candidate_ids:
            raise ValueError(
                "Benchmark v2 Geoffrey track lock directory requires designated candidates"
            )
        allowed = []
        for candidate_id in designated_candidate_ids:
            slug = namespace["_geoffrey_track_slug"](candidate_id)
            allowed.append(workspace / "locks" / "geoffrey" / slug)
            if allow_pf:
                allowed.append(workspace / "preflight" / "locks" / "geoffrey" / slug)

        if candidate.is_symlink():
            raise ValueError("Benchmark v2 decision lock directory must not be a symlink")
        candidate_resolved = candidate.resolve(strict=False)
        if not any(candidate_resolved == path.resolve(strict=False) for path in allowed):
            raise ValueError(
                "Benchmark v2 bound Geoffrey track lock directory must be canonical"
            )
        return candidate

    @wraps(original_qualification_writer)
    def guarded_qualification_writer(*args, **kwargs):
        lock_dir = kwargs.get("lock_dir")
        if lock_dir is None and args:
            lock_dir = args[0]
        require_canonical_benchmark_v2_lock_dir(lock_dir)
        return original_qualification_writer(*args, **kwargs)

    @wraps(original_geoffrey_writer)
    def guarded_geoffrey_writer(*args, **kwargs):
        lock_dir = kwargs.get("lock_dir")
        if lock_dir is None and args:
            lock_dir = args[0]
        require_canonical_benchmark_v2_lock_dir(lock_dir)
        return original_geoffrey_writer(*args, **kwargs)

    @wraps(original_qualification_designated_tracks_writer)
    def guarded_qualification_designated_tracks_writer(*args, **kwargs):
        lock_dir = kwargs.get("lock_dir")
        if lock_dir is None and args:
            lock_dir = args[0]
        require_canonical_benchmark_v2_lock_dir(lock_dir)
        return original_qualification_designated_tracks_writer(*args, **kwargs)

    @wraps(original_geoffrey_track_writer)
    def guarded_geoffrey_track_writer(*args, **kwargs):
        lock_dir = kwargs.get("lock_dir")
        if lock_dir is None and args:
            lock_dir = args[0]
        require_canonical_benchmark_v2_geoffrey_track_lock_dir(lock_dir)
        return original_geoffrey_track_writer(*args, **kwargs)

    @wraps(original_validate_served_provider)
    def hardened_validate_served_provider(
        provider_lock,
        model_id,
        served_provider,
        *,
        expected_served_provider=None,
    ):
        active_expected = _EXPECTED_SERVED_PROVIDER.get()
        if active_expected is not None:
            if (
                expected_served_provider is not None
                and expected_served_provider != active_expected
            ):
                raise ValueError(
                    "Benchmark v2 served-provider continuity context conflicts"
                )
            expected_served_provider = active_expected
        return original_validate_served_provider(
            provider_lock,
            model_id,
            served_provider,
            expected_served_provider=expected_served_provider,
        )

    @wraps(original_execute_case)
    def hardened_execute_case(*args, expected_served_provider=None, **kwargs):
        if expected_served_provider is None:
            return original_execute_case(*args, **kwargs)
        if (
            not isinstance(expected_served_provider, str)
            or not expected_served_provider
        ):
            raise ValueError(
                "Benchmark v2 expected served provider must be non-empty"
            )
        token = _EXPECTED_SERVED_PROVIDER.set(expected_served_provider)
        try:
            result = original_execute_case(*args, **kwargs)
        finally:
            _EXPECTED_SERVED_PROVIDER.reset(token)
        if (
            not isinstance(result, dict)
            or result.get("served_provider") != expected_served_provider
        ):
            raise ValueError(
                "Benchmark v2 served provider changed from expected identity"
            )
        return result

    @wraps(original_build_bundle)
    def hardened_build_bundle(*args, **kwargs):
        bundle = original_build_bundle(*args, **kwargs)
        validate_benchmark_v2_provider_pricing_snapshot(
            bundle["provider_lock"],
            bundle["pricing_evidence"],
            require_present=False,
            check_age=False,
        )
        metadata_snapshots = bundle["provider_lock"].get("metadata_snapshots")
        if metadata_snapshots:
            repository_root = kwargs.get("repository_root")
            if repository_root is None and args:
                repository_root = args[0]
            validate_benchmark_v2_request_parameter_compatibility(
                config=bundle["config"],
                provider_lock=bundle["provider_lock"],
                pricing_evidence=bundle["pricing_evidence"],
                repository_root=Path(repository_root),
            )
        return bundle

    def write_benchmark_v2_preflight_artifact(
        *,
        repository_root: Path,
        preflight_result: dict,
        provider_lock: dict,
        pricing_evidence: dict,
    ) -> dict:
        if not _ALLOW_PREFLIGHT_ARTIFACT_WRITE.get():
            raise ValueError(
                "Benchmark v2 preflight artifact writer requires the canonical preflight"
            )
        if not isinstance(preflight_result, dict):
            raise ValueError("Benchmark v2 preflight result must be an object")
        binding = namespace["_require_benchmark_binding"](
            preflight_result.get("benchmark_binding")
        )
        snapshot = validate_benchmark_v2_provider_pricing_snapshot(
            provider_lock,
            pricing_evidence,
            repository_root=Path(repository_root),
        )
        if not snapshot:
            raise ValueError(
                "Benchmark v2 preflight artifact requires provider/pricing snapshot metadata"
            )
        result_artifact = _write_immutable_json(
            Path(repository_root),
            _artifact_path(Path(repository_root), "preflight-result.json"),
            preflight_result,
        )
        payload = {
            "authorization_format": BENCHMARK_V2_AUTHORIZATION_FORMAT,
            "kind": "benchmark_v2_preflight",
            "benchmark_binding": binding,
            "budget_usd": BENCHMARK_V2_BUDGET_USD,
            "provider_pricing_snapshot": snapshot,
            "preflight_result_sha256": _sha256_bytes(result_artifact["bytes"]),
        }
        return _write_immutable_json(
            Path(repository_root),
            _artifact_path(Path(repository_root), "preflight-authorization.json"),
            payload,
        )

    @wraps(original_preflight)
    def hardened_preflight(*args, **kwargs):
        repository_root = kwargs.get("repository_root")
        if repository_root is None and args:
            repository_root = args[0]
        repository_root = Path(repository_root)
        with _paid_runner_lock(repository_root):
            root_token = _ACTIVE_REPOSITORY_ROOT.set(repository_root)
            preflight_token = _ALLOW_PREFLIGHT_LOCKS.set(True)
            try:
                result = original_preflight(*args, **kwargs)
            finally:
                _ALLOW_PREFLIGHT_LOCKS.reset(preflight_token)
                _ACTIVE_REPOSITORY_ROOT.reset(root_token)

            provider_bytes = kwargs.get("provider_lock_bytes")
            pricing_bytes = kwargs.get("pricing_evidence_bytes")
            try:
                provider = _parse_canonical_json_bytes(
                    provider_bytes,
                    label="provider lock",
                )
                pricing = _parse_canonical_json_bytes(
                    pricing_bytes,
                    label="pricing evidence",
                )
                snapshot = validate_benchmark_v2_provider_pricing_snapshot(
                    provider,
                    pricing,
                    repository_root=repository_root,
                )
                config, _config_bytes = namespace[
                    "load_canonical_benchmark_v2_config"
                ](root=repository_root)
                parameter_compatibility = (
                    validate_benchmark_v2_request_parameter_compatibility(
                        config=config,
                        provider_lock=provider,
                        pricing_evidence=pricing,
                        repository_root=repository_root,
                    )
                )
            except ValueError:
                # Legacy unit seams replace the canonical bundle builder and use
                # sentinel bytes. They may inspect the pure result, but cannot
                # produce either durable preflight artifact or paid authorization.
                if namespace.get("build_benchmark_v2_run_bundle") is not hardened_build_bundle:
                    return result
                raise

            if isinstance(result, dict):
                result = {
                    **result,
                    "request_parameter_compatibility": parameter_compatibility,
                }

            if snapshot and isinstance(result, dict) and isinstance(
                result.get("benchmark_binding"), dict
            ):
                artifact_token = _ALLOW_PREFLIGHT_ARTIFACT_WRITE.set(True)
                try:
                    write_benchmark_v2_preflight_artifact(
                        repository_root=repository_root,
                        preflight_result=result,
                        provider_lock=provider,
                        pricing_evidence=pricing,
                    )
                finally:
                    _ALLOW_PREFLIGHT_ARTIFACT_WRITE.reset(artifact_token)
            return result

    def validate_preflight_result_artifact(
        *,
        repository_root: Path,
        binding: dict,
        preflight_payload: dict,
        readiness: dict | None = None,
        request_parameter_compatibility: dict | None = None,
    ) -> dict:
        _path, result_bytes, result_payload = _read_canonical_artifact(
            repository_root,
            "preflight-result.json",
        )
        if preflight_payload.get("preflight_result_sha256") != _sha256_bytes(
            result_bytes
        ):
            raise ValueError("Benchmark v2 preflight result hash mismatch")
        if (
            result_payload.get("status") != "READY FOR PAID AUTHORIZATION"
            or result_payload.get("paid_execution_authorized") is not False
            or result_payload.get("mode") != "zero_paid_preflight"
            or result_payload.get("benchmark_binding") != binding
            or result_payload.get("workspace_root")
            != str(_workspace_root(repository_root))
        ):
            raise ValueError("Benchmark v2 preflight result artifact is invalid")
        dry_run = result_payload.get("dry_run")
        designated_tracks = (
            dry_run.get("designated_tracks") if isinstance(dry_run, dict) else None
        )
        designated_tracks_binding = (
            designated_tracks.get("binding")
            if isinstance(designated_tracks, dict)
            else None
        )
        if (
            not isinstance(dry_run, dict)
            or dry_run.get("mode") != "dry_run"
            or dry_run.get("spent_usd") != "0.00"
            or dry_run.get("benchmark_binding") != binding
            or not isinstance(designated_tracks_binding, dict)
            or designated_tracks_binding.get("benchmark_binding") != binding
        ):
            raise ValueError("Benchmark v2 preflight result dry-run proof is invalid")
        frozen = result_payload.get("frozen_readiness")
        if (
            not isinstance(frozen, dict)
            or frozen.get("provider_lock_sha256") != binding["provider_lock_sha256"]
            or frozen.get("pricing_evidence_sha256")
            != binding["pricing_evidence_sha256"]
        ):
            raise ValueError("Benchmark v2 preflight result frozen evidence is invalid")
        stored_readiness = result_payload.get("paid_readiness")
        if not isinstance(stored_readiness, dict) or stored_readiness.get("ready") is not True:
            raise ValueError("Benchmark v2 preflight result readiness is invalid")
        if readiness is not None and stored_readiness != _json_safe(readiness):
            raise ValueError("Benchmark v2 preflight result readiness has drifted")
        stored_parameter_compatibility = result_payload.get(
            "request_parameter_compatibility"
        )
        if (
            request_parameter_compatibility is not None
            and stored_parameter_compatibility
            != _json_safe(request_parameter_compatibility)
        ):
            raise ValueError(
                "Benchmark v2 preflight request parameter compatibility has drifted"
            )
        return result_payload

    def _authorize_benchmark_v2_paid_execution_unlocked(
        *,
        repository_root: Path,
        provider_lock_bytes: bytes,
        pricing_evidence_bytes: bytes,
    ) -> dict:
        """Issue authorization only from an existing matching zero-paid preflight."""

        repository_root = Path(repository_root)
        provider = _parse_canonical_json_bytes(
            provider_lock_bytes,
            label="provider lock",
        )
        pricing = _parse_canonical_json_bytes(
            pricing_evidence_bytes,
            label="pricing evidence",
        )
        snapshot = validate_benchmark_v2_provider_pricing_snapshot(
            provider,
            pricing,
            repository_root=repository_root,
        )

        geoffrey_transcript_bytes, geoffrey_draft_bytes = namespace[
            "load_canonical_geoffrey_fixture_bytes"
        ](repository_root=repository_root)
        bundle = namespace["build_benchmark_v2_run_bundle"](
            repository_root=repository_root,
            provider_lock_bytes=provider_lock_bytes,
            pricing_evidence_bytes=pricing_evidence_bytes,
            geoffrey_transcript_bytes=geoffrey_transcript_bytes,
            geoffrey_draft_bytes=geoffrey_draft_bytes,
        )
        binding = namespace["_require_benchmark_binding"](
            bundle["benchmark_binding"]
        )
        parameter_compatibility = None
        if namespace.get("build_benchmark_v2_run_bundle") is hardened_build_bundle:
            parameter_compatibility = validate_benchmark_v2_request_parameter_compatibility(
                config=bundle["config"],
                provider_lock=bundle["provider_lock"],
                pricing_evidence=bundle["pricing_evidence"],
                repository_root=repository_root,
            )
        if str(bundle["config"].get("budget_usd")) != BENCHMARK_V2_BUDGET_USD:
            raise ValueError("Benchmark v2 paid authorization budget conflicts with config")

        _path, preflight_bytes, preflight_payload = _read_canonical_artifact(
            repository_root,
            "preflight-authorization.json",
        )
        expected_preflight_fields = {
            "authorization_format",
            "kind",
            "benchmark_binding",
            "budget_usd",
            "provider_pricing_snapshot",
            "preflight_result_sha256",
        }
        preflight_result_sha256 = preflight_payload.get("preflight_result_sha256")
        if (
            set(preflight_payload) != expected_preflight_fields
            or preflight_payload.get("authorization_format")
            != BENCHMARK_V2_AUTHORIZATION_FORMAT
            or preflight_payload.get("kind") != "benchmark_v2_preflight"
            or preflight_payload.get("benchmark_binding") != binding
            or preflight_payload.get("budget_usd") != BENCHMARK_V2_BUDGET_USD
            or preflight_payload.get("provider_pricing_snapshot") != snapshot
            or not isinstance(preflight_result_sha256, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", preflight_result_sha256) is None
        ):
            raise ValueError("Benchmark v2 preflight authorization artifact is invalid")

        workspace = bundle.get("workspace")
        if not isinstance(workspace, dict):
            raise ValueError("Benchmark v2 operator workspace result is invalid")
        journal_path = workspace.get("execution_state_path")
        if not isinstance(journal_path, Path):
            raise ValueError("Benchmark v2 operator execution journal path is invalid")
        journal_state = namespace["load_or_initialize_execution_journal"](
            journal_path,
            benchmark_binding=binding,
            ordered_case_keys=bundle["ordered_case_keys"],
        )
        readiness = namespace["build_paid_execution_readiness"](
            run_bundle=bundle,
            journal_state=journal_state,
        )
        if not isinstance(readiness, dict) or readiness.get("ready") is not True:
            raise ValueError("Benchmark v2 paid authorization requires ready preflight state")
        validate_preflight_result_artifact(
            repository_root=repository_root,
            binding=binding,
            preflight_payload=preflight_payload,
            readiness=readiness,
            request_parameter_compatibility=parameter_compatibility,
        )

        payload = {
            "authorization_format": BENCHMARK_V2_AUTHORIZATION_FORMAT,
            "kind": "benchmark_v2_paid_authorization",
            "benchmark_binding": binding,
            "budget_usd": BENCHMARK_V2_BUDGET_USD,
            "preflight_artifact_sha256": _sha256_bytes(preflight_bytes),
            "provider_pricing_snapshot": snapshot,
        }
        return _write_immutable_json(
            repository_root,
            _artifact_path(repository_root, "paid-authorization.json"),
            payload,
        )

    def authorize_benchmark_v2_paid_execution(
        *,
        repository_root: Path,
        provider_lock_bytes: bytes,
        pricing_evidence_bytes: bytes,
    ) -> dict:
        repository_root = Path(repository_root)
        with _paid_runner_lock(repository_root):
            return _authorize_benchmark_v2_paid_execution_unlocked(
                repository_root=repository_root,
                provider_lock_bytes=provider_lock_bytes,
                pricing_evidence_bytes=pricing_evidence_bytes,
            )

    def validate_benchmark_v2_paid_authorization(
        *,
        repository_root: Path,
        provider_lock_bytes: bytes,
        pricing_evidence_bytes: bytes,
        authorization_bytes: bytes,
    ) -> dict:
        repository_root = Path(repository_root)
        geoffrey_transcript_bytes, geoffrey_draft_bytes = namespace[
            "load_canonical_geoffrey_fixture_bytes"
        ](repository_root=repository_root)
        bundle = namespace["build_benchmark_v2_run_bundle"](
            repository_root=repository_root,
            provider_lock_bytes=provider_lock_bytes,
            pricing_evidence_bytes=pricing_evidence_bytes,
            geoffrey_transcript_bytes=geoffrey_transcript_bytes,
            geoffrey_draft_bytes=geoffrey_draft_bytes,
        )
        binding = namespace["_require_benchmark_binding"](
            bundle["benchmark_binding"]
        )
        snapshot = validate_benchmark_v2_provider_pricing_snapshot(
            bundle["provider_lock"],
            bundle["pricing_evidence"],
            repository_root=repository_root,
        )
        parameter_compatibility = None
        if namespace.get("build_benchmark_v2_run_bundle") is hardened_build_bundle:
            parameter_compatibility = validate_benchmark_v2_request_parameter_compatibility(
                config=bundle["config"],
                provider_lock=bundle["provider_lock"],
                pricing_evidence=bundle["pricing_evidence"],
                repository_root=repository_root,
            )

        auth_payload = _parse_canonical_json_bytes(
            authorization_bytes,
            label="paid authorization",
        )
        _auth_path, stored_auth_bytes, stored_auth_payload = _read_canonical_artifact(
            repository_root,
            "paid-authorization.json",
        )
        if stored_auth_bytes != authorization_bytes or stored_auth_payload != auth_payload:
            raise ValueError("Benchmark v2 paid authorization bytes do not match canonical artifact")

        expected_fields = {
            "authorization_format",
            "kind",
            "benchmark_binding",
            "budget_usd",
            "preflight_artifact_sha256",
            "provider_pricing_snapshot",
        }
        if set(auth_payload) != expected_fields:
            raise ValueError("Benchmark v2 paid authorization artifact shape is invalid")
        if (
            auth_payload.get("authorization_format")
            != BENCHMARK_V2_AUTHORIZATION_FORMAT
            or auth_payload.get("kind") != "benchmark_v2_paid_authorization"
            or auth_payload.get("benchmark_binding") != binding
            or auth_payload.get("budget_usd") != BENCHMARK_V2_BUDGET_USD
            or auth_payload.get("provider_pricing_snapshot") != snapshot
        ):
            raise ValueError("Benchmark v2 paid authorization does not match current run binding")
        if str(bundle["config"].get("budget_usd")) != BENCHMARK_V2_BUDGET_USD:
            raise ValueError("Benchmark v2 paid authorization budget conflicts with config")

        _preflight_path, preflight_bytes, preflight_payload = _read_canonical_artifact(
            repository_root,
            "preflight-authorization.json",
        )
        if auth_payload.get("preflight_artifact_sha256") != _sha256_bytes(preflight_bytes):
            raise ValueError("Benchmark v2 paid authorization preflight hash mismatch")
        if (
            preflight_payload.get("authorization_format")
            != BENCHMARK_V2_AUTHORIZATION_FORMAT
            or preflight_payload.get("kind") != "benchmark_v2_preflight"
            or preflight_payload.get("benchmark_binding") != binding
            or preflight_payload.get("budget_usd") != BENCHMARK_V2_BUDGET_USD
            or preflight_payload.get("provider_pricing_snapshot") != snapshot
            or not isinstance(preflight_payload.get("preflight_result_sha256"), str)
        ):
            raise ValueError("Benchmark v2 paid authorization preflight artifact is invalid")
        validate_preflight_result_artifact(
            repository_root=repository_root,
            binding=binding,
            preflight_payload=preflight_payload,
            request_parameter_compatibility=parameter_compatibility,
        )
        return auth_payload

    def run_benchmark_v2_paid_execution(
        *,
        repository_root: Path,
        provider_lock_bytes: bytes,
        pricing_evidence_bytes: bytes,
        authorization_bytes: bytes,
        transport,
    ) -> dict:
        """Run paid Benchmark v2 only with the exact canonical authorization artifact."""

        repository_root = Path(repository_root)
        # Authorization is validated before transport callability is even
        # considered, keeping the public core boundary fail-closed.
        validate_benchmark_v2_paid_authorization(
            repository_root=repository_root,
            provider_lock_bytes=provider_lock_bytes,
            pricing_evidence_bytes=pricing_evidence_bytes,
            authorization_bytes=authorization_bytes,
        )
        if not callable(transport):
            raise ValueError("Benchmark v2 paid transport must be callable")

        with _paid_runner_lock(repository_root):
            validate_benchmark_v2_paid_authorization(
                repository_root=repository_root,
                provider_lock_bytes=provider_lock_bytes,
                pricing_evidence_bytes=pricing_evidence_bytes,
                authorization_bytes=authorization_bytes,
            )
            root_token = _ACTIVE_REPOSITORY_ROOT.set(repository_root)
            preflight_token = _ALLOW_PREFLIGHT_LOCKS.set(False)
            previous_execute_case = namespace["execute_case"]
            previous_final_gate = namespace["run_geoffrey_final_gate_with_qualification_lock"]
            previous_final_gate_for_track = namespace[
                "run_geoffrey_final_gate_with_designated_track_lock"
            ]

            @wraps(original_execute_case)
            def provider_continuous_execute_case(*args, **kwargs):
                case_key = kwargs.get("case_key")
                if not (isinstance(case_key, str) and case_key.startswith("geoffrey::")):
                    return hardened_execute_case(*args, **kwargs)

                model_id = kwargs.get("model_id")
                journal_path = kwargs.get("journal_path")
                benchmark_binding = kwargs.get("benchmark_binding")
                ordered_case_keys = kwargs.get("ordered_case_keys")
                if (
                    not isinstance(model_id, str)
                    or not isinstance(journal_path, Path)
                    or not isinstance(benchmark_binding, dict)
                    or not isinstance(ordered_case_keys, list)
                ):
                    raise ValueError("Benchmark v2 Geoffrey provider continuity inputs are invalid")

                segments = case_key.split("::")
                lock_dir = _workspace_root(repository_root) / "locks"
                if (
                    len(segments) == 3
                    and segments[0] == "geoffrey"
                    and segments[2].startswith("run-")
                ):
                    expected_provider = _qualified_provider_from_designated_tracks_journal(
                        namespace,
                        lock_dir=lock_dir,
                        journal_path=journal_path,
                        benchmark_binding=benchmark_binding,
                        ordered_case_keys=ordered_case_keys,
                        model_id=model_id,
                    )
                elif (
                    len(segments) == 2
                    and segments[0] == "geoffrey"
                    and segments[1].startswith("run-")
                ):
                    expected_provider = _qualified_provider_from_journal(
                        namespace,
                        lock_dir=lock_dir,
                        journal_path=journal_path,
                        benchmark_binding=benchmark_binding,
                        ordered_case_keys=ordered_case_keys,
                        model_id=model_id,
                    )
                else:
                    raise ValueError(
                        "Benchmark v2 Geoffrey case key does not match a known provider continuity shape"
                    )
                return hardened_execute_case(
                    *args,
                    expected_served_provider=expected_provider,
                    **kwargs,
                )

            @wraps(original_final_gate)
            def provider_continuous_final_gate(*args, **kwargs):
                executor = kwargs.get("geoffrey_run_executor")
                if executor is None and len(args) >= 5:
                    executor = args[4]
                if not callable(executor):
                    return original_final_gate(*args, **kwargs)

                lock_dir = kwargs.get("lock_dir")
                expected_binding = kwargs.get("expected_binding")
                if lock_dir is None and args:
                    lock_dir = args[0]
                if expected_binding is None and len(args) >= 2:
                    expected_binding = args[1]
                journal_path = _workspace_root(repository_root) / "execution-state.json"

                def guarded_executor(selected_candidate, run_number):
                    model_id = (
                        selected_candidate.get("candidate_id")
                        if isinstance(selected_candidate, dict)
                        else None
                    )
                    if not isinstance(model_id, str) or not model_id:
                        raise ValueError("Benchmark v2 Geoffrey candidate is invalid")
                    # Use the already-persisted live journal to bind Geoffrey
                    # to the provider that actually served qualification.
                    try:
                        journal_payload = json.loads(
                            journal_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError) as error:
                        raise ValueError(
                            "Benchmark v2 qualification journal is unavailable"
                        ) from error
                    ordered_keys = journal_payload.get("ordered_case_keys")
                    if not isinstance(ordered_keys, list) or not ordered_keys:
                        raise ValueError(
                            "Benchmark v2 qualification journal ordering is invalid"
                        )
                    expected_provider = _qualified_provider_from_journal(
                        namespace,
                        lock_dir=Path(lock_dir),
                        journal_path=journal_path,
                        benchmark_binding=namespace["_require_benchmark_binding"](
                            expected_binding.get("benchmark_binding")
                        ),
                        ordered_case_keys=ordered_keys,
                        model_id=model_id,
                    )
                    result = executor(selected_candidate, run_number)
                    if (
                        not isinstance(result, dict)
                        or result.get("served_provider") != expected_provider
                    ):
                        raise ValueError(
                            "Benchmark v2 Geoffrey served provider drifted from qualification winner"
                        )
                    return result

                guarded_kwargs = dict(kwargs)
                guarded_kwargs["geoffrey_run_executor"] = guarded_executor
                return original_final_gate(*args, **guarded_kwargs)

            @wraps(original_final_gate_for_track)
            def provider_continuous_final_gate_for_track(*args, **kwargs):
                executor = kwargs.get("geoffrey_run_executor")
                if executor is None and len(args) >= 7:
                    executor = args[6]
                if not callable(executor):
                    return original_final_gate_for_track(*args, **kwargs)

                qualification_lock_dir = kwargs.get("qualification_lock_dir")
                if qualification_lock_dir is None and args:
                    qualification_lock_dir = args[0]
                candidate_id = kwargs.get("candidate_id")
                if candidate_id is None and len(args) >= 4:
                    candidate_id = args[3]
                expected_binding = kwargs.get("expected_binding")
                if expected_binding is None and len(args) >= 3:
                    expected_binding = args[2]
                if (
                    qualification_lock_dir is None
                    or not isinstance(candidate_id, str)
                    or not candidate_id
                    or not isinstance(expected_binding, dict)
                ):
                    raise ValueError(
                        "Benchmark v2 Geoffrey track provider continuity inputs are invalid"
                    )
                journal_path = _workspace_root(repository_root) / "execution-state.json"

                def guarded_executor(selected_candidate, run_number):
                    model_id = (
                        selected_candidate.get("candidate_id")
                        if isinstance(selected_candidate, dict)
                        else None
                    )
                    if (
                        not isinstance(model_id, str)
                        or not model_id
                        or model_id != candidate_id
                    ):
                        raise ValueError(
                            "Benchmark v2 Geoffrey track candidate is invalid"
                        )
                    try:
                        journal_payload = json.loads(
                            journal_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError) as error:
                        raise ValueError(
                            "Benchmark v2 designated-tracks journal is unavailable"
                        ) from error
                    ordered_keys = journal_payload.get("ordered_case_keys")
                    if not isinstance(ordered_keys, list) or not ordered_keys:
                        raise ValueError(
                            "Benchmark v2 designated-tracks journal ordering is invalid"
                        )
                    expected_provider = _qualified_provider_from_designated_tracks_journal(
                        namespace,
                        lock_dir=Path(qualification_lock_dir),
                        journal_path=journal_path,
                        benchmark_binding=namespace["_require_benchmark_binding"](
                            expected_binding.get("benchmark_binding")
                        ),
                        ordered_case_keys=ordered_keys,
                        model_id=model_id,
                    )
                    result = executor(selected_candidate, run_number)
                    if (
                        not isinstance(result, dict)
                        or result.get("served_provider") != expected_provider
                    ):
                        raise ValueError(
                            "Benchmark v2 Geoffrey served provider drifted from designated-tracks candidate"
                        )
                    return result

                guarded_kwargs = dict(kwargs)
                guarded_kwargs["geoffrey_run_executor"] = guarded_executor
                return original_final_gate_for_track(*args, **guarded_kwargs)

            namespace["execute_case"] = provider_continuous_execute_case
            namespace["run_geoffrey_final_gate_with_qualification_lock"] = (
                provider_continuous_final_gate
            )
            namespace["run_geoffrey_final_gate_with_designated_track_lock"] = (
                provider_continuous_final_gate_for_track
            )
            try:
                return original_paid_runner(
                    repository_root=repository_root,
                    provider_lock_bytes=provider_lock_bytes,
                    pricing_evidence_bytes=pricing_evidence_bytes,
                    transport=transport,
                    paid_execution_authorized=True,
                )
            finally:
                namespace["execute_case"] = previous_execute_case
                namespace["run_geoffrey_final_gate_with_qualification_lock"] = previous_final_gate
                namespace["run_geoffrey_final_gate_with_designated_track_lock"] = (
                    previous_final_gate_for_track
                )
                _ALLOW_PREFLIGHT_LOCKS.reset(preflight_token)
                _ACTIVE_REPOSITORY_ROOT.reset(root_token)

    namespace["BENCHMARK_V2_EVIDENCE_MAX_AGE"] = BENCHMARK_V2_EVIDENCE_MAX_AGE
    namespace["BENCHMARK_V2_PROVIDER_PRICING_SOURCE_API"] = (
        BENCHMARK_V2_PROVIDER_PRICING_SOURCE_API
    )
    namespace["install_benchmark_v2_provider_pricing_capture"] = (
        install_benchmark_v2_provider_pricing_capture
    )
    namespace["reconcile_benchmark_v2_aborted_execution_epoch"] = (
        reconcile_benchmark_v2_aborted_execution_epoch
    )
    namespace["validate_benchmark_v2_provider_pricing_snapshot"] = (
        validate_benchmark_v2_provider_pricing_snapshot
    )
    namespace["validate_benchmark_v2_request_parameter_compatibility"] = (
        validate_benchmark_v2_request_parameter_compatibility
    )
    namespace["require_canonical_benchmark_v2_lock_dir"] = (
        require_canonical_benchmark_v2_lock_dir
    )
    namespace["require_canonical_benchmark_v2_geoffrey_track_lock_dir"] = (
        require_canonical_benchmark_v2_geoffrey_track_lock_dir
    )
    namespace["write_qualification_decision_lock"] = guarded_qualification_writer
    namespace["write_geoffrey_final_decision_lock"] = guarded_geoffrey_writer
    namespace["write_qualification_designated_tracks_lock"] = (
        guarded_qualification_designated_tracks_writer
    )
    namespace["write_geoffrey_final_decision_lock_for_track"] = (
        guarded_geoffrey_track_writer
    )
    namespace["validate_served_provider"] = hardened_validate_served_provider
    namespace["execute_case"] = hardened_execute_case
    namespace["build_benchmark_v2_run_bundle"] = hardened_build_bundle
    namespace["write_benchmark_v2_preflight_artifact"] = write_benchmark_v2_preflight_artifact
    namespace["run_benchmark_v2_paid_authorization_preflight"] = hardened_preflight
    namespace["authorize_benchmark_v2_paid_execution"] = authorize_benchmark_v2_paid_execution
    namespace["validate_benchmark_v2_paid_authorization"] = validate_benchmark_v2_paid_authorization
    namespace["run_benchmark_v2_paid_execution"] = run_benchmark_v2_paid_execution
