#!/usr/bin/env python3
"""Canonical paid-operator CLI for Summary Review Benchmark v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from podcast_engine.knowledge import summary_review_benchmark_v2 as benchmark


WORKSPACE = ROOT / "var" / "benchmarks" / "summary-review-v2"
PROVIDER_LOCK = WORKSPACE / "provider-lock.json"
PRICING_EVIDENCE = WORKSPACE / "pricing-evidence.json"
PAID_AUTHORIZATION = WORKSPACE / "paid-authorization.json"
PROVIDER_PRICING_CAPTURE = WORKSPACE / "provider-pricing-capture.json"


def _read_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing canonical Benchmark v2 {label}: {path}")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"Could not read canonical Benchmark v2 {label}: {path}") from error
    if not data:
        raise RuntimeError(f"Canonical Benchmark v2 {label} is empty: {path}")
    return data


def _evidence_bytes() -> tuple[bytes, bytes]:
    return (
        _read_bytes(PROVIDER_LOCK, label="provider lock"),
        _read_bytes(PRICING_EVIDENCE, label="pricing evidence"),
    )


def _json_safe(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _emit(value: object) -> None:
    print(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True))


def _require_dedicated_key() -> str:
    value = os.getenv("PODCAST_SUMMARY_REVIEW_API_KEY")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "Missing PODCAST_SUMMARY_REVIEW_API_KEY for paid Benchmark v2 run"
        )
    return value.strip()


def _build_transport(api_key: str):
    # Import only after canonical paid authorization has already been validated.
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Metadata": "enabled",
    }

    def transport(payload: dict) -> dict:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError("OpenRouter Benchmark v2 response must be a JSON object")
        return value

    return transport


def preflight() -> dict:
    provider_bytes, pricing_bytes = _evidence_bytes()
    return benchmark.run_benchmark_v2_paid_authorization_preflight(
        repository_root=ROOT,
        provider_lock_bytes=provider_bytes,
        pricing_evidence_bytes=pricing_bytes,
    )


def authorize() -> dict:
    provider_bytes, pricing_bytes = _evidence_bytes()
    return benchmark.authorize_benchmark_v2_paid_execution(
        repository_root=ROOT,
        provider_lock_bytes=provider_bytes,
        pricing_evidence_bytes=pricing_bytes,
    )


def refresh() -> dict:
    capture_bytes = _read_bytes(
        PROVIDER_PRICING_CAPTURE,
        label="provider/pricing capture",
    )
    result = benchmark.install_benchmark_v2_provider_pricing_capture(
        repository_root=ROOT,
        capture_bytes=capture_bytes,
    )
    return {
        key: value
        for key, value in result.items()
        if key not in {"provider_lock_bytes", "pricing_evidence_bytes"}
    }


def reconcile(evidence_path: Path) -> dict:
    evidence_bytes = _read_bytes(
        Path(evidence_path),
        label="reconciliation evidence",
    )
    return benchmark.reconcile_benchmark_v2_aborted_execution_epoch(
        repository_root=ROOT,
        evidence_bytes=evidence_bytes,
    )


def retire() -> dict:
    return benchmark.retire_benchmark_v2_partial_spend_epoch(
        repository_root=ROOT,
    )


def archive_decided_epoch() -> dict:
    return benchmark.archive_benchmark_v2_decided_epoch(
        repository_root=ROOT,
    )


def compare_geoffrey_candidates() -> dict:
    return benchmark.compare_benchmark_v2_geoffrey_candidates(
        repository_root=ROOT,
    )


def carry_forward() -> dict:
    return benchmark.carry_forward_benchmark_v2_completed_cases(
        repository_root=ROOT,
    )


def run() -> dict:
    provider_bytes, pricing_bytes = _evidence_bytes()
    authorization_bytes = _read_bytes(
        PAID_AUTHORIZATION,
        label="paid authorization",
    )

    # Fail closed before reading credentials or constructing network transport.
    benchmark.validate_benchmark_v2_paid_authorization(
        repository_root=ROOT,
        provider_lock_bytes=provider_bytes,
        pricing_evidence_bytes=pricing_bytes,
        authorization_bytes=authorization_bytes,
    )
    api_key = _require_dedicated_key()
    transport = _build_transport(api_key)
    return benchmark.run_benchmark_v2_paid_execution(
        repository_root=ROOT,
        provider_lock_bytes=provider_bytes,
        pricing_evidence_bytes=pricing_bytes,
        authorization_bytes=authorization_bytes,
        transport=transport,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical Summary Review Benchmark v2 paid-operator workflow",
    )
    parser.add_argument(
        "command",
        choices=(
            "refresh",
            "preflight",
            "authorize",
            "reconcile",
            "retire",
            "archive-decided-epoch",
            "carry-forward",
            "compare-geoffrey-candidates",
            "run",
        ),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        help="Canonical aborted-execution reconciliation evidence JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "reconcile" and args.evidence is None:
        parser.error("reconcile requires --evidence")
    if args.command != "reconcile" and args.evidence is not None:
        parser.error("--evidence is only valid with reconcile")
    try:
        if args.command == "refresh":
            result = refresh()
        elif args.command == "preflight":
            result = preflight()
        elif args.command == "authorize":
            result = authorize()
        elif args.command == "reconcile":
            result = reconcile(args.evidence)
        elif args.command == "retire":
            result = retire()
        elif args.command == "archive-decided-epoch":
            result = archive_decided_epoch()
        elif args.command == "carry-forward":
            result = carry_forward()
        elif args.command == "compare-geoffrey-candidates":
            result = compare_geoffrey_candidates()
        else:
            result = run()
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
