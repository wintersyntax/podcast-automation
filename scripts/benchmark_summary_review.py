#!/usr/bin/env python3
"""Run the evaluation-only TASK-051 blind summary reviewer benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from podcast_engine.knowledge.summary_review_benchmark import (
    ReviewContractError,
    build_blind_plan,
    execute_review,
    execute_review_preflight,
    failed_machine_gate,
    load_verified_fixture,
    machine_gate,
)
from podcast_engine.knowledge.summary_review_contract_v1 import (
    accepted_final_markdown,
    validate_review_result,
)


CONFIG_PATH = ROOT / "config" / "summary-review-benchmark-v1.json"
BENCHMARK_ROOT = ROOT / "var" / "benchmarks" / "summary-review-v1"
FIXTURE_DIR = BENCHMARK_ROOT / "fixture"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _require_dedicated_key() -> None:
    if not os.getenv("PODCAST_SUMMARY_REVIEW_API_KEY"):
        raise RuntimeError(
            "Missing PODCAST_SUMMARY_REVIEW_API_KEY for summary review benchmark"
        )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _create_unique_dir(
    root: Path,
    *,
    run_timestamp: str | None,
    run_suffix: str | None,
) -> Path:
    timestamp = run_timestamp or _timestamp()
    suffix = run_suffix or secrets.token_hex(4)
    path = root / f"{timestamp}-{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _score_template(run_ids: list[str]) -> dict:
    return {
        run_id: {
            "fidelity": None,
            "coverage": None,
            "non_regression": None,
            "structure_redundancy": None,
            "readability_density": None,
            "material_unsupported_claim": None,
            "transcript_authority_violation": None,
            "coverage_acceptable": None,
            "notes": "",
        }
        for run_id in sorted(run_ids)
    }


def _failure_gate(error: Exception) -> dict:
    if isinstance(error, (RuntimeError, requests.RequestException)):
        return failed_machine_gate("request_failed", request_succeeded=False)
    if isinstance(error, (json.JSONDecodeError, ValueError, KeyError, TypeError)):
        return failed_machine_gate("contract_failed", request_succeeded=True)
    raise error


def run_benchmark(
    *,
    config: dict,
    system_prompt: str,
    fixture_dir: Path,
    benchmark_root: Path,
    config_sha256: str,
    prompt_sha256: str,
    execute: Callable = execute_review,
    rng=None,
    run_timestamp: str | None = None,
    run_suffix: str | None = None,
) -> dict:
    """Execute all anonymous runs and persist strictly separated blind/private data."""

    _require_dedicated_key()
    fixture = load_verified_fixture(fixture_dir, config)
    rng = rng or secrets.SystemRandom()
    plan = build_blind_plan(config["models"], config["runs_per_model"], rng)

    benchmark_root = Path(benchmark_root)
    run_dir = _create_unique_dir(
        benchmark_root / "runs",
        run_timestamp=run_timestamp,
        run_suffix=run_suffix,
    )
    blind_dir = run_dir / "blind"
    private_dir = run_dir / "private"
    blind_dir.mkdir()
    private_dir.mkdir()

    relative_run = run_dir.relative_to(benchmark_root)
    benchmark_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / "latest-run.txt").write_text(
        relative_run.as_posix() + "\n",
        encoding="utf-8",
    )

    _write_json(private_dir / "model-map.json", plan["model_map"])
    run_ids = sorted(call["run_id"] for call in plan["calls"])
    _write_json(
        blind_dir / "manifest.json",
        {
            "benchmark_id": config["benchmark_id"],
            "episode_key": config["episode_key"],
            "compiled_transcript_sha256": fixture["compiled_transcript_sha256"],
            "draft_sha256": fixture["draft_sha256"],
            "reviewer_prompt_sha256": prompt_sha256,
            "benchmark_config_sha256": config_sha256,
            "anonymous_labels": sorted(plan["model_map"]),
            "run_ids": run_ids,
            "run_parameters": {
                "runs_per_model": config["runs_per_model"],
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
            },
            "execution_started_at": run_timestamp or _timestamp(),
        },
    )

    draft_summary = fixture["frozen_inputs"]["draft_summary"]
    machine_pass_count = 0
    for call in plan["calls"]:
        run_id = call["run_id"]
        model_id = call["model_id"]
        blind_run = blind_dir / run_id
        private_run = private_dir / run_id
        blind_run.mkdir()
        private_run.mkdir()

        try:
            result = execute(
                model_id,
                system_prompt,
                fixture["frozen_inputs"],
                config,
            )
            review_result = result["review_result"]
            gate = machine_gate(review_result, draft_summary)
            _write_json(blind_run / "review.json", review_result)
            final_markdown = accepted_final_markdown(review_result, draft_summary)
            if isinstance(final_markdown, str):
                (blind_run / "final.md").write_text(final_markdown, encoding="utf-8")
            provenance = {
                "requested_model": model_id,
                **dict(result.get("completion_metadata") or {}),
                "elapsed_ms": result.get("elapsed_ms"),
            }
            _write_json(private_run / "provenance.json", provenance)
        except ReviewContractError as error:
            gate = failed_machine_gate("contract_failed", request_succeeded=True)
            gate["validation_code"] = error.code
            gate["validation_message"] = str(error)
            _write_json(
                blind_run / "review.json",
                error.review_result
                if isinstance(error.review_result, dict)
                else {"review_result": None, "failure_code": "contract_failed"},
            )
            _write_json(
                private_run / "provenance.json",
                {
                    "requested_model": model_id,
                    **dict(error.completion_metadata or {}),
                    "elapsed_ms": error.elapsed_ms,
                    "error_class": type(error).__name__,
                },
            )
        except (
            RuntimeError,
            requests.RequestException,
            json.JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
        ) as error:
            gate = _failure_gate(error)
            _write_json(
                blind_run / "review.json",
                {"review_result": None, "failure_code": gate["failure_code"]},
            )
            _write_json(
                private_run / "provenance.json",
                {
                    "requested_model": model_id,
                    "error_class": type(error).__name__,
                },
            )

        gate = {"run": run_id, **gate}
        _write_json(blind_run / "machine-gate.json", gate)
        if gate["machine_pass"]:
            machine_pass_count += 1

    _write_json(blind_dir / "scores.input.json", _score_template(run_ids))
    return {
        "run_dir": run_dir,
        "completed_count": len(plan["calls"]),
        "machine_pass_count": machine_pass_count,
        "blind_dir": blind_dir,
    }


def preflight_models(
    *,
    config: dict,
    system_prompt: str,
    benchmark_root: Path,
    execute: Callable = execute_review_preflight,
    run_timestamp: str | None = None,
    run_suffix: str | None = None,
) -> dict:
    """Verify every approved exact model can satisfy the structured review route."""

    _require_dedicated_key()
    preflight_dir = _create_unique_dir(
        Path(benchmark_root) / "preflight",
        run_timestamp=run_timestamp,
        run_suffix=run_suffix,
    )
    private_dir = preflight_dir / "private" / "preflight"
    private_dir.mkdir(parents=True)

    synthetic_inputs = {
        "episode": {"episode_key": "preflight", "title": "Structured-output preflight"},
        "podcast_profile": {"category": "preflight", "profile": "preflight"},
        "compiled_transcript": "The speaker said the plan might help.",
        "draft_summary": (
            "## TL;DR\n\n- The speaker said the plan definitely works.\n\n"
            "## Key Ideas\n\n### Expected outcome\n\n"
            "The speaker said the plan definitely works.\n"
        ),
    }

    passed = 0
    for index, model_id in enumerate(config["models"], start=1):
        path = private_dir / f"model-{index}.json"
        try:
            result = execute(model_id, system_prompt, synthetic_inputs, config)
            review_result = result["review_result"]
            validate_review_result(
                review_result,
                synthetic_inputs["draft_summary"],
                synthetic_inputs["compiled_transcript"],
            )
            if review_result.get("status") != "revised":
                raise ValueError("Summary review preflight requires revised status")
            final = accepted_final_markdown(
                review_result,
                synthetic_inputs["draft_summary"],
            )
            if "definitely works" in final.casefold():
                raise ValueError("Summary review preflight failed to correct the seeded claim")
            _write_json(
                path,
                {
                    "requested_model": model_id,
                    **dict(result.get("completion_metadata") or {}),
                    "elapsed_ms": result.get("elapsed_ms"),
                },
            )
            passed += 1
        except Exception as error:
            _write_json(
                path,
                {
                    "requested_model": model_id,
                    "error_class": type(error).__name__,
                },
            )
            raise RuntimeError(
                f"Summary reviewer preflight failed for approved model {model_id}"
            ) from error

    return {
        "preflight_dir": preflight_dir,
        "passed_count": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TASK-051 summary reviewer Benchmark v1.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    prompt_path = ROOT / config["prompt_path"]
    prompt_bytes = prompt_path.read_bytes()
    system_prompt = prompt_bytes.decode("utf-8")

    if args.preflight:
        result = preflight_models(
            config=config,
            system_prompt=system_prompt,
            benchmark_root=BENCHMARK_ROOT,
        )
        print(f"benchmark_id: {config['benchmark_id']}")
        print(f"preflight_passed: {result['passed_count']}")
        return

    result = run_benchmark(
        config=config,
        system_prompt=system_prompt,
        fixture_dir=FIXTURE_DIR,
        benchmark_root=BENCHMARK_ROOT,
        config_sha256=_sha256(config_bytes),
        prompt_sha256=_sha256(prompt_bytes),
    )
    print(f"benchmark_id: {config['benchmark_id']}")
    print(f"completed_runs: {result['completed_count']}")
    print(f"machine_pass_runs: {result['machine_pass_count']}")
    print(f"blind_directory: {result['blind_dir']}")
    print("Do not inspect private benchmark identity or cost data before score lock.")


if __name__ == "__main__":
    main()
