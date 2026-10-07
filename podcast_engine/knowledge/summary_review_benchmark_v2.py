"""Canonical Summary Review Benchmark v2 API with paid-operator safety hardening."""

from __future__ import annotations

from functools import wraps as _wraps
from pathlib import Path as _Path

_CANONICAL_REPOSITORY_ROOT = _Path(__file__).resolve().parents[2]
_IMPL_PATH = _Path(__file__).with_name("summary_review_benchmark_v2.impl")
try:
    _SOURCE = _IMPL_PATH.read_text(encoding="utf-8")
except OSError as _error:
    raise RuntimeError("Benchmark v2 implementation snapshot is unavailable") from _error

exec(compile(_SOURCE, str(_IMPL_PATH), "exec"), globals(), globals())

from .summary_review_benchmark_v2_schema_compat import (
    install_benchmark_v2_schema_compat as _install_benchmark_v2_schema_compat,
)

_install_benchmark_v2_schema_compat(globals())

from .summary_review_benchmark_v2_safety import (
    install_paid_operator_safety as _install_paid_operator_safety,
)

_install_paid_operator_safety(globals())

from .summary_review_benchmark_v2_epoch_retirement import (
    install_partial_epoch_retirement as _install_partial_epoch_retirement,
)

_install_partial_epoch_retirement(globals())

from .summary_review_benchmark_v2_carry_forward import (
    install_execution_journal_carry_forward as _install_execution_journal_carry_forward,
)

_install_execution_journal_carry_forward(globals())

from .summary_review_benchmark_v2_decided_epoch_archival import (
    install_decided_epoch_archival as _install_decided_epoch_archival,
)

_install_decided_epoch_archival(globals())

from .summary_review_benchmark_v2_geoffrey_comparison import (
    install_geoffrey_candidate_comparison as _install_geoffrey_candidate_comparison,
)

_install_geoffrey_candidate_comparison(globals())

# Unit fixtures may temporarily patch the public REPOSITORY_ROOT so the lock
# writer can target a canonical-shaped temp workspace.  Keep committed config
# authority independent from that patch: default config reads always come from
# the repository that contains this module, while explicit root= continues to
# support the canonical run-bundle copy/equality checks.
_load_canonical_config = load_canonical_benchmark_v2_config


@_wraps(_load_canonical_config)
def load_canonical_benchmark_v2_config(*, root=None):
    return _load_canonical_config(
        root=_CANONICAL_REPOSITORY_ROOT if root is None else root
    )


def _load_canonical_qualification_fixture_set():
    fixture_path = (
        _CANONICAL_REPOSITORY_ROOT
        / "config"
        / "summary-review-benchmark-v2-fixtures.json"
    )
    try:
        fixture_set = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Benchmark v2 canonical qualification fixtures are unavailable"
        ) from error
    fixtures = fixture_set.get("cases") if isinstance(fixture_set, dict) else None
    if not isinstance(fixtures, list) or len(fixtures) != 6:
        raise ValueError(
            "Benchmark v2 canonical qualification fixtures must contain exactly six cases"
        )
    case_ids = []
    for fixture in fixtures:
        case_id = fixture.get("case_id") if isinstance(fixture, dict) else None
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in case_ids
        ):
            raise ValueError(
                "Benchmark v2 canonical qualification case IDs must be unique non-empty strings"
            )
        case_ids.append(case_id)
    return fixture_set


_validate_candidate_results = _validate_qualification_candidate_results


@_wraps(_validate_candidate_results)
def _validate_qualification_candidate_results(*args, **kwargs):
    # This validator owns only committed Task-6 authority reads.  A few legacy
    # lock-writer tests patch the public repository root to a temp workspace;
    # never let that alter the frozen qualification fixture authority.
    patched_root = globals()["REPOSITORY_ROOT"]
    globals()["REPOSITORY_ROOT"] = _CANONICAL_REPOSITORY_ROOT
    try:
        return _validate_candidate_results(*args, **kwargs)
    finally:
        globals()["REPOSITORY_ROOT"] = patched_root


del (
    _Path,
    _IMPL_PATH,
    _SOURCE,
    _install_benchmark_v2_schema_compat,
    _install_paid_operator_safety,
    _install_partial_epoch_retirement,
    _install_execution_journal_carry_forward,
    _install_decided_epoch_archival,
    _install_geoffrey_candidate_comparison,
    _wraps,
)
