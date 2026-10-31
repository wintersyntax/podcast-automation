#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.review_policy import evaluate_policy_cases, load_golden_cases


FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "human_review"
    / "resolver-golden-v1.json"
)


def main() -> int:
    cases = load_golden_cases(FIXTURE)
    payload = {
        "current": evaluate_policy_cases(cases, candidate=False),
        "candidate": evaluate_policy_cases(cases, candidate=True),
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
