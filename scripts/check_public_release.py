#!/usr/bin/env python3
"""Fail-closed, standard-library-only safety gate for this public extraction.

Run on the *tracked* public checkout. It is deliberately stricter about
production identities and private tool/evidence paths than a generic
secret scanner; it does not claim to detect every possible secret.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

PRIVATE_PREFIXES = (
    ".clinerules/", "tools/cline-", "docs/superpowers/",
    "docs/knowledge-eval/", "config/knowledge-eval/",
    "tests/fixtures/semantic-auditor/",
    "scripts/cline_", "scripts/semantic_auditor",
)
PRIVATE_EXACT = frozenset({
    "AGENTS.md", "docs/do-not-forget-ledger.md",
    "docs/compiler-review-evidence.md", "docs/self-hosted-ci-runbook.md",
    "config/podcasts.private.json", ".env", ".env.production",
})
RULES = (
    ("OpenRouter credential", re.compile(r"sk-or-v1-[A-Za-z0-9_-]{12,}")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{30,}")),
    ("GitHub token", re.compile(r"gh(?:p|o|u|s|r)_[A-Za-z0-9]{24,}")),
    ("PEM private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("production GCS bucket", re.compile(r"podcast-worker-data-[0-9]+")),
    ("production GCP project", re.compile(r"podcast-ai-[0-9]+")),
    ("production review host", re.compile(r"review[.]podcastops[.]link", re.I)),
    ("real macOS user path", re.compile(r"/Users/(?!YOUR_USER(?:/|\\b))[A-Za-z0-9._-]+/")),
    ("personal runner identity", re.compile("Petars" + "-MacBook" + "-Air", re.I)),
    ("personal email", re.compile(r"[A-Za-z0-9._%+-]+@gmail[.]com", re.I)),
)
BINARY_SUFFIXES = {".png", ".gif", ".jpg", ".jpeg", ".webp", ".pdf", ".zip", ".ico"}


def path_problem(name: str) -> str | None:
    if name in PRIVATE_EXACT or name.startswith(PRIVATE_PREFIXES):
        return "private or internal workflow artifact"
    if name.endswith((".pem", ".p12", ".pfx")) or (
        name.endswith(".key") and not name.endswith(".example.key")
    ):
        return "credential-like tracked filename"
    return None


def text_problems(value: str) -> list[str]:
    return [label for label, pattern in RULES if pattern.search(value)]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    findings: list[str] = []
    paths = [raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]
    for name in paths:
        if problem := path_problem(name):
            findings.append(f"{name}: {problem}")
            continue
        if Path(name).suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            raw = (root / name).read_bytes()
            if b"\0" in raw:
                continue
            content = raw.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            findings.append(f"{name}: cannot verify text ({type(exc).__name__})")
            continue
        for problem in text_problems(content):
            findings.append(f"{name}: {problem}")
    if findings:
        for line in findings:
            print(f"PUBLICATION CHECK RED: {line}", file=sys.stderr)
        return 1
    print(f"PUBLICATION CHECK GREEN: {len(paths)} tracked paths checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
