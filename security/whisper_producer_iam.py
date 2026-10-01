"""Staged Cloud Storage access contract for Whisper producer identity.

This module is deliberately local-only.  It models the named-object access
that a later reviewed IAM policy must grant; it neither calls Google Cloud nor
changes runtime application behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal


PROJECT_ID = "YOUR_GCP_PROJECT_ID"
BUCKET_NAME = "YOUR_GCS_BUCKET"
OBJECT_RESOURCE_PREFIX = f"projects/_/buckets/{BUCKET_NAME}/objects/"

WORKER_PRINCIPAL = (
    "podcast-worker-runtime@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com"
)
APPLE_INGEST_PRINCIPAL = (
    "podcast-apple-ingest-runtime@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com"
)
REVIEW_PRINCIPAL = (
    "podcast-review-runtime@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com"
)
PRINCIPALS = (
    WORKER_PRINCIPAL,
    APPLE_INGEST_PRINCIPAL,
    REVIEW_PRINCIPAL,
)

READER_PERMISSIONS = frozenset({"storage.objects.get"})
WRITER_PERMISSIONS = frozenset(
    {
        "storage.objects.get",
        "storage.objects.create",
        "storage.objects.delete",
    }
)
FORBIDDEN_PERMISSIONS = frozenset(
    {
        "storage.objects.list",
        "storage.objects.update",
        "storage.objects.move",
        "storage.objects.restore",
    }
)

READER_ROLE = f"projects/{PROJECT_ID}/roles/podcastObjectReader"
WRITER_ROLE = f"projects/{PROJECT_ID}/roles/podcastObjectReadWrite"
MAX_CEL_LOGICAL_OPERATORS = 10

_EPISODE_KEY = re.compile(r"^[0-9a-f]{24}$")
_REVIEW_ID = re.compile(r"^sr-[0-9a-f]{24}$")
_STALLED_EMAIL_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_STALLED_EMAIL_PATTERN = (
    "/summary/review_notifications/stalled_email/<digest>.json"
)
_AI_BUDGET_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_AI_BUDGET_PATTERN = "/ai/budgets/<digest>.json"
_BUDGET_IDENTITY_RECONCILIATION_PATTERN = (
    "/ai/budget-identity-reconciliations/<digest>.json"
)
_PREPARE_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
_PREPARE_SESSION_PATTERN = "/ai/prepare-sessions/<digest>/<session_id>.json"
_THIRD_ASR_CLAIM_DIFFERENCE_ID = re.compile(r"^(0|[1-9][0-9]*)$")
_THIRD_ASR_CLAIM_PATTERN = "/ai/third-asr-claims/<digest>/<difference_id>.json"
AccessKind = Literal["reader", "writer"]
PatternKind = Literal[
    "exact",
    "episode",
    "incoming_apple",
    "summary_review",
    "stalled_email",
    "ai_budget",
    "budget_identity_reconciliation",
    "prepare_session",
    "third_asr_claim",
]


@dataclass(frozen=True)
class ObjectPattern:
    """One allowed exact object or canonical dynamic episode object pattern."""

    kind: PatternKind
    value: str

    def matches(self, object_name: str) -> bool:
        """Return whether a canonical, non-malformed object name matches."""

        if not isinstance(object_name, str):
            return False
        if self.kind == "exact":
            return object_name == self.value
        if self.kind == "episode":
            return _matches_keyed_path(object_name, "episodes/", self.value)
        if self.kind == "incoming_apple":
            return _matches_keyed_path(object_name, "incoming/apple/", self.value)
        if self.kind == "summary_review":
            return _matches_summary_review_artifact(object_name, self.value)
        if self.kind == "stalled_email":
            return _matches_stalled_email_claim(object_name)
        if self.kind == "ai_budget":
            return _matches_ai_budget_ledger(object_name)
        if self.kind == "budget_identity_reconciliation":
            return _matches_budget_identity_reconciliation(object_name)
        if self.kind == "prepare_session":
            return _matches_prepare_session(object_name)
        if self.kind == "third_asr_claim":
            return _matches_third_asr_claim(object_name)
        raise ValueError(f"Unsupported object pattern kind: {self.kind}")

    def cel_clause(self) -> str:
        """Render this pattern with the same limited CEL primitives we stage."""

        if self.kind == "exact":
            return f"resource.name == '{_resource_name(self.value)}'"
        if self.kind == "episode":
            return _prefix_suffix_clause("episodes/", self.value)
        if self.kind == "incoming_apple":
            return _prefix_suffix_clause("incoming/apple/", self.value)
        if self.kind == "summary_review":
            return _summary_review_artifact_clause(self.value)
        if self.kind == "stalled_email":
            return _stalled_email_claim_clause()
        if self.kind == "ai_budget":
            return _ai_budget_ledger_clause()
        if self.kind == "budget_identity_reconciliation":
            return _budget_identity_reconciliation_clause()
        if self.kind == "prepare_session":
            return _prepare_session_clause()
        if self.kind == "third_asr_claim":
            return _third_asr_claim_clause()
        raise ValueError(f"Unsupported object pattern kind: {self.kind}")

    def label(self) -> str:
        if self.kind == "exact":
            return self.value
        if self.kind == "episode":
            return f"episodes/<episode_key>{self.value}"
        if self.kind == "incoming_apple":
            return f"incoming/apple/<episode_key>{self.value}"
        if self.kind == "summary_review":
            return (
                "episodes/<episode_key>/summary/reviews/"
                f"<review_id>/{self.value}"
            )
        if self.kind == "stalled_email":
            return f"episodes/<episode_key>{self.value}"
        if self.kind == "ai_budget":
            return f"episodes/<episode_key>{self.value}"
        if self.kind == "budget_identity_reconciliation":
            return f"episodes/<episode_key>{self.value}"
        if self.kind == "prepare_session":
            return f"episodes/<episode_key>{self.value}"
        if self.kind == "third_asr_claim":
            return f"episodes/<episode_key>{self.value}"
        raise ValueError(f"Unsupported object pattern kind: {self.kind}")


@dataclass(frozen=True)
class AccessBinding:
    """One conceptual conditional grant in the local security policy."""

    principal: str
    kind: AccessKind
    permissions: frozenset[str]
    patterns: tuple[ObjectPattern, ...]

    @property
    def role(self) -> str:
        return _role_for(self.kind)

    def allows(self, permission: str, object_name: str) -> bool:
        return permission in self.permissions and any(
            pattern.matches(object_name) for pattern in self.patterns
        )

    def cel_expression(self) -> str:
        return _cel_expression_for(self.patterns)


@dataclass(frozen=True)
class PhysicalIamBinding:
    """One Google IAM-ready conditional binding emitted from a conceptual grant."""

    principal: str
    kind: AccessKind
    permissions: frozenset[str]
    patterns: tuple[ObjectPattern, ...]

    @property
    def role(self) -> str:
        return _role_for(self.kind)

    def allows(self, permission: str, object_name: str) -> bool:
        return permission in self.permissions and any(
            pattern.matches(object_name) for pattern in self.patterns
        )

    def cel_expression(self) -> str:
        return _cel_expression_for(self.patterns)


def _resource_name(object_name: str) -> str:
    return f"{OBJECT_RESOURCE_PREFIX}{object_name}"


def _role_for(kind: AccessKind) -> str:
    return READER_ROLE if kind == "reader" else WRITER_ROLE


def _cel_expression_for(patterns: Iterable[ObjectPattern]) -> str:
    clauses = tuple(pattern.cel_clause() for pattern in patterns)
    if not clauses:
        raise ValueError("IAM bindings must never be unconditional")
    joined = " ||\n  ".join(clauses)
    return (
        "resource.type == 'storage.googleapis.com/Object' && (\n"
        f"  {joined}\n"
        ")"
    )


def logical_operator_count(cel_expression: str) -> int:
    """Count CEL ``&&`` and ``||`` operators in an emitted condition."""

    return len(re.findall(r"&&|\|\|", cel_expression))


def _matches_keyed_path(object_name: str, prefix: str, suffix: str) -> bool:
    if not object_name.startswith(prefix) or not object_name.endswith(suffix):
        return False
    episode_key = object_name[len(prefix) : len(object_name) - len(suffix)]
    return bool(_EPISODE_KEY.fullmatch(episode_key))


def _prefix_suffix_clause(prefix: str, suffix: str) -> str:
    return (
        "(resource.name.startsWith("
        f"'{_resource_name(prefix)}') && "
        f"resource.name.endsWith('{suffix}'))"
    )


def _matches_summary_review_artifact(
    object_name: str,
    filename: str,
) -> bool:
    prefix = "episodes/"
    marker = "/summary/reviews/"
    suffix = f"/{filename}"

    if not object_name.startswith(prefix) or not object_name.endswith(suffix):
        return False

    middle = object_name[len(prefix) : -len(suffix)]
    parts = middle.split(marker)
    if len(parts) != 2:
        return False

    episode_key, review_id = parts
    return bool(
        _EPISODE_KEY.fullmatch(episode_key)
        and _REVIEW_ID.fullmatch(review_id)
    )


def _matches_stalled_email_claim(object_name: str) -> bool:
    prefix = "episodes/"
    marker = "/summary/review_notifications/stalled_email/"
    suffix = ".json"

    if not object_name.startswith(prefix) or not object_name.endswith(suffix):
        return False

    middle = object_name[len(prefix) : -len(suffix)]
    parts = middle.split(marker)
    if len(parts) != 2:
        return False

    episode_key, digest = parts
    return bool(
        _EPISODE_KEY.fullmatch(episode_key)
        and _STALLED_EMAIL_DIGEST.fullmatch(digest)
    )


def _matches_ai_budget_ledger(object_name: str) -> bool:
    prefix = "episodes/"
    marker = "/ai/budgets/"
    suffix = ".json"

    if not object_name.startswith(prefix) or not object_name.endswith(suffix):
        return False

    middle = object_name[len(prefix) : -len(suffix)]
    parts = middle.split(marker)
    if len(parts) != 2:
        return False

    episode_key, digest = parts
    return bool(
        _EPISODE_KEY.fullmatch(episode_key)
        and _AI_BUDGET_DIGEST.fullmatch(digest)
    )


def _matches_budget_identity_reconciliation(object_name: str) -> bool:
    prefix = "episodes/"
    marker = "/ai/budget-identity-reconciliations/"
    suffix = ".json"

    if not object_name.startswith(prefix) or not object_name.endswith(suffix):
        return False

    middle = object_name[len(prefix) : -len(suffix)]
    parts = middle.split(marker)
    if len(parts) != 2:
        return False

    episode_key, digest = parts
    return bool(
        _EPISODE_KEY.fullmatch(episode_key)
        and _AI_BUDGET_DIGEST.fullmatch(digest)
    )


def _budget_identity_reconciliation_clause() -> str:
    template = "/ai/budget-identity-reconciliations/{digest}.json"
    return (
        "(resource.name.startsWith("
        f"'{_resource_name('episodes/')}') && "
        "resource.name.endsWith('.json') && "
        f"resource.name.extract('{template}') != '')"
    )


def _matches_prepare_session(object_name: str) -> bool:
    prefix = "episodes/"
    marker = "/ai/prepare-sessions/"
    suffix = ".json"

    if not object_name.startswith(prefix) or not object_name.endswith(suffix):
        return False

    middle = object_name[len(prefix) : -len(suffix)]
    parts = middle.split(marker)
    if len(parts) != 2:
        return False

    episode_key, digest_and_session = parts
    if not _EPISODE_KEY.fullmatch(episode_key):
        return False

    segments = digest_and_session.split("/")
    if len(segments) != 2:
        return False

    digest, session_id = segments
    return bool(
        _AI_BUDGET_DIGEST.fullmatch(digest) and _PREPARE_SESSION_ID.fullmatch(session_id)
    )


def _prepare_session_clause() -> str:
    # One extract() group spans "<digest>/<session_id>" as a single
    # capture (matching the proven ai_budget-clause shape rather than an
    # unproven multi-placeholder extract()): CEL's extract() captures
    # everything between the surrounding literal text, slashes included,
    # so a single {path} group over the whole dynamic segment is exact
    # and avoids relying on untested multi-group extract() semantics.
    template = "/ai/prepare-sessions/{path}.json"
    return (
        "(resource.name.startsWith("
        f"'{_resource_name('episodes/')}') && "
        "resource.name.endsWith('.json') && "
        f"resource.name.extract('{template}') != '')"
    )


def _matches_third_asr_claim(object_name: str) -> bool:
    prefix = "episodes/"
    marker = "/ai/third-asr-claims/"
    suffix = ".json"

    if not object_name.startswith(prefix) or not object_name.endswith(suffix):
        return False

    middle = object_name[len(prefix) : -len(suffix)]
    parts = middle.split(marker)
    if len(parts) != 2:
        return False

    episode_key, digest_and_id = parts
    if not _EPISODE_KEY.fullmatch(episode_key):
        return False

    segments = digest_and_id.split("/")
    if len(segments) != 2:
        return False

    digest, difference_id = segments
    return bool(
        _AI_BUDGET_DIGEST.fullmatch(digest)
        and _THIRD_ASR_CLAIM_DIFFERENCE_ID.fullmatch(difference_id)
    )


def _third_asr_claim_clause() -> str:
    # Same single-capture extract() shape as prepare_session/ai_budget:
    # one {path} group spans "<digest>/<difference_id>" as a whole.
    template = "/ai/third-asr-claims/{path}.json"
    return (
        "(resource.name.startsWith("
        f"'{_resource_name('episodes/')}') && "
        "resource.name.endsWith('.json') && "
        f"resource.name.extract('{template}') != '')"
    )


def _summary_review_artifact_clause(filename: str) -> str:
    template = f"/summary/reviews/sr-{{review_token}}/{filename}"
    return (
        "(resource.name.startsWith("
        f"'{_resource_name('episodes/')}') && "
        f"resource.name.endsWith('/{filename}') && "
        f"resource.name.extract('{template}') != '')"
    )


def _stalled_email_claim_clause() -> str:
    template = (
        "/summary/review_notifications/stalled_email/"
        "{digest}.json"
    )
    return (
        "(resource.name.startsWith("
        f"'{_resource_name('episodes/')}') && "
        "resource.name.endsWith('.json') && "
        f"resource.name.extract('{template}') != '')"
    )


def _ai_budget_ledger_clause() -> str:
    template = "/ai/budgets/{digest}.json"
    return (
        "(resource.name.startsWith("
        f"'{_resource_name('episodes/')}') && "
        "resource.name.endsWith('.json') && "
        f"resource.name.extract('{template}') != '')"
    )


def _exact(object_name: str) -> ObjectPattern:
    return ObjectPattern("exact", object_name)


def _episode(suffix: str) -> ObjectPattern:
    return ObjectPattern("episode", suffix)


def _incoming_apple(filename: str) -> ObjectPattern:
    return ObjectPattern("incoming_apple", f"/{filename}")


def _summary_review(filename: str) -> ObjectPattern:
    return ObjectPattern("summary_review", filename)


def _stalled_email_claim() -> ObjectPattern:
    return ObjectPattern("stalled_email", _STALLED_EMAIL_PATTERN)


def _ai_budget_ledger() -> ObjectPattern:
    return ObjectPattern("ai_budget", _AI_BUDGET_PATTERN)


def _budget_identity_reconciliation() -> ObjectPattern:
    return ObjectPattern(
        "budget_identity_reconciliation",
        _BUDGET_IDENTITY_RECONCILIATION_PATTERN,
    )


def _prepare_session() -> ObjectPattern:
    return ObjectPattern("prepare_session", _PREPARE_SESSION_PATTERN)


def _third_asr_claim() -> ObjectPattern:
    return ObjectPattern("third_asr_claim", _THIRD_ASR_CLAIM_PATTERN)


EPISODES_INDEX = _exact("episodes.json")
APPLE_STAGING_JSON = _incoming_apple("apple-transcript.json")
APPLE_STAGING_TEXT = _incoming_apple("apple-transcript.txt")
APPLE_SOURCE_TEXT = _episode("/sources/apple/transcript.txt")
APPLE_SOURCE_METADATA = _episode("/sources/apple/transcript.json")
WHISPER_SOURCE_TEXT = _episode("/sources/whisper/transcript.txt")
WHISPER_SOURCE_METADATA = _episode("/sources/whisper/transcript.json")
COMPILED_TEXT = _episode("/compiled/transcript.txt")
COMPILER_REPORT = _episode("/compiled/report.json")
RESOLVER_RECORD = _episode("/review/resolver.json")
SUMMARY_BODY = _episode("/summary/body.md")
SUMMARY_METADATA = _episode("/summary/metadata.json")
SUMMARY_FINAL = _episode("/summary/summary.md")
SUMMARY_REVIEW_HISTORY = _episode("/summary/reviews/index.json")
SUMMARY_REVIEW_DRAFT = _summary_review("draft.md")
SUMMARY_REVIEW_FINAL = _summary_review("final.md")
SUMMARY_REVIEW_RECORD = _summary_review("review.json")
SUMMARY_REVIEW_TRANSCRIPT_INDEX = _summary_review(
    "transcript-span-index.json"
)
SUMMARY_REVIEW_DRAFT_INDEX = _summary_review(
    "draft-block-index.json"
)
SUMMARY_REVIEW_RISK_INVENTORY = _summary_review(
    "risk-inventory.json"
)
SUMMARY_REVIEW_ARTIFACTS = (
    SUMMARY_REVIEW_DRAFT,
    SUMMARY_REVIEW_FINAL,
    SUMMARY_REVIEW_RECORD,
    SUMMARY_REVIEW_TRANSCRIPT_INDEX,
    SUMMARY_REVIEW_DRAFT_INDEX,
    SUMMARY_REVIEW_RISK_INVENTORY,
)
STALLED_EMAIL_CLAIM = _stalled_email_claim()
AI_BUDGET_LEDGER = _ai_budget_ledger()
BUDGET_IDENTITY_RECONCILIATION = _budget_identity_reconciliation()
PREPARE_SESSION = _prepare_session()
THIRD_ASR_CLAIM = _third_asr_claim()
TAG_REGISTRY = _exact("knowledge/tags/registry-v1.json")
ARTIFACT_INDEX = _exact("knowledge/artifacts/index-v1.json")
KNOWLEDGE_AGENT_STATUS = _exact("knowledge/agents/macos-knowledge-sync-v1.json")


_WORKER_READER = (
    EPISODES_INDEX,
    APPLE_SOURCE_TEXT,
    APPLE_SOURCE_METADATA,
    WHISPER_SOURCE_TEXT,
    WHISPER_SOURCE_METADATA,
    COMPILED_TEXT,
    RESOLVER_RECORD,
    SUMMARY_BODY,
    SUMMARY_METADATA,
    SUMMARY_FINAL,
    SUMMARY_REVIEW_HISTORY,
    *SUMMARY_REVIEW_ARTIFACTS,
    STALLED_EMAIL_CLAIM,
    AI_BUDGET_LEDGER,
    TAG_REGISTRY,
)
_WORKER_WRITER = (
    EPISODES_INDEX,
    APPLE_STAGING_JSON,
    APPLE_STAGING_TEXT,
    WHISPER_SOURCE_TEXT,
    WHISPER_SOURCE_METADATA,
    COMPILED_TEXT,
    COMPILER_REPORT,
    RESOLVER_RECORD,
    SUMMARY_BODY,
    SUMMARY_METADATA,
    SUMMARY_FINAL,
    SUMMARY_REVIEW_HISTORY,
    *SUMMARY_REVIEW_ARTIFACTS,
    STALLED_EMAIL_CLAIM,
    AI_BUDGET_LEDGER,
    TAG_REGISTRY,
)
_APPLE_INGEST_READER = (
    EPISODES_INDEX,
    APPLE_STAGING_JSON,
    APPLE_STAGING_TEXT,
    APPLE_SOURCE_TEXT,
    APPLE_SOURCE_METADATA,
)
_APPLE_INGEST_WRITER = _APPLE_INGEST_READER
_REVIEW_READER = (
    EPISODES_INDEX,
    APPLE_SOURCE_TEXT,
    APPLE_SOURCE_METADATA,
    WHISPER_SOURCE_TEXT,
    WHISPER_SOURCE_METADATA,
    RESOLVER_RECORD,
    SUMMARY_BODY,
    SUMMARY_METADATA,
    SUMMARY_FINAL,
    SUMMARY_REVIEW_HISTORY,
    SUMMARY_REVIEW_RECORD,
    AI_BUDGET_LEDGER,
    BUDGET_IDENTITY_RECONCILIATION,
    PREPARE_SESSION,
    THIRD_ASR_CLAIM,
    TAG_REGISTRY,
    ARTIFACT_INDEX,
    KNOWLEDGE_AGENT_STATUS,
)
_REVIEW_WRITER = (
    RESOLVER_RECORD,
    SUMMARY_METADATA,
    SUMMARY_FINAL,
    AI_BUDGET_LEDGER,
    PREPARE_SESSION,
    THIRD_ASR_CLAIM,
    TAG_REGISTRY,
    ARTIFACT_INDEX,
)


ACCESS_BINDINGS = (
    AccessBinding(WORKER_PRINCIPAL, "reader", READER_PERMISSIONS, _WORKER_READER),
    AccessBinding(WORKER_PRINCIPAL, "writer", WRITER_PERMISSIONS, _WORKER_WRITER),
    AccessBinding(
        APPLE_INGEST_PRINCIPAL,
        "reader",
        READER_PERMISSIONS,
        _APPLE_INGEST_READER,
    ),
    AccessBinding(
        APPLE_INGEST_PRINCIPAL,
        "writer",
        WRITER_PERMISSIONS,
        _APPLE_INGEST_WRITER,
    ),
    AccessBinding(REVIEW_PRINCIPAL, "reader", READER_PERMISSIONS, _REVIEW_READER),
    AccessBinding(REVIEW_PRINCIPAL, "writer", WRITER_PERMISSIONS, _REVIEW_WRITER),
)


def bindings_for(principal: str, kind: AccessKind | None = None) -> tuple[AccessBinding, ...]:
    """Return deterministic staged bindings for one declared principal."""

    return tuple(
        binding
        for binding in ACCESS_BINDINGS
        if binding.principal == principal and (kind is None or binding.kind == kind)
    )


def is_allowed(principal: str, permission: str, object_name: str) -> bool:
    """Evaluate only this local, declared named-object access contract."""

    return any(
        binding.allows(permission, object_name)
        for binding in bindings_for(principal)
    )


def generate_cel_conditions() -> tuple[AccessBinding, ...]:
    """Return the six inspectable conceptual bindings in this contract."""

    validate_contract()
    return ACCESS_BINDINGS


def generate_physical_iam_bindings(
    *,
    max_logical_operators: int = MAX_CEL_LOGICAL_OPERATORS,
) -> tuple[PhysicalIamBinding, ...]:
    """Split conceptual grants into lint-safe Google IAM conditional bindings.

    Patterns retain their declared sequence.  Each output binding has the same
    principal, role, and permissions as its source conceptual binding, and
    differs only by containing a consecutive subset of its object patterns.
    """

    if max_logical_operators < 1:
        raise ValueError("The CEL logical-operator limit must be at least one")

    validate_contract()

    physical_bindings: list[PhysicalIamBinding] = []
    for conceptual_binding in ACCESS_BINDINGS:
        physical_bindings.extend(
            _split_conceptual_binding(
                conceptual_binding,
                max_logical_operators=max_logical_operators,
            )
        )
    return tuple(physical_bindings)


def _split_conceptual_binding(
    conceptual_binding: AccessBinding,
    *,
    max_logical_operators: int,
) -> tuple[PhysicalIamBinding, ...]:
    """Greedily pack a binding's ordered patterns without changing its union."""

    if not conceptual_binding.patterns:
        raise ValueError("IAM bindings must never be unconditional")

    chunks: list[tuple[ObjectPattern, ...]] = []
    current_chunk: tuple[ObjectPattern, ...] = ()
    for pattern in conceptual_binding.patterns:
        clause_operator_count = logical_operator_count(pattern.cel_clause())
        if clause_operator_count > max_logical_operators:
            raise ValueError(
                "Object pattern "
                f"{pattern.label()!r} CEL clause alone has {clause_operator_count} "
                "logical operators, exceeding the "
                f"{max_logical_operators}-operator IAM limit"
            )

        candidate_chunk = (*current_chunk, pattern)
        candidate_operator_count = logical_operator_count(
            _cel_expression_for(candidate_chunk)
        )
        if candidate_operator_count <= max_logical_operators:
            current_chunk = candidate_chunk
            continue

        if not current_chunk:
            raise ValueError(
                "Object pattern "
                f"{pattern.label()!r} requires {candidate_operator_count} "
                "logical operators in its complete IAM condition, exceeding the "
                f"{max_logical_operators}-operator IAM limit"
            )
        chunks.append(current_chunk)
        current_chunk = (pattern,)
        single_pattern_operator_count = logical_operator_count(
            _cel_expression_for(current_chunk)
        )
        if single_pattern_operator_count > max_logical_operators:
            raise ValueError(
                "Object pattern "
                f"{pattern.label()!r} requires {single_pattern_operator_count} "
                "logical operators in its complete IAM condition, exceeding the "
                f"{max_logical_operators}-operator IAM limit"
            )

    chunks.append(current_chunk)
    return tuple(
        PhysicalIamBinding(
            principal=conceptual_binding.principal,
            kind=conceptual_binding.kind,
            permissions=conceptual_binding.permissions,
            patterns=chunk,
        )
        for chunk in chunks
    )


def validate_contract() -> None:
    """Reject broad, unconditional, or unsupported permission drift locally."""

    if {binding.principal for binding in ACCESS_BINDINGS} != set(PRINCIPALS):
        raise ValueError("Contract must represent exactly the three production runtimes")
    for binding in ACCESS_BINDINGS:
        expected = READER_PERMISSIONS if binding.kind == "reader" else WRITER_PERMISSIONS
        if binding.permissions != expected:
            raise ValueError(f"{binding.principal} {binding.kind} permissions drifted")
        if not binding.patterns:
            raise ValueError("Unconditional IAM bindings are forbidden")
        if binding.kind == "writer" and any(
            pattern.kind == "episode" and not pattern.value for pattern in binding.patterns
        ):
            raise ValueError("Episode writer patterns must have a fixed suffix")
        for pattern in binding.patterns:
            if (
                pattern.kind == "summary_review"
                and pattern.value
                not in {
                    "draft.md",
                    "final.md",
                    "review.json",
                    "transcript-span-index.json",
                    "draft-block-index.json",
                    "risk-inventory.json",
                }
            ):
                raise ValueError(
                    "Unsupported summary-review artifact was declared"
                )
            if (
                pattern.kind == "stalled_email"
                and pattern.value != _STALLED_EMAIL_PATTERN
            ):
                raise ValueError(
                    "Stalled-email claim pattern must stay canonical"
                )
            if (
                pattern.kind == "ai_budget"
                and pattern.value != _AI_BUDGET_PATTERN
            ):
                raise ValueError(
                    "AI budget ledger pattern must stay canonical"
                )
            if (
                pattern.kind == "budget_identity_reconciliation"
                and pattern.value != _BUDGET_IDENTITY_RECONCILIATION_PATTERN
            ):
                raise ValueError(
                    "Budget identity reconciliation pattern must stay canonical"
                )
        if binding.permissions & FORBIDDEN_PERMISSIONS:
            raise ValueError("Forbidden storage permission was declared")


__all__ = [
    "ACCESS_BINDINGS",
    "APPLE_INGEST_PRINCIPAL",
    "BUDGET_IDENTITY_RECONCILIATION",
    "BUCKET_NAME",
    "FORBIDDEN_PERMISSIONS",
    "KNOWLEDGE_AGENT_STATUS",
    "MAX_CEL_LOGICAL_OPERATORS",
    "OBJECT_RESOURCE_PREFIX",
    "PhysicalIamBinding",
    "PRINCIPALS",
    "READER_PERMISSIONS",
    "REVIEW_PRINCIPAL",
    "WORKER_PRINCIPAL",
    "WRITER_PERMISSIONS",
    "bindings_for",
    "generate_cel_conditions",
    "generate_physical_iam_bindings",
    "is_allowed",
    "logical_operator_count",
    "validate_contract",
]
