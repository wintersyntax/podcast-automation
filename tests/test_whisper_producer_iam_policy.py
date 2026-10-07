"""Local guardrails for the staged Whisper producer-identity IAM contract."""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from podcast_engine import (
    ai_budget,
    corpus_checkpoints,
    episode_contract,
    episode_generation,
    human_review,
)
from security.whisper_producer_iam import (
    ACCESS_BINDINGS,
    APPLE_INGEST_PRINCIPAL,
    FORBIDDEN_PERMISSIONS,
    MAX_CEL_LOGICAL_OPERATORS,
    OBJECT_RESOURCE_PREFIX,
    READER_PERMISSIONS,
    REVIEW_PRINCIPAL,
    WORKER_PRINCIPAL,
    WRITER_PERMISSIONS,
    bindings_for,
    generate_cel_conditions,
    generate_physical_iam_bindings,
    is_allowed,
    logical_operator_count,
    validate_contract,
)


KEY = "0123456789abcdef01234567"
ROOT = f"episodes/{KEY}"
WHISPER_TEXT = f"{ROOT}/sources/whisper/transcript.txt"
WHISPER_METADATA = f"{ROOT}/sources/whisper/transcript.json"
AI_BUDGET_LEDGER_PATH = f"{ROOT}/ai/budgets/{'e' * 64}.json"
BUDGET_IDENTITY_RECONCILIATION_PATH = (
    f"{ROOT}/ai/budget-identity-reconciliations/{'e' * 64}.json"
)
PREPARE_SESSION_PATH = f"{ROOT}/ai/prepare-sessions/{'e' * 64}/{'a' * 32}.json"
THIRD_ASR_CLAIM_PATH = f"{ROOT}/ai/third-asr-claims/{'e' * 64}/184.json"

NON_GCS_PATH_FUNCTIONS = {
    "podcast_engine/apple_acquisition.py::_token_path",
    "podcast_engine/compilation.py::_review_source_fingerprint_from_paths",
}


def _discover_path_named_functions(root: Path) -> set[str]:
    """Return top-level path-shaped functions in podcast_engine/*.py."""

    discovered: set[str] = set()
    for path in sorted((root / "podcast_engine").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not (
                node.name == "paths_for"
                or node.name.endswith("_path")
                or node.name.endswith("_paths")
            ):
                continue
            discovered.add(
                f"{path.relative_to(root).as_posix()}::{node.name}"
            )
    return discovered


def _gcs_path_function_cases():
    source_fingerprint = "sha256:" + ("e" * 64)
    review_id = f"sr-{KEY}"
    session_id = "a" * 32
    compiler_report = f"{ROOT}/compiled/report.json"

    return {
        "podcast_engine/ai_budget.py::budget_ledger_path": lambda: (
            ai_budget.budget_ledger_path(KEY, source_fingerprint),
        ),
        "podcast_engine/ai_budget.py::third_asr_budget_reconciliation_path": lambda: (
            ai_budget.third_asr_budget_reconciliation_path(
                KEY, source_fingerprint
            ),
        ),
        "podcast_engine/corpus_checkpoints.py::_report_path": lambda: (
            corpus_checkpoints._report_path(
                {"files": {"compiled": {"report": compiler_report}}}
            ),
        ),
        "podcast_engine/episode_contract.py::paths_for": lambda: tuple(
            episode_contract.paths_for(KEY).values()
        ),
        "podcast_engine/episode_contract.py::summary_review_artifact_paths": lambda: tuple(
            value
            for name, value in episode_contract.summary_review_artifact_paths(
                KEY,
                review_id,
            ).items()
            if name != "root"  # Prefix convenience, not an object path.
        ),
        "podcast_engine/episode_contract.py::incoming_apple_paths": lambda: tuple(
            episode_contract.incoming_apple_paths(KEY).values()
        ),
        "podcast_engine/episode_generation.py::_source_text_path": lambda: (
            episode_generation._source_text_path(
                {"text": f"{ROOT}/sources/apple/transcript.txt"}
            ),
            episode_generation._source_text_path({"text": WHISPER_TEXT}),
        ),
        "podcast_engine/episode_generation.py::_source_metadata_path": lambda: (
            episode_generation._source_metadata_path(
                {"metadata": f"{ROOT}/sources/apple/transcript.json"}
            ),
            episode_generation._source_metadata_path(
                {"metadata": WHISPER_METADATA}
            ),
        ),
        "podcast_engine/human_review.py::third_asr_claim_path": lambda: (
            human_review.third_asr_claim_path(
                KEY,
                source_fingerprint,
                184,
            ),
        ),
        "podcast_engine/human_review.py::review_record_path": lambda: (
            human_review.review_record_path(KEY),
        ),
        "podcast_engine/human_review.py::prepare_session_path": lambda: (
            human_review.prepare_session_path(
                KEY,
                source_fingerprint,
                session_id,
            ),
        ),
    }


def _assert_path_function_iam_completeness(
    root: Path,
    cases: dict[str, object],
    *,
    non_gcs: set[str] | frozenset[str] = frozenset(),
) -> None:
    discovered = _discover_path_named_functions(root)
    classified = set(cases) | set(non_gcs)
    if discovered != classified:
        unexpected = sorted(discovered - classified)
        missing = sorted(classified - discovered)
        raise AssertionError(
            "path-function inventory drifted; "
            f"unclassified={unexpected}, missing={missing}"
        )

    patterns = tuple(
        pattern
        for binding in ACCESS_BINDINGS
        for pattern in binding.patterns
    )
    if not patterns:
        raise AssertionError("IAM contract declares no object patterns")

    for qualified_name, produce_paths in cases.items():
        object_names = tuple(produce_paths())
        if not object_names:
            raise AssertionError(
                f"{qualified_name} produced no object paths for its audit case"
            )
        for object_name in object_names:
            if not isinstance(object_name, str) or not object_name:
                raise AssertionError(
                    f"{qualified_name} produced a non-object path: {object_name!r}"
                )
            if not any(pattern.matches(object_name) for pattern in patterns):
                raise AssertionError(
                    f"{qualified_name} produced IAM-uncovered object path: "
                    f"{object_name}"
                )


WRITER_PATHS = {
    WORKER_PRINCIPAL: (
        "episodes.json",
        f"incoming/apple/{KEY}/apple-transcript.json",
        f"incoming/apple/{KEY}/apple-transcript.txt",
        WHISPER_TEXT,
        WHISPER_METADATA,
        f"{ROOT}/compiled/transcript.txt",
        f"{ROOT}/compiled/report.json",
        f"{ROOT}/review/resolver.json",
        f"{ROOT}/summary/body.md",
        f"{ROOT}/summary/metadata.json",
        f"{ROOT}/summary/summary.md",
        AI_BUDGET_LEDGER_PATH,
        BUDGET_IDENTITY_RECONCILIATION_PATH,
        THIRD_ASR_CLAIM_PATH,
        "knowledge/tags/registry-v1.json",
    ),
    APPLE_INGEST_PRINCIPAL: (
        "episodes.json",
        f"incoming/apple/{KEY}/apple-transcript.json",
        f"incoming/apple/{KEY}/apple-transcript.txt",
        f"{ROOT}/sources/apple/transcript.txt",
        f"{ROOT}/sources/apple/transcript.json",
    ),
    REVIEW_PRINCIPAL: (
        f"{ROOT}/review/resolver.json",
        f"{ROOT}/summary/metadata.json",
        f"{ROOT}/summary/summary.md",
        AI_BUDGET_LEDGER_PATH,
        PREPARE_SESSION_PATH,
        THIRD_ASR_CLAIM_PATH,
        "knowledge/tags/registry-v1.json",
        "knowledge/artifacts/index-v1.json",
    ),
}

READER_PATHS = {
    WORKER_PRINCIPAL: (
        "episodes.json",
        f"{ROOT}/sources/apple/transcript.txt",
        f"{ROOT}/sources/apple/transcript.json",
        WHISPER_TEXT,
        WHISPER_METADATA,
        f"{ROOT}/compiled/transcript.txt",
        f"{ROOT}/review/resolver.json",
        f"{ROOT}/summary/body.md",
        f"{ROOT}/summary/metadata.json",
        f"{ROOT}/summary/summary.md",
        AI_BUDGET_LEDGER_PATH,
        BUDGET_IDENTITY_RECONCILIATION_PATH,
        THIRD_ASR_CLAIM_PATH,
        "knowledge/tags/registry-v1.json",
    ),
    APPLE_INGEST_PRINCIPAL: (
        "episodes.json",
        f"incoming/apple/{KEY}/apple-transcript.json",
        f"incoming/apple/{KEY}/apple-transcript.txt",
        f"{ROOT}/sources/apple/transcript.txt",
        f"{ROOT}/sources/apple/transcript.json",
    ),
    REVIEW_PRINCIPAL: (
        "episodes.json",
        f"{ROOT}/sources/apple/transcript.txt",
        f"{ROOT}/sources/apple/transcript.json",
        WHISPER_TEXT,
        WHISPER_METADATA,
        f"{ROOT}/review/resolver.json",
        f"{ROOT}/summary/body.md",
        f"{ROOT}/summary/metadata.json",
        f"{ROOT}/summary/summary.md",
        AI_BUDGET_LEDGER_PATH,
        BUDGET_IDENTITY_RECONCILIATION_PATH,
        PREPARE_SESSION_PATH,
        THIRD_ASR_CLAIM_PATH,
        "knowledge/tags/registry-v1.json",
        "knowledge/artifacts/index-v1.json",
        "knowledge/agents/macos-knowledge-sync-v1.json",
    ),
}


class WhisperProducerIamPolicyTests(unittest.TestCase):
    def test_declares_only_minimum_reader_and_writer_permissions(self):
        self.assertEqual(READER_PERMISSIONS, {"storage.objects.get"})
        self.assertEqual(
            WRITER_PERMISSIONS,
            {
                "storage.objects.get",
                "storage.objects.create",
                "storage.objects.delete",
            },
        )
        for binding in ACCESS_BINDINGS:
            self.assertFalse(binding.permissions & FORBIDDEN_PERMISSIONS)
            self.assertTrue(binding.patterns, "unconditional binding is forbidden")
            self.assertEqual(
                binding.permissions,
                READER_PERMISSIONS if binding.kind == "reader" else WRITER_PERMISSIONS,
            )
        validate_contract()

    def test_summary_review_paths_preserve_named_object_least_privilege(self):
        review_id = "sr-0123456789abcdef01234567"
        claim_digest = "a" * 64

        history_index = f"{ROOT}/summary/reviews/index.json"
        review_root = f"{ROOT}/summary/reviews/{review_id}"
        evidence_paths = (
            f"{review_root}/draft.md",
            f"{review_root}/final.md",
            f"{review_root}/review.json",
            f"{review_root}/transcript-span-index.json",
            f"{review_root}/draft-block-index.json",
            f"{review_root}/risk-inventory.json",
        )
        stalled_claim = (
            f"{ROOT}/summary/review_notifications/"
            f"stalled_email/{claim_digest}.json"
        )

        for object_name in (
            history_index,
            *evidence_paths,
            stalled_claim,
        ):
            for permission in (
                "storage.objects.get",
                "storage.objects.create",
                "storage.objects.delete",
            ):
                with self.subTest(
                    principal="worker",
                    permission=permission,
                    object_name=object_name,
                ):
                    self.assertTrue(
                        is_allowed(
                            WORKER_PRINCIPAL,
                            permission,
                            object_name,
                        )
                    )

        review_json = f"{review_root}/review.json"
        for object_name in (history_index, review_json):
            self.assertTrue(
                is_allowed(
                    REVIEW_PRINCIPAL,
                    "storage.objects.get",
                    object_name,
                )
            )
            for permission in (
                "storage.objects.create",
                "storage.objects.delete",
            ):
                self.assertFalse(
                    is_allowed(
                        REVIEW_PRINCIPAL,
                        permission,
                        object_name,
                    )
                )

        for object_name in (
            f"{review_root}/draft.md",
            f"{review_root}/final.md",
            f"{review_root}/transcript-span-index.json",
            f"{review_root}/draft-block-index.json",
            f"{review_root}/risk-inventory.json",
            stalled_claim,
        ):
            self.assertFalse(
                is_allowed(
                    REVIEW_PRINCIPAL,
                    "storage.objects.get",
                    object_name,
                )
            )

        for principal in (WORKER_PRINCIPAL, REVIEW_PRINCIPAL):
            for object_name in (history_index, review_json):
                self.assertFalse(
                    is_allowed(
                        principal,
                        "storage.objects.list",
                        object_name,
                    )
                )

        for object_name in (
            f"{review_root}/private-debug.json",
            f"{review_root}/review.json.bak",
            f"{ROOT}/summary/review_notifications/"
            f"stalled_email/{claim_digest}.json.bak",
        ):
            for principal in (WORKER_PRINCIPAL, REVIEW_PRINCIPAL):
                for permission in WRITER_PERMISSIONS:
                    self.assertFalse(
                        is_allowed(
                            principal,
                            permission,
                            object_name,
                        )
                    )

    def test_all_audited_writer_paths_are_allowed_for_create_and_delete(self):
        positive_cases = 0
        for principal, paths in WRITER_PATHS.items():
            for object_name in paths:
                for permission in ("storage.objects.create", "storage.objects.delete"):
                    with self.subTest(principal=principal, permission=permission, object_name=object_name):
                        self.assertTrue(is_allowed(principal, permission, object_name))
                    positive_cases += 1
        self.assertEqual(positive_cases, 56)

    def test_all_audited_reader_paths_are_allowed(self):
        positive_cases = 0
        for principal, paths in READER_PATHS.items():
            for object_name in paths:
                with self.subTest(principal=principal, object_name=object_name):
                    self.assertTrue(is_allowed(principal, "storage.objects.get", object_name))
                positive_cases += 1
        self.assertEqual(positive_cases, 35)

    def test_budget_identity_reconciliation_marker_is_review_and_worker_only(self):
        # TASK-126: Human Review and the Worker-side Third-ASR prefetch may
        # create the marker for a generation proven clean by named reads;
        # Apple ingest never touches it, and listing stays forbidden.
        for principal in (REVIEW_PRINCIPAL, WORKER_PRINCIPAL):
            for permission in ("storage.objects.get", "storage.objects.create"):
                self.assertTrue(
                    is_allowed(
                        principal,
                        permission,
                        BUDGET_IDENTITY_RECONCILIATION_PATH,
                    )
                )
        for permission in ("storage.objects.get", "storage.objects.create"):
            self.assertFalse(
                is_allowed(APPLE_INGEST_PRINCIPAL, permission, BUDGET_IDENTITY_RECONCILIATION_PATH)
            )
        self.assertIn("storage.objects.list", FORBIDDEN_PERMISSIONS)

    def test_only_worker_can_create_or_delete_canonical_whisper_artifacts(self):
        for object_name in (WHISPER_TEXT, WHISPER_METADATA):
            for permission in ("storage.objects.create", "storage.objects.delete"):
                with self.subTest(permission=permission, object_name=object_name, principal="worker"):
                    self.assertTrue(is_allowed(WORKER_PRINCIPAL, permission, object_name))
                for principal in (APPLE_INGEST_PRINCIPAL, REVIEW_PRINCIPAL):
                    with self.subTest(permission=permission, object_name=object_name, principal=principal):
                        self.assertFalse(is_allowed(principal, permission, object_name))

    def test_non_producers_reject_whisper_near_misses_and_malformed_paths(self):
        near_misses = (
            f"{ROOT}/sources/whisper/transcript.json.bak",
            f"{ROOT}/sources/whisper/transcript.txt.tmp",
            f"{ROOT}/foo/sources/whisper/transcript.txt",
            f"incoming/apple/{KEY}/../sources/whisper/transcript.txt",
            "episodes/0123456789ABCDEF01234567/sources/whisper/transcript.txt",
            "episodes/0123456789abcdef0123456/sources/whisper/transcript.json",
        )
        negative_cases = 0
        for principal in (APPLE_INGEST_PRINCIPAL, REVIEW_PRINCIPAL):
            for object_name in near_misses:
                for permission in ("storage.objects.create", "storage.objects.delete"):
                    with self.subTest(principal=principal, permission=permission, object_name=object_name):
                        self.assertFalse(is_allowed(principal, permission, object_name))
                    negative_cases += 1
        self.assertEqual(negative_cases, 24)

    def test_all_principals_reject_malformed_episode_keys_for_named_object_access(self):
        malformed_paths = (
            "episodes/0123456789ABCDEF01234567/sources/whisper/transcript.txt",
            "episodes/0123456789abcdef0123456/sources/whisper/transcript.json",
            "episodes/0123456789abcdef012345670/sources/apple/transcript.txt",
            "episodes/../sources/whisper/transcript.txt",
        )
        for principal in (WORKER_PRINCIPAL, APPLE_INGEST_PRINCIPAL, REVIEW_PRINCIPAL):
            for object_name in malformed_paths:
                for permission in WRITER_PERMISSIONS:
                    with self.subTest(principal=principal, permission=permission, object_name=object_name):
                        self.assertFalse(is_allowed(principal, permission, object_name))

    def test_cross_namespace_writer_denials(self):
        denied_paths = {
            APPLE_INGEST_PRINCIPAL: (
                WHISPER_TEXT,
                f"{ROOT}/compiled/transcript.txt",
                f"{ROOT}/review/resolver.json",
                f"{ROOT}/summary/summary.md",
                "knowledge/tags/registry-v1.json",
                "knowledge/artifacts/index-v1.json",
                "knowledge/agents/macos-knowledge-sync-v1.json",
            ),
            REVIEW_PRINCIPAL: (
                WHISPER_METADATA,
                f"{ROOT}/sources/apple/transcript.txt",
                f"{ROOT}/sources/apple/transcript.json",
                f"incoming/apple/{KEY}/apple-transcript.txt",
                f"{ROOT}/compiled/report.json",
                "episodes.json",
                f"{ROOT}/summary/body.md",
            ),
            WORKER_PRINCIPAL: (
                "knowledge/artifacts/index-v1.json",
                "knowledge/agents/macos-knowledge-sync-v1.json",
            ),
        }
        negative_cases = 0
        for principal, paths in denied_paths.items():
            for object_name in paths:
                for permission in ("storage.objects.create", "storage.objects.delete"):
                    with self.subTest(principal=principal, permission=permission, object_name=object_name):
                        self.assertFalse(is_allowed(principal, permission, object_name))
                    negative_cases += 1
        self.assertEqual(negative_cases, 32)

    def test_cel_conditions_are_deterministic_and_scope_every_binding_to_objects(self):
        bindings = generate_cel_conditions()
        self.assertEqual(bindings, ACCESS_BINDINGS)
        self.assertEqual(len(bindings), 6)
        for binding in bindings:
            expression = binding.cel_expression()
            with self.subTest(principal=binding.principal, kind=binding.kind):
                self.assertTrue(expression.startswith("resource.type == 'storage.googleapis.com/Object' && ("))
                self.assertIn("resource.name", expression)
                self.assertNotIn("*", expression)
                self.assertNotIn("storage.objects.list", expression)

    def test_physical_iam_bindings_are_deterministic_and_google_lint_safe(self):
        first_emission = generate_physical_iam_bindings()
        second_emission = generate_physical_iam_bindings()

        self.assertEqual(first_emission, second_emission)
        self.assertEqual(len(first_emission), 21)
        for binding in first_emission:
            expression = binding.cel_expression()
            with self.subTest(principal=binding.principal, kind=binding.kind, expression=expression):
                self.assertTrue(binding.patterns, "unconditional binding is forbidden")
                self.assertEqual(
                    binding.permissions,
                    READER_PERMISSIONS if binding.kind == "reader" else WRITER_PERMISSIONS,
                )
                self.assertLessEqual(
                    logical_operator_count(expression),
                    MAX_CEL_LOGICAL_OPERATORS,
                )
                self.assertIn("resource.name", expression)
                self.assertNotIn(OBJECT_RESOURCE_PREFIX + "'", expression)
                self.assertFalse(binding.permissions & FORBIDDEN_PERMISSIONS)

    def test_split_physical_bindings_preserve_each_conceptual_decision(self):
        physical_bindings = generate_physical_iam_bindings()
        object_names = {
            *WRITER_PATHS[WORKER_PRINCIPAL],
            *WRITER_PATHS[APPLE_INGEST_PRINCIPAL],
            *WRITER_PATHS[REVIEW_PRINCIPAL],
            *READER_PATHS[WORKER_PRINCIPAL],
            *READER_PATHS[APPLE_INGEST_PRINCIPAL],
            *READER_PATHS[REVIEW_PRINCIPAL],
            "episodes/0123456789ABCDEF01234567/sources/whisper/transcript.txt",
            "episodes/0123456789abcdef0123456/sources/whisper/transcript.json",
            "episodes/not-an-episode-key/sources/whisper/transcript.txt",
            "unrelated/object.txt",
        }
        permissions = {
            *READER_PERMISSIONS,
            *WRITER_PERMISSIONS,
            *FORBIDDEN_PERMISSIONS,
        }

        self.assertEqual(len(ACCESS_BINDINGS), 6)
        for conceptual_binding in ACCESS_BINDINGS:
            emitted_bindings = tuple(
                binding
                for binding in physical_bindings
                if binding.principal == conceptual_binding.principal
                and binding.kind == conceptual_binding.kind
            )
            self.assertTrue(emitted_bindings)
            self.assertEqual(
                tuple(pattern for binding in emitted_bindings for pattern in binding.patterns),
                conceptual_binding.patterns,
            )
            for permission in permissions:
                for object_name in object_names:
                    with self.subTest(
                        principal=conceptual_binding.principal,
                        kind=conceptual_binding.kind,
                        permission=permission,
                        object_name=object_name,
                    ):
                        self.assertEqual(
                            any(
                                binding.allows(permission, object_name)
                                for binding in emitted_bindings
                            ),
                            conceptual_binding.allows(permission, object_name),
                        )

    def test_physical_bindings_preserve_whisper_writer_boundary(self):
        physical_bindings = generate_physical_iam_bindings()
        for principal, expected in (
            (APPLE_INGEST_PRINCIPAL, False),
            (REVIEW_PRINCIPAL, False),
            (WORKER_PRINCIPAL, True),
        ):
            for object_name in (WHISPER_TEXT, WHISPER_METADATA):
                for permission in ("storage.objects.create", "storage.objects.delete"):
                    with self.subTest(principal=principal, permission=permission, object_name=object_name):
                        self.assertEqual(
                            any(
                                binding.allows(permission, object_name)
                                for binding in physical_bindings
                                if binding.principal == principal
                            ),
                            expected,
                        )

    def test_generator_fails_closed_when_a_pattern_clause_exceeds_the_limit(self):
        oversized_clause = " && ".join("true" for _ in range(12))
        with mock.patch(
            "security.whisper_producer_iam.ObjectPattern.cel_clause",
            return_value=oversized_clause,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "CEL clause alone has 11 logical operators, exceeding the 10-operator IAM limit",
            ):
                generate_physical_iam_bindings()

    def test_non_producer_writer_conditions_have_no_broad_episode_match(self):
        for principal in (APPLE_INGEST_PRINCIPAL, REVIEW_PRINCIPAL):
            binding = bindings_for(principal, "writer")[0]
            expression = binding.cel_expression()
            with self.subTest(principal=principal):
                self.assertNotIn("resource.name == '" + OBJECT_RESOURCE_PREFIX + "episodes/'", expression)
                for pattern in binding.patterns:
                    if pattern.kind == "episode":
                        self.assertTrue(pattern.value.startswith("/"))
                        self.assertIn("resource.name.endsWith(", pattern.cel_clause())
                self.assertNotIn("/sources/whisper/", expression)

    def test_worker_writer_condition_is_not_bucket_wide_or_unconditional(self):
        binding = bindings_for(WORKER_PRINCIPAL, "writer")[0]
        self.assertTrue(binding.patterns)
        self.assertTrue(all(pattern.value for pattern in binding.patterns))
        self.assertTrue(
            all(
                pattern.kind
                in {
                    "exact",
                    "episode",
                    "incoming_apple",
                    "summary_review",
                    "stalled_email",
                    "ai_budget",
                    "budget_identity_reconciliation",
                    "third_asr_claim",
                }
                for pattern in binding.patterns
            )
        )
        self.assertNotIn(OBJECT_RESOURCE_PREFIX + "'", binding.cel_expression())

    def test_unknown_principal_and_unsupported_permissions_are_denied(self):
        self.assertFalse(is_allowed("unknown@example.test", "storage.objects.get", WHISPER_TEXT))
        for permission in FORBIDDEN_PERMISSIONS:
            self.assertFalse(is_allowed(WORKER_PRINCIPAL, permission, WHISPER_TEXT))

    def test_every_podcast_engine_gcs_path_function_is_covered_by_iam_pattern(self):
        root = Path(__file__).resolve().parents[1]
        _assert_path_function_iam_completeness(
            root,
            _gcs_path_function_cases(),
            non_gcs=NON_GCS_PATH_FUNCTIONS,
        )

    def test_iam_path_completeness_guard_rejects_deliberately_uncovered_path_function(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine_root = root / "podcast_engine"
            engine_root.mkdir()
            module_path = engine_root / "future_paths.py"
            module_path.write_text(
                "def deliberately_uncovered_path(episode_key: str) -> str:\n"
                "    return f\"episodes/{episode_key}/future/uncovered.json\"\n",
                encoding="utf-8",
            )
            namespace: dict[str, object] = {}
            exec(
                compile(
                    module_path.read_text(encoding="utf-8"),
                    str(module_path),
                    "exec",
                ),
                namespace,
            )
            cases = {
                "podcast_engine/future_paths.py::deliberately_uncovered_path": lambda: (
                    namespace["deliberately_uncovered_path"](KEY),
                ),
            }

            with self.assertRaisesRegex(
                AssertionError,
                "IAM-uncovered object path",
            ):
                _assert_path_function_iam_completeness(root, cases)

    def test_generic_upload_helper_has_only_the_audited_producer_call_sites(self):
        root = Path(__file__).resolve().parents[1]
        call_sites = {}
        for path in sorted((root / "podcast_engine").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if any(
                    isinstance(call.func, ast.Name) and call.func.id == "upload_path_to_gcs"
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call)
                ):
                    functions.append(node.name)
            if functions:
                call_sites[path.relative_to(root).as_posix()] = sorted(functions)
        self.assertEqual(
            call_sites,
            {
                "podcast_engine/compilation.py": ["compile_episode_sources"],
                "podcast_engine/transcriber.py": ["transcribe_audio"],
            },
        )


if __name__ == "__main__":
    unittest.main()
