import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BLOCKED_IMPORT_SCRIPT = textwrap.dedent(
    """
    import importlib
    import importlib.abc
    import json
    import sys

    class BlockGoogle(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "google" or fullname.startswith("google."):
                raise ModuleNotFoundError(f"blocked Google dependency: {fullname}")
            return None

    sys.meta_path.insert(0, BlockGoogle())
    imported = importlib.import_module(sys.argv[1])
    print(json.dumps({
        "module": imported.__name__,
        "storage_loaded": "podcast_engine.storage" in sys.modules,
        "orchestration_loaded": "podcast_engine.knowledge.orchestration" in sys.modules,
        "google_cloud_storage_loaded": "google.cloud.storage" in sys.modules,
    }, sort_keys=True))
    """
)

EXPORT_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import podcast_engine.knowledge as package

    package_was_lightweight = (
        "podcast_engine.storage" not in sys.modules
        and "podcast_engine.knowledge.orchestration" not in sys.modules
    )
    from podcast_engine.knowledge import (
        build_knowledge_note,
        canonical_summary_generated_at,
    )
    from podcast_engine.knowledge import orchestration

    print(json.dumps({
        "package_was_lightweight": package_was_lightweight,
        "storage_loaded": "podcast_engine.storage" in sys.modules,
        "orchestration_loaded": "podcast_engine.knowledge.orchestration" in sys.modules,
        "build_identity": build_knowledge_note is orchestration.build_knowledge_note,
        "generated_at_identity": (
            canonical_summary_generated_at
            is orchestration.canonical_summary_generated_at
        ),
    }, sort_keys=True))
    """
)


class KnowledgeImportBoundaryTests(unittest.TestCase):
    def _run_script(self, script: str, *arguments: str) -> dict:
        result = subprocess.run(
            [sys.executable, "-c", script, *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_pure_modules_import_without_google_or_storage(self):
        modules = (
            "podcast_engine.knowledge.episode_inputs",
            "podcast_engine.knowledge.models",
            "podcast_engine.knowledge.notes_v3.schema",
            "podcast_engine.knowledge.notes_v3.writer",
        )
        for module_name in modules:
            with self.subTest(module=module_name):
                result = self._run_script(BLOCKED_IMPORT_SCRIPT, module_name)
                self.assertEqual(result["module"], module_name)
                self.assertFalse(result["storage_loaded"])
                self.assertFalse(result["orchestration_loaded"])
                self.assertFalse(result["google_cloud_storage_loaded"])

    def test_package_import_is_lightweight_with_google_blocked(self):
        result = self._run_script(
            BLOCKED_IMPORT_SCRIPT,
            "podcast_engine.knowledge",
        )
        self.assertEqual(result["module"], "podcast_engine.knowledge")
        self.assertFalse(result["storage_loaded"])
        self.assertFalse(result["orchestration_loaded"])
        self.assertFalse(result["google_cloud_storage_loaded"])

    def test_supported_exports_resolve_to_real_orchestration(self):
        result = self._run_script(EXPORT_SCRIPT)
        self.assertTrue(result["package_was_lightweight"])
        self.assertTrue(result["storage_loaded"])
        self.assertTrue(result["orchestration_loaded"])
        self.assertTrue(result["build_identity"])
        self.assertTrue(result["generated_at_identity"])


if __name__ == "__main__":
    unittest.main()
