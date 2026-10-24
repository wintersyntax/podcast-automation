import plistlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = "/Users/YOUR_USER/Documents/GitHub/podcast-automation/.venv/bin/python"
LAUNCHD_PLISTS = (
    "com.user.podcast-vault-sync.plist",
    "com.user.podcast-apple-token-maintenance.plist",
    "com.user.podcast-knowledge-sync.plist",
)


def test_all_macos_launchagents_use_canonical_project_venv():
    for plist_name in LAUNCHD_PLISTS:
        plist_path = REPO_ROOT / "macos_agent" / "launchd" / plist_name
        with plist_path.open("rb") as handle:
            payload = plistlib.load(handle)

        assert payload["ProgramArguments"][0] == EXPECTED_PYTHON, plist_name


def test_knowledge_sync_manual_run_uses_canonical_project_venv():
    readme = (REPO_ROOT / "macos_agent" / "README.md").read_text(encoding="utf-8")

    assert ".venv/bin/python -m macos_agent.knowledge_sync" in readme
    assert "\nvenv/bin/python -m macos_agent.knowledge_sync\n" not in readme
