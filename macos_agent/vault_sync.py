#!/usr/bin/env python3
"""Copy canonical GCS podcast summaries into a local Markdown vault.

This is intentionally a Mac-side, manual/launchd-friendly command. It is not
imported by the Cloud Run worker, so a missing local vault configuration never
affects the cloud pipeline.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from dotenv import load_dotenv

from podcast_engine.episode_contract import (
    paths_for,
    summary_is_vault_sync_ready,
)
from podcast_engine.storage import download_gcs_bytes, load_episodes


DEFAULT_VAULT_SUBDIR = "Podcasts"
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

VAULT_SYNC_HEARTBEAT_ENV = "PODCAST_VAULT_SYNC_HEARTBEAT_URL"
CURL_BIN = "/usr/bin/curl"
HEARTBEAT_TIMEOUT_SECONDS = 10

_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


def load_repo_env() -> None:
    """Load the repository .env without overriding explicit environment values."""
    load_dotenv(
        dotenv_path=ENV_PATH,
        override=False,
    )


def vault_root_from_env(env: dict[str, str] | None = None) -> Path:
    """Return the configured vault root, or a clear local-only error."""
    source = os.environ if env is None else env
    value = source.get("PODCAST_VAULT_DIR", "").strip()

    if not value:
        raise ValueError(
            "PODCAST_VAULT_DIR is not set; "
            "set it to your Obsidian/Minknote vault folder"
        )

    return Path(value).expanduser()


def vault_subdir_from_env(env: dict[str, str] | None = None) -> Path:
    """Return a safe relative folder under the vault root."""
    source = os.environ if env is None else env
    value = source.get(
        "PODCAST_VAULT_SUBDIR",
        DEFAULT_VAULT_SUBDIR,
    ).strip()

    subdir = Path(value or DEFAULT_VAULT_SUBDIR)

    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError(
            "PODCAST_VAULT_SUBDIR must be a relative folder "
            "inside PODCAST_VAULT_DIR"
        )

    return subdir


def vault_sync_heartbeat_url(
    env: dict[str, str] | None = None,
) -> str:
    """Return the optional Healthchecks heartbeat URL."""
    source = os.environ if env is None else env
    return source.get(VAULT_SYNC_HEARTBEAT_ENV, "").strip()


def send_vault_sync_heartbeat(
    env: dict[str, str] | None = None,
) -> bool:
    """Send a success heartbeat without exposing the configured URL.

    A missing heartbeat URL is a supported configuration and is treated as a
    no-op. A failed heartbeat never turns an otherwise successful vault sync
    into a failed vault sync; Healthchecks itself will detect the missed ping.
    """
    heartbeat_url = vault_sync_heartbeat_url(env)

    if not heartbeat_url:
        return False

    try:
        result = subprocess.run(
            [
                CURL_BIN,
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                str(HEARTBEAT_TIMEOUT_SECONDS),
                heartbeat_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=HEARTBEAT_TIMEOUT_SECONDS + 2,
        )
    except (OSError, subprocess.SubprocessError):
        print(
            '{"severity":"WARNING",'
            '"event":"vault_sync_heartbeat_failed",'
            '"message":"Vault sync heartbeat delivery failed."}'
        )
        return False

    if result.returncode != 0:
        print(
            '{"severity":"WARNING",'
            '"event":"vault_sync_heartbeat_failed",'
            '"message":"Vault sync heartbeat delivery failed.",'
            f'"return_code":{result.returncode}'
            "}"
        )
        return False

    print(
        '{"severity":"INFO",'
        '"event":"vault_sync_heartbeat_sent",'
        '"message":"Vault sync heartbeat delivered successfully."}'
    )
    return True


def sanitize_filename(value: object, fallback: str) -> str:
    """Keep readable Unicode while removing characters unsafe for note names."""
    text = unicodedata.normalize("NFC", str(value or ""))
    text = _UNSAFE_FILENAME_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip(" .")
    return text or fallback


def episode_date(episode: dict) -> str:
    """Derive the stable YYYY-MM-DD filename prefix from public metadata."""
    for field in ("published", "created_at"):
        value = episode.get(field)

        if not isinstance(value, str) or not value.strip():
            continue

        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError, OverflowError):
            try:
                parsed = datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )
            except ValueError:
                continue

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc).date().isoformat()

    return "Undated"


def destination_for_episode(
    episode: dict,
    vault_root: Path,
    vault_subdir: Path,
) -> Path:
    podcast = sanitize_filename(
        episode.get("podcast"),
        "Unknown Podcast",
    )
    title = sanitize_filename(
        episode.get("title"),
        "Untitled episode",
    )

    return (
        vault_root
        / vault_subdir
        / podcast
        / f"{episode_date(episode)} - {title}.md"
    )


def _has_yaml_frontmatter(content: bytes) -> bool:
    return (
        content.startswith(b"---\n")
        and b"\n---\n" in content[4:]
    )


def export_vault_summary(
    episode: dict,
    *,
    vault_root: Path | None = None,
    vault_subdir: Path | None = None,
) -> tuple[Path, bool]:
    """Copy one canonical final summary, preserving mtime when bytes match."""
    episode_key = (
        episode.get("episode_key")
        or episode.get("id")
    )

    if not episode_key:
        raise ValueError("episode is missing episode_key")

    summary_path = paths_for(episode_key)["summary"]
    content = download_gcs_bytes(summary_path)

    if not _has_yaml_frontmatter(content):
        raise ValueError(
            f"canonical summary has no YAML frontmatter: {summary_path}"
        )

    root = (
        vault_root
        if vault_root is not None
        else vault_root_from_env()
    )
    subdir = (
        vault_subdir
        if vault_subdir is not None
        else vault_subdir_from_env()
    )

    destination = destination_for_episode(
        episode,
        root,
        subdir,
    )

    if (
        destination.exists()
        and destination.read_bytes() == content
    ):
        print(f"Vault summary unchanged: {destination}")
        return destination, False

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_bytes(content)

    print(f"Vault summary exported: {destination}")
    return destination, True


def ready_summary_episodes(
    episodes: list[dict],
) -> list[dict]:
    """Select index records whose final summaries are ready for local export."""
    selected = []

    for episode in episodes:
        if summary_is_vault_sync_ready(episode):
            selected.append(episode)

    return selected


def sync_episode(
    episode_key: str,
) -> tuple[Path, bool]:
    for episode in load_episodes():
        if (
            episode.get("episode_key") == episode_key
            or episode.get("id") == episode_key
        ):
            return export_vault_summary(episode)

    raise ValueError(
        f"episode not found in GCS index: {episode_key}"
    )


def sync_all_ready() -> tuple[int, int]:
    exported = 0
    unchanged = 0

    for episode in ready_summary_episodes(
        load_episodes()
    ):
        _, wrote = export_vault_summary(episode)

        if wrote:
            exported += 1
        else:
            unchanged += 1

    return exported, unchanged


def main(
    argv: list[str] | None = None,
) -> int:
    load_repo_env()

    parser = argparse.ArgumentParser(
        description=__doc__
    )
    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--episode-key",
        help="Sync one episode from canonical GCS summary.md",
    )
    group.add_argument(
        "--all-ready",
        action="store_true",
        help="Sync every ready/completed summary",
    )

    args = parser.parse_args(argv)

    try:
        # Fail before contacting GCS when the Mac has not been configured.
        vault_root_from_env()
        vault_subdir_from_env()

        if args.episode_key:
            sync_episode(args.episode_key)
        else:
            exported, unchanged = sync_all_ready()

            print(
                "Vault summary sync complete: "
                f"exported={exported} "
                f"unchanged={unchanged}"
            )

            # Only the scheduled --all-ready workflow owns the heartbeat.
            # Manual one-episode syncs must not reset the Healthchecks timer.
            send_vault_sync_heartbeat()

    except ValueError as error:
        print(f"Vault summary sync failed: {error}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
