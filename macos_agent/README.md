# macOS agent

This package has three local responsibilities:

1. Maintain the Apple Podcasts bearer credential for the cloud worker.
2. Copy canonical final summaries from GCS into the local Obsidian/Minknote
   Vault.
3. Index configured local knowledge notes and execute explicit knowledge-tag
   adoption/backfill work.

It does not acquire Apple transcript source files, update cloud episode state,
or run GUI/AppleScript recovery tools.

## Apple bearer-token maintenance

`apple_token_maintenance.py` reads only the local JWT expiry. At five days or
less remaining it runs the existing `bin/FetchTranscript` helper solely to
generate a credential, validates that result with one Apple catalogue probe,
and then adds a new version to Secret Manager. The probe does not write GCS.

The helper always runs in a fresh, empty directory. Its `--cache-bearer-token`
mode reuses any `bearer_token.txt` in its working directory whose file is
younger than 30 days, whatever the token's real expiry, so running it beside
the installed token would only return that same token (TASK-116). A new token
is installed atomically and only if it expires later than the current one; on
every failure the installed token is left untouched. A token the helper wrote
before its seed-transcript step failed is still accepted, because the Apple
probe gates the Secret Manager update.
Existing versions are never disabled or deleted. The helper, token and Secret
Manager payload are never logged.

`launchd/com.user.podcast-apple-token-maintenance.plist` runs at 20:00 in the
Mac's local time and at load. This is after the 18:00 Cloud Scheduler run and
before the next day's 09:00 run. It executes `macos_agent.apple_token_maintenance`.
The optional `--force` flag is only for a controlled operational verification;
normal scheduled execution preserves the five-day threshold.

`PODCAST_TOKEN_MAINTENANCE_HEARTBEAT_URL` may contain an external monitor URL.
Set it in the uncommitted repository `.env` (or the installed LaunchAgent
environment); it is never committed. A heartbeat is sent only after a
successful no-op or a successful refresh, probe and Secret Manager update; any
failure sends none.

To install the local helper when needed:

```bash
./macos_agent/install_fetch_transcript_helper.sh
```

## Vault summary sync

`vault_sync.py` copies only ready/completed canonical GCS `summary.md` files
to `PODCAST_VAULT_DIR` (optional relative `PODCAST_VAULT_SUBDIR`). Identical
files are no-ops. Changed files are replaced atomically. If an existing iCloud
file cannot be read because macOS reports `Resource deadlock avoided`, Vault
Sync replaces it from the canonical GCS bytes instead of abandoning the run;
other read/write failures still stop the sync. A scheduled failure can send an
immediate Healthchecks `/fail` signal through `PODCAST_VAULT_SYNC_HEARTBEAT_URL`,
and success is reported only after the full scheduled run completes. Single-
episode manual syncs do not change monitor state. Its LaunchAgent runs on load
and at 09:00, 15:00 and 21:00.

Both plist files are validated with `plutil -lint`; installed jobs must point
to `macos_agent.apple_token_maintenance` and `macos_agent.vault_sync`.

## Knowledge sync

`knowledge_sync.py` is a small, finite executor. It scans only roots named by
`PODCAST_KNOWLEDGE_ROOTS`; it is not a watcher, daemon, backup, notification,
or an Obsidian/iCloud integration. Configure one or more entries in the
uncommitted `.env`, separated by `:` on macOS:

```bash
PODCAST_KNOWLEDGE_ROOTS='personal=/Users/YOUR_USER/Library/Mobile Documents/com~apple~CloudDocs/Knowledge:work=/Users/YOUR_USER/Documents/Work Knowledge'
```

Every entry is `vault-id=/absolute/path`. The agent reads only `.md` files
under those roots. A note is relevant when its YAML frontmatter has `type`,
`tags`, or `knowledge_id`. A note without `knowledge_id` becomes a visible
candidate; the web UI's **Adopt into Knowledge System** action queues work in
GCS. On a later run the agent re-reads that exact local file and adds a stable
`knowledge_id: note:...` atomically. Existing IDs remain the artifact ID when
notes move between folders.

Only `knowledge_id` and `tags` are system-managed. Writes are minimal,
idempotent, same-directory atomic replacements: user-owned frontmatter and
the Markdown body are retained. Markdown body text is never uploaded to GCS.
Known canonical tags are accepted; aliases remain unchanged until an explicit
review/backfill; unknown tags become global unresolved candidates. Promote,
Map, Reject and Reopen decide registry state only. Only **Run backfill** makes
local `waiting_for_agent` work executable.

Run it manually with:

```bash
.venv/bin/python -m macos_agent.knowledge_sync
```

`launchd/com.user.podcast-knowledge-sync.plist` runs at 08:00, 11:00, 14:00,
18:00 and 22:00 local Mac time, without `KeepAlive` or `RunAtLoad`. Validate
before installing, then copy/load it only when explicitly desired:

```bash
plutil -lint examples/launchd/com.user.podcast-knowledge-sync.plist
cp examples/launchd/com.user.podcast-knowledge-sync.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.user.podcast-knowledge-sync.plist
```

To uninstall later, use `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.user.podcast-knowledge-sync.plist` and then remove that installed copy.

Each run records a GCS-only management heartbeat (no note content): status,
timestamps, duration, scan/change/backfill/no-op/error totals, unresolved-tag
count, and waiting work. The Tag Registry UI reads that saved state; it never
contacts the Mac directly. A status becomes `Stale` after 18 hours without a
completed run, deliberately allowing several missed scheduled runs. If the
Mac is offline, `waiting_for_agent` remains normal and work is retried later.
Malformed notes and individual work-item errors are reported and do not stop
the remaining scan/work items. For troubleshooting, first verify the root
configuration and local file permissions, then inspect the structured local
launchd log and the web heartbeat/error counters. The lock file is an advisory
OS lock under the system temporary directory; a second run exits safely.
