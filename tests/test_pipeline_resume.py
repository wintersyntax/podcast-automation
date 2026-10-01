import unittest
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from podcast_engine.episode_contract import new_episode_record
from podcast_engine.episode_identity import episode_key_for
from podcast_engine.pipeline import (
    _apple_acquisition_due,
    _episode_is_completed,
    _transcribe_if_needed,
    discover_latest_rss_episode,
    resume_tracked_episode,
    run_pipeline,
)


class PipelineResumeTests(unittest.TestCase):
    podcast = {
        "id": "test-podcast",
        "name": "Test Podcast",
        "rss": "https://example.test/feed.xml",
    }

    def _episode(
        self,
        key,
        guid,
        *,
        published="Mon, 10 Aug 2026 12:00:00 +0000",
    ):
        return new_episode_record(
            episode_key=key,
            podcast="Test Podcast",
            podcast_id="test-podcast",
            feed_url="https://example.test/feed.xml",
            rss_guid=guid,
            title=f"Episode {guid}",
            published=published,
            link="https://example.test/episode",
            audio_url="https://example.test/audio.mp3",
        )

    def _waiting_for_apple(
        self,
        key,
        guid,
    ):
        episode = self._episode(
            key,
            guid,
        )
        episode["status"]["download"]["state"] = "completed"
        episode["status"]["whisper"]["state"] = "ready"
        episode["files"]["sources"]["whisper"]["text"] = "whisper.txt"
        return episode

    def _completed(
        self,
        key,
        guid,
    ):
        episode = self._waiting_for_apple(
            key,
            guid,
        )
        episode["files"]["sources"]["apple"]["text"] = "apple.txt"
        episode["status"]["apple_transcript"]["state"] = "ready"
        episode["status"]["compiler"]["state"] = "completed"
        episode["status"]["summary"]["state"] = "ready"
        return episode

    def _transcribe_after_failed_reconciliation(
        self,
        episode,
        *,
        file_exists,
        metadata_bytes=None,
    ):
        downloaded = deepcopy(episode)
        downloaded["status"]["download"]["state"] = "completed"
        transcribed = deepcopy(downloaded)
        transcribed["status"]["whisper"]["state"] = "ready"

        @contextmanager
        def local_audio(_url, _filename):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "episode.mp3"
                path.write_bytes(b"audio")
                yield path

        with (
            patch(
                "podcast_engine.pipeline.file_exists_in_gcs",
                side_effect=file_exists,
            ),
            patch(
                "podcast_engine.pipeline.download_gcs_bytes",
                return_value=metadata_bytes,
            ) as metadata,
            patch(
                "podcast_engine.pipeline.downloaded_audio",
                side_effect=local_audio,
            ) as download,
            patch(
                "podcast_engine.pipeline.transcribe_audio",
                return_value={
                    "text": "transcript.txt",
                    "metadata": "transcript.json",
                },
            ) as transcribe,
            patch(
                "podcast_engine.pipeline.update_episode",
                side_effect=[downloaded, transcribed],
            ),
        ):
            result = _transcribe_if_needed(episode)

        self.assertEqual(result, transcribed)
        download.assert_called_once()
        transcribe.assert_called_once()
        return metadata

    def test_completed_episode_requires_only_compiler_and_summary(self):
        episode = self._completed("completed-key", "completed-guid")

        self.assertTrue(_episode_is_completed(episode))

    def test_completed_compiler_with_pending_summary_is_incomplete(self):
        episode = self._completed("incomplete-key", "incomplete-guid")
        episode["status"]["summary"]["state"] = "pending"

        self.assertFalse(_episode_is_completed(episode))

    def _review_required(
        self,
        key="review-key",
        guid="review-guid",
    ):
        episode = self._waiting_for_apple(
            key,
            guid,
        )
        episode["files"]["sources"]["apple"]["text"] = "apple.txt"
        episode["status"]["apple_transcript"]["state"] = "ready"
        episode["status"]["compiler"]["state"] = "review_required"
        return episode

    def test_latest_rss_discovery_still_adds_a_new_episode(self):
        rss_episode = {
            "podcast": "Test Podcast",
            "guid": "latest-guid",
            "title": "Episode latest-guid",
            "feed_url": self.podcast["rss"],
        }
        added = self._waiting_for_apple(
            "latest-key",
            "latest-guid",
        )

        with (
            patch(
                "podcast_engine.pipeline._rss_episode",
                return_value=rss_episode,
            ),
            patch(
                "podcast_engine.pipeline.get_episode_by_key",
                return_value=None,
            ),
            patch(
                "podcast_engine.pipeline.add_episode",
                return_value=added,
            ) as add,
        ):
            result = discover_latest_rss_episode(
                self.podcast
            )

        add.assert_called_once_with(
            rss_episode
        )
        self.assertEqual(
            result["episode_key"],
            "latest-key",
        )

    def test_rss_discovery_does_not_cross_feeds_with_a_shared_guid(self):
        feed_a = "https://example.test/feed-a.xml"
        feed_b = "https://example.test/feed-b.xml"
        guid = "shared-guid"
        existing_a = self._waiting_for_apple(
            episode_key_for(feed_a, guid),
            guid,
        )
        existing_a["feed_url"] = feed_a
        rss_episode_b = {
            "podcast": "Feed B",
            "podcast_id": "feed-b",
            "feed_url": feed_b,
            "guid": guid,
            "title": "Feed B episode",
        }
        added_b = self._waiting_for_apple(
            episode_key_for(feed_b, guid),
            guid,
        )
        added_b["feed_url"] = feed_b

        with (
            patch(
                "podcast_engine.pipeline._rss_episode",
                return_value=rss_episode_b,
            ),
            patch(
                "podcast_engine.pipeline.get_episode_by_key",
                return_value=None,
            ) as get_by_key,
            patch(
                "podcast_engine.pipeline.add_episode",
                return_value=added_b,
            ) as add,
            patch("podcast_engine.pipeline.merge_episode_metadata") as merge,
        ):
            result = discover_latest_rss_episode(
                {"name": "Feed B", "rss": feed_b},
            )

        self.assertNotEqual(existing_a["episode_key"], added_b["episode_key"])
        get_by_key.assert_called_once_with(added_b["episode_key"])
        add.assert_called_once_with(rss_episode_b)
        merge.assert_not_called()
        self.assertEqual(existing_a["title"], "Episode shared-guid")
        self.assertEqual(result["episode_key"], added_b["episode_key"])

    def test_rss_discovery_refreshes_metadata_for_the_same_canonical_key(self):
        rss_episode = {
            "podcast": "Test Podcast",
            "podcast_id": "test-podcast",
            "feed_url": self.podcast["rss"],
            "guid": "latest-guid",
            "title": "Renamed episode",
        }
        existing = self._waiting_for_apple(
            episode_key_for(self.podcast["rss"], "latest-guid"),
            "latest-guid",
        )
        refreshed = deepcopy(existing)
        refreshed["title"] = rss_episode["title"]

        with (
            patch("podcast_engine.pipeline._rss_episode", return_value=rss_episode),
            patch(
                "podcast_engine.pipeline.get_episode_by_key",
                return_value=existing,
            ),
            patch(
                "podcast_engine.pipeline.merge_episode_metadata",
                return_value=refreshed,
            ) as merge,
            patch("podcast_engine.pipeline.add_episode") as add,
        ):
            result = discover_latest_rss_episode(self.podcast)

        merge.assert_called_once_with(existing["episode_key"], rss_episode)
        add.assert_not_called()
        self.assertEqual(result["title"], "Renamed episode")

    def test_whisper_ready_episode_does_not_reacquire_audio(self):
        episode = self._waiting_for_apple("ready-key", "ready-guid")

        with (
            patch("podcast_engine.pipeline.file_exists_in_gcs") as exists,
            patch("podcast_engine.pipeline.downloaded_audio") as download,
            patch("podcast_engine.pipeline.transcribe_audio") as transcribe,
        ):
            result = _transcribe_if_needed(episode)

        self.assertIs(result, episode)
        exists.assert_not_called()
        download.assert_not_called()
        transcribe.assert_not_called()

    def test_pending_whisper_reconciles_valid_canonical_artifacts(self):
        import json

        from podcast_engine.whisper_provenance import (
            WHISPER_METADATA_SCHEMA_VERSION,
            WHISPER_METADATA_SOURCE,
            current_whisper_producer,
        )

        episode = self._episode(
            "reconcile-key",
            "reconcile-guid",
        )
        reconciled = deepcopy(episode)
        reconciled["status"]["whisper"]["state"] = "ready"

        source_paths = {
            "text": (
                "episodes/reconcile-key/"
                "sources/whisper/transcript.txt"
            ),
            "metadata": (
                "episodes/reconcile-key/"
                "sources/whisper/transcript.json"
            ),
        }

        metadata_bytes = json.dumps(
            {
                "schema_version": WHISPER_METADATA_SCHEMA_VERSION,
                "source": WHISPER_METADATA_SOURCE,
                "model": "openai/whisper-large-v3",
                "producer": current_whisper_producer(),
                "segments": [],
            }
        ).encode("utf-8")

        with (
            patch(
                "podcast_engine.pipeline.file_exists_in_gcs",
                return_value=True,
            ),
            patch(
                "podcast_engine.pipeline.download_gcs_bytes",
                return_value=metadata_bytes,
            ),
            patch(
                "podcast_engine.pipeline.update_episode",
                return_value=reconciled,
            ) as update,
            patch(
                "podcast_engine.pipeline.downloaded_audio"
            ) as download,
            patch(
                "podcast_engine.pipeline.transcribe_audio"
            ) as transcribe,
        ):
            result = _transcribe_if_needed(episode)

        self.assertEqual(result, reconciled)

        update.assert_called_once_with(
            episode["episode_key"],
            transcript_file=source_paths["text"],
            whisper_metadata_file=source_paths["metadata"],
        )

        download.assert_not_called()
        transcribe.assert_not_called()

    def test_only_whisper_text_artifact_transcribes_normally(self):
        episode = self._episode("text-only-key", "text-only-guid")

        metadata = self._transcribe_after_failed_reconciliation(
            episode,
            file_exists=[True, False],
        )

        metadata.assert_not_called()

    def test_only_whisper_metadata_artifact_transcribes_normally(self):
        episode = self._episode("metadata-only-key", "metadata-only-guid")

        metadata = self._transcribe_after_failed_reconciliation(
            episode,
            file_exists=[False],
        )

        metadata.assert_not_called()

    def test_malformed_whisper_metadata_transcribes_normally(self):
        episode = self._episode("malformed-key", "malformed-guid")

        self._transcribe_after_failed_reconciliation(
            episode,
            file_exists=[True, True],
            metadata_bytes=b"not json",
        )

    def test_wrong_whisper_metadata_contract_transcribes_normally(self):
        for metadata in (
            b'{"schema_version": 1, "source": "faster-whisper", "segments": []}',
            b'{"schema_version": 2, "source": "faster-whisper", "segments": []}',
            b'{"schema_version": 1, "source": "other", "segments": []}',
        ):
            with self.subTest(metadata=metadata):
                episode = self._episode("invalid-key", "invalid-guid")
                self._transcribe_after_failed_reconciliation(
                    episode,
                    file_exists=[True, True],
                    metadata_bytes=metadata,
                )

    def test_incomplete_episode_reacquires_audio_from_rss_url(self):
        episode = self._episode("resume-key", "resume-guid")
        downloaded = deepcopy(episode)
        downloaded["status"]["download"]["state"] = "completed"
        transcribed = deepcopy(downloaded)
        transcribed["status"]["whisper"]["state"] = "ready"
        transcribed["files"]["sources"]["whisper"] = {
            "text": "episodes/resume-key/sources/whisper/transcript.txt",
            "metadata": "episodes/resume-key/sources/whisper/transcript.json",
        }

        @contextmanager
        def local_audio(_url, _filename):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "episode.mp3"
                path.write_bytes(b"audio")
                yield path

        def transcribe_local(path, key):
            self.assertTrue(path.is_file())
            self.assertEqual(key, "resume-key")
            return {
                "text": transcribed["files"]["sources"]["whisper"]["text"],
                "metadata": transcribed["files"]["sources"]["whisper"]["metadata"],
            }

        download = Mock(side_effect=local_audio)
        with (
            patch(
                "podcast_engine.pipeline.file_exists_in_gcs",
                return_value=False,
            ),
            patch("podcast_engine.pipeline.downloaded_audio", download),
            patch(
                "podcast_engine.pipeline.transcribe_audio",
                side_effect=transcribe_local,
            ) as transcribe,
            patch(
                "podcast_engine.pipeline.update_episode",
                side_effect=[downloaded, transcribed],
            ) as update,
        ):
            result = _transcribe_if_needed(episode)

        self.assertEqual(result, transcribed)
        self.assertEqual(download.call_args.args, (episode["audio_url"], "resume-key.mp3"))
        self.assertFalse(transcribe.call_args.args[0].exists())
        self.assertEqual(transcribe.call_args.args[1], "resume-key")
        self.assertEqual(update.call_args_list[0].kwargs, {"download_completed": True})

    def test_older_waiting_episode_resumes_once_after_newer_rss_episode(self):
        latest = self._waiting_for_apple(
            "latest-key",
            "latest-guid",
        )
        older = self._waiting_for_apple(
            "older-key",
            "older-guid",
        )

        with (
            patch(
                "podcast_engine.pipeline.ENABLED_PODCASTS",
                [self.podcast],
            ),
            patch(
                "podcast_engine.pipeline.discover_latest_rss_episode",
                return_value=latest,
            ),
            patch(
                "podcast_engine.pipeline.load_episodes",
                return_value=[
                    latest,
                    older,
                ],
            ),
            patch(
                "podcast_engine.pipeline.resume_tracked_episode",
                side_effect=lambda _podcast, episode: {
                    "status": "waiting_for_apple_transcript",
                    "episode": episode,
                },
            ) as resume,
        ):
            outcomes = run_pipeline()

        self.assertEqual(
            len(outcomes),
            2,
        )
        self.assertEqual(
            [
                call.args[1]["episode_key"]
                for call in resume.call_args_list
            ],
            [
                "latest-key",
                "older-key",
            ],
        )

    def test_completed_older_episode_is_not_resumed(self):
        latest = self._waiting_for_apple(
            "latest-key",
            "latest-guid",
        )
        completed = self._completed(
            "completed-key",
            "completed-guid",
        )

        with (
            patch(
                "podcast_engine.pipeline.ENABLED_PODCASTS",
                [self.podcast],
            ),
            patch(
                "podcast_engine.pipeline.discover_latest_rss_episode",
                return_value=latest,
            ),
            patch(
                "podcast_engine.pipeline.load_episodes",
                return_value=[
                    latest,
                    completed,
                ],
            ),
            patch(
                "podcast_engine.pipeline.resume_tracked_episode",
                return_value={
                    "status": "waiting_for_apple_transcript"
                },
            ) as resume,
        ):
            run_pipeline()

        self.assertEqual(
            resume.call_count,
            1,
        )
        self.assertEqual(
            resume.call_args.args[1]["episode_key"],
            "latest-key",
        )

    def test_pending_review_required_episode_short_circuits_without_recompile(self):
        episode = self._review_required()

        record = {
            "episode_key": episode["episode_key"],
            "human_review": [
                {
                    "id": 1,
                    "reason": "needs_human_review",
                }
            ],
            "human_decisions": [],
        }

        with (
            patch(
                "podcast_engine.pipeline._transcribe_if_needed",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline.load_review_record",
                return_value=record,
            ),
            patch(
                "podcast_engine.pipeline.review_record_is_current",
                return_value=True,
            ),
            patch(
                "podcast_engine.pipeline.compile_episode_sources"
            ) as compile_sources,
            patch(
                "podcast_engine.pipeline.build_knowledge_note"
            ) as knowledge,
            patch(
                "podcast_engine.pipeline._log_human_review_required"
            ) as notify,
        ):
            outcome = resume_tracked_episode(
                self.podcast,
                episode,
            )

        self.assertEqual(
            outcome["status"],
            "review_required",
        )
        self.assertEqual(
            outcome["pending_count"],
            1,
        )
        compile_sources.assert_not_called()
        knowledge.assert_not_called()
        notify.assert_not_called()

    def test_cleared_review_queue_allows_normal_recompile(self):
        episode = self._review_required()

        record = {
            "episode_key": episode["episode_key"],
            "human_review": [],
            "human_decisions": [
                {
                    "id": 1,
                    "chosen_source": "apple",
                    "chosen_text": "resolved",
                    "scope": "full",
                    "reviewed_by": "human",
                }
            ],
        }

        after_compile = deepcopy(
            episode
        )
        after_compile["status"]["compiler"]["state"] = "completed"
        after_summary = deepcopy(after_compile)
        after_summary["status"]["summary"]["state"] = "ready"

        with (
            patch(
                "podcast_engine.pipeline._transcribe_if_needed",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline.load_review_record",
                return_value=record,
            ),
            patch(
                "podcast_engine.pipeline.review_record_is_current",
                return_value=True,
            ),
            patch(
                "podcast_engine.pipeline.compile_episode_sources",
                return_value={
                    "transcript": "compiled.txt",
                    "report": "report.json",
                    "review_required": 0,
                    "review": [],
                },
            ) as compile_sources,
            patch(
                "podcast_engine.pipeline.update_episode",
                side_effect=[after_compile, after_summary],
            ),
            patch(
                "podcast_engine.pipeline.build_knowledge_note"
            ),
            patch(
                "podcast_engine.pipeline.canonical_summary_generated_at",
                return_value="2026-08-30T12:34:56+00:00",
            ),
            patch(
                "podcast_engine.pipeline.send_slack_summary_ready_notification"
            ),
        ):
            outcome = resume_tracked_episode(
                self.podcast,
                episode,
            )

        self.assertEqual(
            outcome["status"],
            "completed",
        )
        compile_sources.assert_called_once_with(
            episode
        )

    def test_ep_387_stale_v2_36_cards_refreshes_to_current_21_cards(self):
        episode = self._review_required("ep-387", "ep-387-guid")
        episode["title"] = "Ep 387"
        record = {
            "episode_key": episode["episode_key"],
            "policy_version": "resolver-policy-v2",
            "human_review": [{"id": item_id} for item_id in range(1, 37)],
            "human_decisions": [],
        }
        refreshed = deepcopy(episode)
        refreshed["status"]["compiler"]["state"] = "review_required"
        refreshed["status"]["compiler"]["review_required"] = 21

        with (
            patch("podcast_engine.pipeline._transcribe_if_needed", return_value=episode),
            patch("podcast_engine.pipeline.load_review_record", return_value=record),
            patch("podcast_engine.pipeline.review_record_is_current", return_value=False),
            patch(
                "podcast_engine.pipeline.compile_episode_sources",
                return_value={
                    "transcript": "compiled.txt",
                    "report": "report.json",
                    "review_required": 21,
                    "review": [{"id": item_id} for item_id in range(1, 22)],
                },
            ) as compile_sources,
            patch("podcast_engine.pipeline.update_episode", return_value=refreshed),
            patch("podcast_engine.pipeline.build_knowledge_note") as knowledge,
            patch("podcast_engine.pipeline._log_human_review_refreshed") as refreshed_event,
            patch("podcast_engine.pipeline.send_slack_review_refreshed_notification") as refreshed_slack,
        ):
            outcome = resume_tracked_episode(self.podcast, episode)

        self.assertEqual(outcome["status"], "review_required")
        self.assertEqual(outcome["review"], [{"id": item_id} for item_id in range(1, 22)])
        compile_sources.assert_called_once_with(episode)
        knowledge.assert_not_called()
        refreshed_event.assert_called_once()
        refreshed_slack.assert_called_once_with(
            refreshed,
            36,
            21,
            "https://review.example.com",
        )

    def test_stale_same_count_with_different_card_content_notifies(self):
        episode = self._review_required()
        record = {
            "episode_key": episode["episode_key"],
            "human_review": [{"id": 7, "apple_text": "old"}],
            "human_decisions": [],
        }
        refreshed = deepcopy(episode)

        with (
            patch("podcast_engine.pipeline._transcribe_if_needed", return_value=episode),
            patch("podcast_engine.pipeline.load_review_record", return_value=record),
            patch("podcast_engine.pipeline.review_record_is_current", return_value=False),
            patch(
                "podcast_engine.pipeline.compile_episode_sources",
                return_value={
                    "transcript": "compiled.txt",
                    "report": "report.json",
                    "review_required": 1,
                    "review": [{"id": 7, "apple_text": "new"}],
                },
            ),
            patch("podcast_engine.pipeline.update_episode", return_value=refreshed),
            patch("podcast_engine.pipeline._log_human_review_refreshed") as refreshed_event,
            patch("podcast_engine.pipeline.send_slack_review_refreshed_notification") as refreshed_slack,
        ):
            outcome = resume_tracked_episode(self.podcast, episode)

        self.assertEqual(outcome["status"], "review_required")
        refreshed_event.assert_called_once()
        refreshed_slack.assert_called_once()

    def test_stale_materially_identical_queue_stays_silent(self):
        episode = self._review_required()
        review = [{"id": 7, "apple_text": "same", "whisper_text": "same"}]
        record = {
            "episode_key": episode["episode_key"],
            "human_review": review,
            "human_decisions": [],
        }
        refreshed = deepcopy(episode)

        with (
            patch("podcast_engine.pipeline._transcribe_if_needed", return_value=episode),
            patch("podcast_engine.pipeline.load_review_record", return_value=record),
            patch("podcast_engine.pipeline.review_record_is_current", return_value=False),
            patch(
                "podcast_engine.pipeline.compile_episode_sources",
                return_value={
                    "transcript": "compiled.txt",
                    "report": "report.json",
                    "review_required": 1,
                    "review": deepcopy(review),
                },
            ),
            patch("podcast_engine.pipeline.update_episode", return_value=refreshed),
            patch("podcast_engine.pipeline._log_human_review_refreshed") as refreshed_event,
            patch("podcast_engine.pipeline.send_slack_review_refreshed_notification") as refreshed_slack,
        ):
            outcome = resume_tracked_episode(self.podcast, episode)

        self.assertEqual(outcome["status"], "review_required")
        refreshed_event.assert_not_called()
        refreshed_slack.assert_not_called()

    def test_stale_queue_recompiled_to_zero_continues_without_notification(self):
        episode = self._review_required()
        record = {
            "episode_key": episode["episode_key"],
            "human_review": [{"id": 7}],
            "human_decisions": [],
        }
        completed = deepcopy(episode)
        completed["status"]["compiler"]["state"] = "completed"
        completed["status"]["summary"]["state"] = "ready"
        after_compile = deepcopy(completed)
        after_compile["status"]["summary"]["state"] = "pending"

        with (
            patch("podcast_engine.pipeline._transcribe_if_needed", return_value=episode),
            patch("podcast_engine.pipeline.load_review_record", return_value=record),
            patch("podcast_engine.pipeline.review_record_is_current", return_value=False),
            patch(
                "podcast_engine.pipeline.compile_episode_sources",
                return_value={
                    "transcript": "compiled.txt",
                    "report": "report.json",
                    "review_required": 0,
                    "review": [],
                },
            ),
            patch(
                "podcast_engine.pipeline.update_episode",
                side_effect=[after_compile, completed],
            ),
            patch("podcast_engine.pipeline.build_knowledge_note"),
            patch(
                "podcast_engine.pipeline.canonical_summary_generated_at",
                return_value="2026-08-30T12:34:56+00:00",
            ),
            patch("podcast_engine.pipeline.send_slack_summary_ready_notification"),
            patch("podcast_engine.pipeline._log_human_review_required") as required_event,
            patch("podcast_engine.pipeline.send_slack_review_notification") as required_slack,
            patch("podcast_engine.pipeline._log_human_review_refreshed") as refreshed_event,
            patch("podcast_engine.pipeline.send_slack_review_refreshed_notification") as refreshed_slack,
        ):
            outcome = resume_tracked_episode(self.podcast, episode)

        self.assertEqual(outcome["status"], "completed")
        required_event.assert_not_called()
        required_slack.assert_not_called()
        refreshed_event.assert_not_called()
        refreshed_slack.assert_not_called()

    def test_current_review_gate_stays_blocked_on_subsequent_scheduler_run(self):
        episode = self._review_required()
        record = {
            "episode_key": episode["episode_key"],
            "human_review": [{"id": 1}],
            "human_decisions": [],
        }

        with (
            patch("podcast_engine.pipeline._transcribe_if_needed", return_value=episode),
            patch("podcast_engine.pipeline.load_review_record", return_value=record),
            patch("podcast_engine.pipeline.review_record_is_current", return_value=True),
            patch("podcast_engine.pipeline.compile_episode_sources") as compile_sources,
            patch("podcast_engine.pipeline._log_human_review_required") as required_event,
            patch("podcast_engine.pipeline.send_slack_review_notification") as required_slack,
            patch("podcast_engine.pipeline._log_human_review_refreshed") as refreshed_event,
            patch("podcast_engine.pipeline.send_slack_review_refreshed_notification") as refreshed_slack,
        ):
            first = resume_tracked_episode(self.podcast, episode)
            second = resume_tracked_episode(self.podcast, episode)

        self.assertEqual(first["status"], "review_required")
        self.assertEqual(second["status"], "review_required")
        compile_sources.assert_not_called()
        required_event.assert_not_called()
        required_slack.assert_not_called()
        refreshed_event.assert_not_called()
        refreshed_slack.assert_not_called()

    def test_missing_review_record_fails_loudly_without_bypassing_human(self):
        episode = self._review_required()

        with (
            patch(
                "podcast_engine.pipeline._transcribe_if_needed",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline.load_review_record",
                side_effect=FileNotFoundError(
                    "missing resolver record"
                ),
            ),
            patch(
                "podcast_engine.pipeline.emit_event"
            ) as emit,
            patch(
                "podcast_engine.pipeline.compile_episode_sources"
            ) as compile_sources,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "durable review record",
            ):
                resume_tracked_episode(
                    self.podcast,
                    episode,
                )

        compile_sources.assert_not_called()

        self.assertEqual(
            emit.call_args.args[0],
            "human_review_state_inconsistent",
        )
        self.assertEqual(
            emit.call_args.kwargs["severity"],
            "ERROR",
        )
        self.assertEqual(
            emit.call_args.kwargs["reason"],
            "FileNotFoundError",
        )

    def test_invalid_review_record_contract_fails_without_bypass(self):
        episode = self._review_required()

        invalid_record = {
            "episode_key": episode["episode_key"],
            "human_review": "not-a-list",
            "human_decisions": [],
        }

        with (
            patch(
                "podcast_engine.pipeline._transcribe_if_needed",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline.load_review_record",
                return_value=invalid_record,
            ),
            patch(
                "podcast_engine.pipeline.emit_event"
            ) as emit,
            patch(
                "podcast_engine.pipeline.compile_episode_sources"
            ) as compile_sources,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "expected contract",
            ):
                resume_tracked_episode(
                    self.podcast,
                    episode,
                )

        compile_sources.assert_not_called()

        self.assertEqual(
            emit.call_args.args[0],
            "human_review_state_inconsistent",
        )
        self.assertEqual(
            emit.call_args.kwargs["severity"],
            "ERROR",
        )
        self.assertEqual(
            emit.call_args.kwargs["reason"],
            "invalid_review_record_contract",
        )

    def test_late_notification_can_fire_for_an_older_tracked_episode(self):
        episode = self._waiting_for_apple(
            "older-key",
            "older-guid",
        )

        marked = deepcopy(
            episode
        )
        marked["apple_transcript_late_notified_at"] = datetime(
            2026,
            8,
            27,
            tzinfo=timezone.utc,
        ).isoformat()

        self.assertTrue(
            _apple_acquisition_due(
                episode,
                now=datetime(
                    2026,
                    8,
                    27,
                    tzinfo=timezone.utc,
                ),
            )
        )

        with (
            patch(
                "podcast_engine.pipeline._transcribe_if_needed",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline.update_episode",
                return_value=marked,
            ),
            patch(
                "podcast_engine.pipeline.acquire_apple_transcript",
                return_value={
                    "status": "TRANSCRIPT_NOT_READY"
                },
            ),
            patch(
                "podcast_engine.pipeline.emit_event"
            ) as emit,
        ):
            outcome = resume_tracked_episode(
                self.podcast,
                episode,
            )

        self.assertEqual(
            outcome["status"],
            "waiting_for_apple_transcript",
        )
        self.assertEqual(
            outcome["reason"],
            "not_ready",
        )
        self.assertEqual(
            emit.call_args.args[0],
            "apple_transcript_late",
        )


if __name__ == "__main__":
    unittest.main()
