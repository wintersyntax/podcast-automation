from __future__ import annotations

import unittest
from unittest.mock import ANY, patch

from podcast_engine.ai_budget import (
    BudgetAdmissionError,
    BudgetIntegrityError,
    DownstreamReserveError,
    BudgetConcurrencyError,
)
from podcast_engine.human_review import PrepareSessionConflict, PrepareSessionError, ThirdAsrInFlight
from podcast_engine.review_web import PAGE, create_review_app


def _record(
    fingerprint: str = "sha256:" + "a" * 64,
    source_fingerprint: str = "sha256:" + "b" * 64,
) -> dict:
    return {
        "episode_key": "episode-1",
        "input_fingerprint": fingerprint,
        "source_fingerprint": source_fingerprint,
        "human_review": [{"id": 5}],
    }


def _budget() -> dict:
    from decimal import Decimal

    return {
        "settled_spend": Decimal("1.50"),
        "uncertain_spend": Decimal("0"),
        "live_reservations": Decimal("0.25"),
        "third_asr_settled": Decimal("1.50"),
        "third_asr_uncertain": Decimal("0"),
        "third_asr_live_reservations": Decimal("0.25"),
        "hard_cap_usd": Decimal("50"),
        "third_asr_subcap_usd": Decimal("20"),
        "headroom_usd": Decimal("48.25"),
        "third_asr_headroom_usd": Decimal("18.25"),
    }


class AssistedPrepareWebTests(unittest.TestCase):
    def test_assisted_prepare_delegates_and_returns_session_and_budget(self):
        session = {"session_id": "session-1", "items": {"5": {"state": "queued"}}}

        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch(
                "podcast_engine.review_web.begin_assisted_preparation",
                return_value=session,
            ) as begin,
            patch("podcast_engine.review_web.budget_summary", return_value=_budget()) as summary,
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/prepare",
                json={
                    "selected_ids": [5],
                    "expected_generation_fingerprint": "sha256:generation",
                    "session_id": "session-1",
                },
            )

        self.assertEqual(response.status_code, 201)
        begin.assert_called_once_with(
            "episode-1",
            "sha256:" + "b" * 64,
            [5],
            expected_review_generation_fingerprint="sha256:generation",
            session_id="session-1",
        )
        summary.assert_called_once_with("episode-1", "sha256:" + "b" * 64)
        body = response.get_json()
        self.assertEqual(body["session"], session)
        self.assertEqual(body["budget"]["settled_spend"], 1.50)
        self.assertIsInstance(body["budget"]["settled_spend"], float)

    def test_assisted_prepare_budget_uses_real_source_generation_ledger(self):
        from decimal import Decimal
        from podcast_engine.ai_budget import budget_summary, reserve_budget_batch
        from podcast_engine.human_review import ensure_third_asr
        from tests.test_ai_budget import _FakeBucket

        episode_key = 'a' * 24
        input_fingerprint = 'sha256:' + 'c' * 64
        source_fingerprint = 'sha256:' + 'd' * 64
        record = _record(fingerprint=input_fingerprint, source_fingerprint=source_fingerprint)
        record['episode_key'] = episode_key
        store = {}
        with patch('podcast_engine.ai_budget.get_bucket', return_value=_FakeBucket(store)):
            reserve_budget_batch(episode_key, source_fingerprint, [
                {'attempt_id': 'source-reservation', 'stage': 'resolver', 'reserved_usd': Decimal('0.18')}
            ])
            reserve_budget_batch(episode_key, input_fingerprint, [
                {'attempt_id': 'input-reservation', 'stage': 'resolver', 'reserved_usd': Decimal('0.04')}
            ])
            self.assertEqual(budget_summary(episode_key, source_fingerprint)['headroom_usd'], Decimal('1.42'))
            with patch('podcast_engine.review_web.load_review_record', return_value=record), patch(
                'podcast_engine.review_web.begin_assisted_preparation', return_value={'session_id': 's', 'items': {}}
            ) as begin:
                response = create_review_app().test_client().post(
                    f'/api/review/episodes/{episode_key}/assisted/prepare',
                    json={'selected_ids': [5], 'expected_generation_fingerprint': 'sha256:generation'},
                )
        self.assertEqual(response.status_code, 201)
        begin.assert_called_once_with(
            episode_key, source_fingerprint, [5],
            expected_review_generation_fingerprint='sha256:generation', session_id=None,
        )
        budget = response.get_json()['budget']
        self.assertEqual(budget['live_reservations'], 0.18)
        self.assertEqual(budget['headroom_usd'], 1.42)
        with patch('podcast_engine.human_review.load_review_record_with_generation', return_value=(record, 1)), patch(
            'podcast_engine.human_review.review_clip_window', return_value={'duration': 1}
        ), patch('podcast_engine.human_review.third_asr_cache_key', return_value='uncached'), patch(
            'podcast_engine.human_review.ensure_fresh_third_asr_budget_identity', side_effect=ValueError('gate probe')
        ) as gate:
            with self.assertRaisesRegex(ValueError, 'gate probe'):
                ensure_third_asr({'episode_key': episode_key}, 5)
        gate.assert_called_once_with(episode_key, source_fingerprint, ANY)

    def test_assisted_prepare_defaults_session_id_to_none(self):
        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch(
                "podcast_engine.review_web.begin_assisted_preparation",
                return_value={"session_id": "auto", "items": {}},
            ) as begin,
            patch("podcast_engine.review_web.budget_summary", return_value=_budget()),
        ):
            create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/prepare",
                json={
                    "selected_ids": [5],
                    "expected_generation_fingerprint": "sha256:generation",
                },
            )

        begin.assert_called_once_with(
            "episode-1",
            "sha256:" + "b" * 64,
            [5],
            expected_review_generation_fingerprint="sha256:generation",
            session_id=None,
        )

    def test_assisted_prepare_requires_selected_ids_and_generation(self):
        invalid_bodies = (
            {},
            {"selected_ids": [5]},
            {"expected_generation_fingerprint": "sha256:generation"},
            {"selected_ids": [], "expected_generation_fingerprint": "sha256:generation"},
            {"selected_ids": [5], "expected_generation_fingerprint": ""},
            {"selected_ids": [5], "expected_generation_fingerprint": "sha256:generation", "session_id": ""},
        )

        for body in invalid_bodies:
            with self.subTest(body=body), patch(
                "podcast_engine.review_web.begin_assisted_preparation",
            ) as begin:
                response = create_review_app().test_client().post(
                    "/api/review/episodes/episode-1/assisted/prepare",
                    json=body,
                )

            self.assertEqual(response.status_code, 400)
            begin.assert_not_called()

    def test_assisted_prepare_maps_budget_errors_to_402(self):
        for error in (
            BudgetAdmissionError("cap exceeded"),
            BudgetIntegrityError("ledger corrupt"),
            DownstreamReserveError("downstream reserve failed"),
        ):
            with self.subTest(error=type(error).__name__), (
                patch("podcast_engine.review_web.load_review_record", return_value=_record())
            ), patch(
                "podcast_engine.review_web.begin_assisted_preparation", side_effect=error
            ):
                response = create_review_app().test_client().post(
                    "/api/review/episodes/episode-1/assisted/prepare",
                    json={
                        "selected_ids": [5],
                        "expected_generation_fingerprint": "sha256:generation",
                    },
                )

            self.assertEqual(response.status_code, 402)

    def test_assisted_prepare_maps_conflict_errors_to_409(self):
        for error in (
            PrepareSessionError("session busy"),
            PrepareSessionConflict("session CAS exhausted"),
            BudgetConcurrencyError("ledger changed"),
            ValueError("stale generation"),
        ):
            with self.subTest(error=type(error).__name__), (
                patch("podcast_engine.review_web.load_review_record", return_value=_record())
            ), patch(
                "podcast_engine.review_web.begin_assisted_preparation", side_effect=error
            ):
                response = create_review_app().test_client().post(
                    "/api/review/episodes/episode-1/assisted/prepare",
                    json={
                        "selected_ids": [5],
                        "expected_generation_fingerprint": "sha256:generation",
                    },
                )

            self.assertEqual(response.status_code, 409)

    def test_review_get_counts_only_pending_eligible_unprepared_without_spend(self):
        from tests.test_assisted_review_projection import _eligible_item, _protected_item

        record = {
            'human_review': [
                _eligible_item(1),
                _eligible_item(2, third_asr={'status': 'processing'}),
                _eligible_item(3, third_asr={'text': 'the quick brown fox'}),
                _protected_item(4),
                _eligible_item(5, third_asr={'status': 'failed'}),
            ],
            'human_decisions': [{'id': 6}],
        }
        with (
            patch('podcast_engine.review_web.load_review_record', return_value=record),
            patch('podcast_engine.review_web.recompile_status_for_record', return_value=None),
            patch('podcast_engine.review_web.begin_assisted_preparation') as prepare,
            patch('podcast_engine.review_web.ensure_third_asr') as third,
            patch('podcast_engine.review_web.budget_summary') as budget,
        ):
            response = create_review_app().test_client().get('/api/review/episodes/episode-1')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['progress']['total'], 6)
        self.assertEqual(body['progress']['reviewed'], 1)
        self.assertEqual(body['progress']['remaining'], 5)
        self.assertEqual(body['progress'].get('assisted_unprepared'), 2, 'eligible pending evidence count must be 2')
        self.assertEqual([c['assisted_review']['state'] for c in body['cards'][:2]], ['audio_pending', 'audio_pending'])
        self.assertFalse(body['cards'][3]['assisted_review']['routing']['eligible'])
        self.assertEqual(body['cards'][4]['assisted_review']['state'], 'audio_unavailable')
        prepare.assert_not_called()
        third.assert_not_called()
        budget.assert_not_called()


class AssistedPrepareStatusWebTests(unittest.TestCase):
    def test_assisted_prepare_status_returns_session_and_budget(self):
        session = {"session_id": "session-1", "items": {"5": {"state": "prepared"}}}
        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch("podcast_engine.review_web.load_prepare_session", return_value=session) as load,
            patch("podcast_engine.review_web.budget_summary", return_value=_budget()),
        ):
            response = create_review_app().test_client().get(
                "/api/review/episodes/episode-1/assisted/prepare/session-1"
            )

        self.assertEqual(response.status_code, 200)
        load.assert_called_once_with("episode-1", "sha256:" + "b" * 64, "session-1")
        self.assertEqual(response.get_json()["session"], session)

    def test_assisted_prepare_status_404_when_missing(self):
        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch("podcast_engine.review_web.load_prepare_session", return_value=None),
            patch("podcast_engine.review_web.budget_summary", return_value=_budget()),
        ):
            response = create_review_app().test_client().get(
                "/api/review/episodes/episode-1/assisted/prepare/missing-session"
            )

        self.assertEqual(response.status_code, 404)


class AssistedPrepareItemWebTests(unittest.TestCase):
    def test_idempotent_replay_when_already_prepared(self):
        session_item = {"state": "prepared", "cache_key": "sha256:" + "d" * 64}
        session = {"session_id": "session-1", "items": {"5": session_item}}

        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch("podcast_engine.review_web.load_prepare_session", return_value=session),
            patch("podcast_engine.review_web.advance_prepare_session_item") as advance,
            patch("podcast_engine.review_web.release_budget_attempt_pre_send") as release,
            patch("podcast_engine.review_web.ensure_third_asr") as third,
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/prepare/session-1/items/5",
                json={"attempt_id": "prepare-session-1-1-5"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "prepared", "item": session_item})
        advance.assert_not_called()
        release.assert_not_called()
        third.assert_not_called()

    def test_404_when_session_missing(self):
        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch("podcast_engine.review_web.load_prepare_session", return_value=None),
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/prepare/missing/items/5",
                json={"attempt_id": "prepare-missing-1-5"},
            )

        self.assertEqual(response.status_code, 404)

    def test_404_when_item_not_in_session(self):
        session = {"session_id": "session-1", "items": {"6": {"state": "queued"}}}
        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch("podcast_engine.review_web.load_prepare_session", return_value=session),
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/prepare/session-1/items/5",
                json={"attempt_id": "prepare-session-1-1-5"},
            )

        self.assertEqual(response.status_code, 404)

    def test_executes_and_releases_reservation_before_delegating_to_ensure_third_asr(self):
        session = {"session_id": "session-1", "items": {"5": {"state": "queued"}}}
        episode = {"episode_key": "episode-1", "title": "Episode One"}
        evidence = {"text": "third source evidence", "status": "complete"}
        calls = []

        def _record_advance(*args, **kwargs):
            calls.append((args, kwargs))
            return {"state": kwargs["state"]}

        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch("podcast_engine.review_web.load_prepare_session", return_value=session),
            patch(
                "podcast_engine.review_web.advance_prepare_session_item",
                side_effect=_record_advance,
            ) as advance,
            patch(
                "podcast_engine.review_web.release_budget_attempt_pre_send"
            ) as release,
            patch("podcast_engine.review_web.load_episodes", return_value=[episode]),
            patch(
                "podcast_engine.review_web.ensure_third_asr", return_value=evidence
            ) as third,
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/prepare/session-1/items/5",
                json={"attempt_id": "prepare-session-1-1-5"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "prepared", "evidence": evidence})

        # First transition claims the item (in_flight) BEFORE the reservation
        # is released or the provider call happens; second transition records
        # completion (prepared) only after ensure_third_asr succeeds.
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["state"], "in_flight")
        self.assertEqual(calls[1][1]["state"], "prepared")
        for args, kwargs in calls:
            self.assertEqual(args, ("episode-1", "sha256:" + "b" * 64, "session-1", 5))
            self.assertEqual(kwargs["attempt_id"], "prepare-session-1-1-5")

        release.assert_called_once_with(
            "episode-1", "sha256:" + "b" * 64, "prepare-session-1-1-5"
        )
        third.assert_called_once_with(episode, 5, prepare_session_id="session-1")

    def test_release_value_error_is_swallowed_as_idempotent(self):
        # An already-released or unknown attempt_id (a replay after an
        # earlier partial completion) must not block delegating to
        # ensure_third_asr -- the release is a best-effort idempotent step.
        session = {"session_id": "session-1", "items": {"5": {"state": "queued"}}}
        episode = {"episode_key": "episode-1"}
        evidence = {"text": "evidence", "status": "complete"}

        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch("podcast_engine.review_web.load_prepare_session", return_value=session),
            patch("podcast_engine.review_web.advance_prepare_session_item"),
            patch(
                "podcast_engine.review_web.release_budget_attempt_pre_send",
                side_effect=ValueError("Cannot pre-send release attempt in state 'released'"),
            ),
            patch("podcast_engine.review_web.load_episodes", return_value=[episode]),
            patch("podcast_engine.review_web.ensure_third_asr", return_value=evidence) as third,
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/prepare/session-1/items/5",
                json={"attempt_id": "prepare-session-1-1-5"},
            )

        self.assertEqual(response.status_code, 200)
        third.assert_called_once()

    def test_third_asr_in_flight_moves_item_to_retrying_and_returns_202(self):
        session = {"session_id": "session-1", "items": {"5": {"state": "queued"}}}
        episode = {"episode_key": "episode-1"}
        in_flight = ThirdAsrInFlight(retry_after_seconds=12.5, cache_key="sha256:" + "e" * 64)
        advance_calls = []

        def _record_advance(*args, **kwargs):
            advance_calls.append(kwargs["state"])

        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch("podcast_engine.review_web.load_prepare_session", return_value=session),
            patch(
                "podcast_engine.review_web.advance_prepare_session_item",
                side_effect=_record_advance,
            ),
            patch("podcast_engine.review_web.release_budget_attempt_pre_send"),
            patch("podcast_engine.review_web.load_episodes", return_value=[episode]),
            patch("podcast_engine.review_web.ensure_third_asr", side_effect=in_flight),
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/prepare/session-1/items/5",
                json={"attempt_id": "prepare-session-1-1-5"},
            )

        self.assertEqual(response.status_code, 202)
        body = response.get_json()
        self.assertEqual(body["status"], "in_flight")
        self.assertAlmostEqual(body["retry_after_seconds"], 12.5)
        self.assertEqual(advance_calls, ["in_flight", "retrying"])

    def test_failure_marks_item_failed_and_returns_bounded_error(self):
        session = {"session_id": "session-1", "items": {"5": {"state": "queued"}}}
        episode = {"episode_key": "episode-1"}
        advance_calls = []

        def _record_advance(*args, **kwargs):
            advance_calls.append(kwargs["state"])

        with (
            patch("podcast_engine.review_web.load_review_record", return_value=_record()),
            patch("podcast_engine.review_web.load_prepare_session", return_value=session),
            patch(
                "podcast_engine.review_web.advance_prepare_session_item",
                side_effect=_record_advance,
            ),
            patch("podcast_engine.review_web.release_budget_attempt_pre_send"),
            patch("podcast_engine.review_web.load_episodes", return_value=[episode]),
            patch(
                "podcast_engine.review_web.ensure_third_asr",
                side_effect=RuntimeError("provider exploded"),
            ),
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/prepare/session-1/items/5",
                json={"attempt_id": "prepare-session-1-1-5"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["status"], "failed")
        self.assertEqual(advance_calls, ["in_flight", "failed"])


class AssistedDecisionsWebTests(unittest.TestCase):
    def test_delegates_to_generation_safe_backend(self):
        updated = {"human_review": []}
        decisions = [{"id": 5, "source": "apple"}]

        with (
            patch(
                "podcast_engine.review_web.record_assisted_human_decision_batch",
                return_value=updated,
            ) as batch,
            patch("podcast_engine.review_web.pending_review_items", return_value=[]),
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/decisions",
                json={
                    "decisions": decisions,
                    "expected_generation_fingerprint": "sha256:generation",
                    "assisted_policy_version": "human-review-assisted-v1",
                },
            )

        self.assertEqual(response.status_code, 200)
        batch.assert_called_once_with(
            "episode-1",
            decisions,
            expected_generation_fingerprint="sha256:generation",
            assisted_policy_version="human-review-assisted-v1",
        )
        self.assertEqual(
            response.get_json(),
            {"accepted_count": 1, "ready_to_recompile": True},
        )

    def test_requires_decisions_generation_and_policy_version(self):
        invalid_bodies = (
            {},
            {"decisions": [{"id": 5, "source": "apple"}]},
            {
                "decisions": [{"id": 5, "source": "apple"}],
                "expected_generation_fingerprint": "sha256:generation",
            },
            {
                "decisions": [],
                "expected_generation_fingerprint": "sha256:generation",
                "assisted_policy_version": "human-review-assisted-v1",
            },
            {
                "decisions": [{"id": 5, "source": "apple"}],
                "expected_generation_fingerprint": "",
                "assisted_policy_version": "human-review-assisted-v1",
            },
        )

        for body in invalid_bodies:
            with self.subTest(body=body), patch(
                "podcast_engine.review_web.record_assisted_human_decision_batch",
            ) as batch:
                response = create_review_app().test_client().post(
                    "/api/review/episodes/episode-1/assisted/decisions",
                    json=body,
                )

            self.assertEqual(response.status_code, 400)
            batch.assert_not_called()

    def test_returns_conflict_on_value_error(self):
        with patch(
            "podcast_engine.review_web.record_assisted_human_decision_batch",
            side_effect=ValueError("Review generation changed; reload before Assisted decisions"),
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/assisted/decisions",
                json={
                    "decisions": [{"id": 5, "source": "apple"}],
                    "expected_generation_fingerprint": "sha256:stale",
                    "assisted_policy_version": "human-review-assisted-v1",
                },
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("generation", response.get_json()["error"].casefold())


class AssistedReviewPageContractTests(unittest.TestCase):
    def test_assisted_panel_exposes_three_distinct_lanes_separate_from_batch(self):
        self.assertIn("Assisted review", PAGE)
        self.assertIn("id=assisted-panel", PAGE)
        self.assertIn("Machine-supported", PAGE)
        self.assertIn("Conflicting evidence", PAGE)
        self.assertIn("Insufficient / unavailable evidence", PAGE)

        # Batch approval is untouched and stays a fully separate panel.
        self.assertIn("id=batch-actions", PAGE)
        self.assertIn("Accept recommended batch", PAGE)
        self.assertIn("id=accept-recommended-batch", PAGE)

    def test_conflict_and_insufficient_lanes_are_never_preselected(self):
        # Only the machine-supported lane defaults a row's decision to the
        # derived recommendation; conflict/insufficient rows stay Defer
        # until the reviewer explicitly picks Apple or Whisper.
        self.assertIn(
            "byLane.machine.forEach(c=>{if(!assistedTouched[c.id]){assistedDecisions[c.id]=c.assisted_review.recommendation;assistedTouched[c.id]=true}})",
            PAGE,
        )
        self.assertNotIn("byLane.conflict.forEach(c=>{if(!assistedTouched", PAGE)
        self.assertNotIn("byLane.insufficient.forEach(c=>{if(!assistedTouched", PAGE)
        self.assertIn("Defer", PAGE)

    def test_pending_heading_uses_total_queue_not_reviewed_count(self):
        # The Needs evidence heading must display items as a subset of
        # progress.total (total review items Y), not progress.reviewed
        # (already-reviewed X). This ensures clarity: N pending is part of
        # Y total, not a subset of X reviewed.
        self.assertIn(
            "Needs evidence (${items.length} of ${progress.total} total review items)",
            PAGE,
        )
        self.assertNotIn(
            "Needs evidence (${items.length} of ${progress.reviewed}",
            PAGE,
        )

    def test_no_global_all_apple_or_all_whisper_assisted_action(self):
        self.assertNotIn("assisted-apply-all", PAGE)
        self.assertNotIn("Accept all Apple", PAGE)
        self.assertNotIn("Accept all Whisper", PAGE)
        self.assertNotIn("All Apple", PAGE)
        self.assertNotIn("All Whisper", PAGE)

    def test_row_confirmation_shows_exact_selected_count(self):
        self.assertIn("Confirm ${n} human decision${n===1?'':'s'}", PAGE)
        self.assertIn("id=assisted-confirm", PAGE)

    def test_prepare_scheduler_caps_concurrency_at_four(self):
        self.assertIn("const concurrency=4", PAGE)
        self.assertIn("/assisted/prepare", PAGE)
        self.assertIn("/assisted/decisions", PAGE)
        self.assertIn("assisted_policy_version", PAGE)

    def test_row_evidence_bundle_is_displayed(self):
        # Apple/Whisper/Third ASR text, match evidence, triage, compiler
        # suggestion, machine state, and routing reason codes per row.
        self.assertIn("Match: ${matchText}", PAGE)
        self.assertIn("Triage: ${triage}", PAGE)
        self.assertIn("Compiler suggestion: ${suggestion}", PAGE)
        self.assertIn("Routing: ${routing}", PAGE)
        self.assertIn("assistedStateLabel", PAGE)

    def test_served_assisted_budget_has_adjacent_semantic_help(self):
        from html.parser import HTMLParser

        class BudgetHelp(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_budget = False
                self.after_budget = False
                self.collect = False
                self.help_text = None

            def handle_starttag(self, tag, attrs):
                attributes = dict(attrs)
                if tag == 'p' and self.after_budget:
                    self.after_budget = False
                    if attributes.get('id') == 'assisted-budget-help':
                        self.help_text = ''
                        self.collect = True
                if tag == 'p' and attributes.get('id') == 'assisted-budget':
                    self.in_budget = True

            def handle_endtag(self, tag):
                if tag == 'p' and self.in_budget:
                    self.in_budget = False
                    self.after_budget = True
                elif tag == 'p' and self.collect:
                    self.collect = False

            def handle_data(self, data):
                if self.collect:
                    self.help_text += data

        response = create_review_app().test_client().get('/')
        self.assertEqual(response.status_code, 200)
        parsed = BudgetHelp()
        parsed.feed(response.get_data(as_text=True))
        expected = (
            "These figures cover metered AI attempts for this episode's Apple/Whisper source generation, not just this review session. "
            "Settled is recorded actual cost; reserved is an outstanding upper-bound hold, including queued preparation, not confirmed spend. "
            "Uncertain retains the reserved amount for unverified cost, ambiguous post-send outcomes, over-reservation integrity failure or reconciled legacy Third-ASR; proven pre-send releases count zero. "
            "Remaining is the $1.60 total cap less settled, reserved and uncertain; Third-ASR remaining is the shared $0.10 sub-cap less those Third-ASR amounts, not extra budget. "
            "These are ledger headroom, not approval for another paid call: downstream reserves can restrict optional Third-ASR, identity reconciliation gates Third-ASR, and integrity failure blocks new reservations. "
            "Selected max cost adds this session's quoted per-item maximums, even after a hold is released; session expires marks the preparation lease, not a ledger reset."
        )
        self.assertEqual(parsed.help_text, expected, 'missing budget help beside #assisted-budget: ' + expected)

    def test_normal_review_serves_count_banner_outside_assisted_panel(self):
        response = create_review_app().test_client().get('/')
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn('id=assisted-availability', page, 'normal review count banner is missing')
        self.assertLess(page.index('id=assisted-availability'), page.index('id=assisted-panel'))
        self.assertLess(page.index('id=assisted-availability'), page.index('id=app'))
        self.assertIn('progress.assisted_unprepared', page)
        self.assertIn('Assisted evidence not yet prepared for eligible pending cards:', page)


if __name__ == "__main__":
    unittest.main()
