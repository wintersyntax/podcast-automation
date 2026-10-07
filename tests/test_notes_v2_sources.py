"""RED/GREEN tests for TASK-106 Phase A Task 6: source resolution.

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§5.3 (source resolution).

Covers both `podcast_engine.knowledge.notes_v2.sources` (pure decision
logic: show-notes fuzzy match, bibliographic candidate selection, override
precedence) and `scripts.knowledge_eval.lookup` (HTTP-facing Crossref/
OpenAlex fetch, on-disk cache, override-file loading), per the plan's single
combined `tests/test_notes_v2_sources.py` for Task 6.

The plan asks for "fixture responses and cases from the existing notes
(e.g. 'Gorgie Nijad', 'Nikolaitis', 'MASS')" -- illustrative ASR-mangled
proper nouns that must fuzzy-match their correctly-spelled show-notes
counterparts. The exact production notes live outside this repository
(GCS/Obsidian vault), so the fixtures below are same-character synthetic
cases built in that spirit, not verbatim production text.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

from podcast_engine.knowledge.notes_v2.sources import (
    STATUS_AS_HEARD,
    STATUS_AUTO_MATCHED,
    STATUS_CORRECTED,
    STATUS_SHOW_NOTES,
    extract_surnames,
    extract_topic_words,
    match_show_notes,
    quote_hash,
    resolve_source_mention,
    select_bibliographic_candidate,
)
from scripts.knowledge_eval.lookup import SourceLookupClient, load_overrides


def _mention(**overrides) -> dict:
    mention = {
        "authors_as_heard": "Nikolaidis",
        "year_as_heard": 2019,
        "title_as_heard": "powerlifting training volume",
        "type": "study",
    }
    mention.update(overrides)
    return mention


def _candidate(**overrides) -> dict:
    candidate = {
        "title": "Training volume and powerlifting performance",
        "url": "https://doi.org/10.1234/abc",
        "doi": "10.1234/abc",
        "year": 2019,
        "authors": ["nikolaidis"],
        "score": 0.9,
        "provider": "crossref",
    }
    candidate.update(overrides)
    return candidate


class ExtractSurnamesTests(unittest.TestCase):
    def test_a_single_author_returns_one_surname(self):
        self.assertEqual(extract_surnames("Nikolaidis"), ["nikolaidis"])

    def test_multiple_authors_separated_by_and(self):
        self.assertEqual(extract_surnames("Smith and Jones"), ["smith", "jones"])

    def test_et_al_is_dropped(self):
        self.assertEqual(extract_surnames("Helms et al."), ["helms"])

    def test_none_returns_empty_list(self):
        self.assertEqual(extract_surnames(None), [])


class ExtractTopicWordsTests(unittest.TestCase):
    def test_stopwords_and_short_words_are_dropped(self):
        words = extract_topic_words("The effect of training volume on the powerlifting total")
        self.assertIn("training", words)
        self.assertIn("volume", words)
        self.assertIn("powerlifting", words)
        self.assertNotIn("the", words)
        self.assertNotIn("of", words)
        self.assertNotIn("on", words)

    def test_none_returns_empty_list(self):
        self.assertEqual(extract_topic_words(None), [])


class QuoteHashTests(unittest.TestCase):
    def test_stable_for_identical_quote(self):
        self.assertEqual(quote_hash("a quote"), quote_hash("a quote"))

    def test_differs_for_different_quotes(self):
        self.assertNotEqual(quote_hash("a quote"), quote_hash("a different quote"))


class ShowNotesMatchTests(unittest.TestCase):
    def test_a_close_match_above_threshold_wins(self):
        # "Nikolaitis" (as heard, mangled) should fuzzy-match "Nikolaidis"
        # (correctly spelled in the show notes).
        mention = _mention(authors_as_heard="Nikolaitis", year_as_heard=2019, title_as_heard="powerlifting training volume")
        description = (
            "Show notes:\n"
            "Nikolaidis PT (2019) - Effects of training volume in powerlifting - "
            "https://doi.org/10.1234/real-paper\n"
        )
        result = match_show_notes(mention, description)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_SHOW_NOTES)
        self.assertIn("Nikolaidis", result.title)
        self.assertEqual(result.url, "https://doi.org/10.1234/real-paper")

    def test_a_second_mangled_name_case_gorgie_nijad(self):
        # "Gorgie Nijad" (as heard) vs. the correctly-spelled show-notes form.
        mention = _mention(authors_as_heard="Gorgie Nijad", year_as_heard=2021, title_as_heard="protein timing")
        description = "Referenced: Georgie Najjar (2021) on protein timing for hypertrophy.\n"
        result = match_show_notes(mention, description)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_SHOW_NOTES)

    def test_the_mass_review_abbreviation_case(self):
        # "MASS" (Monthly Applications in Strength Sport) as heard, matched
        # by its abbreviation appearing verbatim in the show notes.
        mention = _mention(authors_as_heard=None, year_as_heard=None, title_as_heard="the MASS review")
        description = "Referenced this month's MASS (Monthly Applications in Strength Sport) review.\n"
        result = match_show_notes(mention, description)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_SHOW_NOTES)

    def test_no_match_below_threshold_returns_none(self):
        mention = _mention(authors_as_heard="Totally Unrelated Author", year_as_heard=1999, title_as_heard="something else entirely")
        description = "Show notes:\nSchoenfeld B (2021) - Hypertrophy training review.\n"
        self.assertIsNone(match_show_notes(mention, description))

    def test_empty_description_returns_none(self):
        self.assertIsNone(match_show_notes(_mention(), None))
        self.assertIsNone(match_show_notes(_mention(), ""))

    def test_empty_source_mention_fields_return_none(self):
        mention = _mention(authors_as_heard=None, year_as_heard=None, title_as_heard=None)
        self.assertIsNone(match_show_notes(mention, "some description text"))


class BibliographicCandidateSelectionTests(unittest.TestCase):
    def test_a_unique_qualifying_candidate_is_accepted(self):
        mention = _mention()
        candidates = [_candidate()]
        result = select_bibliographic_candidate(mention, candidates)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_AUTO_MATCHED)
        self.assertEqual(result.doi, "10.1234/abc")

    def test_two_distinct_qualifying_candidates_is_rejected_as_not_unique(self):
        mention = _mention()
        candidates = [
            _candidate(doi="10.1/one", url="https://doi.org/10.1/one"),
            _candidate(doi="10.1/two", url="https://doi.org/10.1/two"),
        ]
        self.assertIsNone(select_bibliographic_candidate(mention, candidates))

    def test_year_outside_tolerance_is_rejected(self):
        mention = _mention(year_as_heard=2019)
        candidates = [_candidate(year=2023)]
        self.assertIsNone(select_bibliographic_candidate(mention, candidates))

    def test_year_within_tolerance_is_accepted(self):
        mention = _mention(year_as_heard=2019)
        candidates = [_candidate(year=2020)]
        result = select_bibliographic_candidate(mention, candidates)
        self.assertIsNotNone(result)

    def test_no_author_surname_match_is_rejected(self):
        mention = _mention(authors_as_heard="Totally Different Name")
        candidates = [_candidate(authors=["nikolaidis"])]
        self.assertIsNone(select_bibliographic_candidate(mention, candidates))

    def test_below_threshold_is_rejected(self):
        mention = _mention()
        candidates = [_candidate(score=0.1)]
        self.assertIsNone(select_bibliographic_candidate(mention, candidates))

    def test_no_candidates_is_rejected(self):
        self.assertIsNone(select_bibliographic_candidate(_mention(), []))

    def test_duplicate_doi_across_providers_counts_as_one_candidate(self):
        mention = _mention()
        candidates = [
            _candidate(provider="crossref"),
            _candidate(provider="openalex"),  # same DOI as the crossref one
        ]
        result = select_bibliographic_candidate(mention, candidates)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_AUTO_MATCHED)

    def test_missing_heard_year_does_not_reject_a_candidate(self):
        mention = _mention(year_as_heard=None)
        candidates = [_candidate(year=2019)]
        self.assertIsNotNone(select_bibliographic_candidate(mention, candidates))

    def test_missing_heard_authors_does_not_reject_a_candidate(self):
        mention = _mention(authors_as_heard=None)
        candidates = [_candidate()]
        self.assertIsNotNone(select_bibliographic_candidate(mention, candidates))


class ResolveSourceMentionPrecedenceTests(unittest.TestCase):
    def test_override_wins_over_a_show_notes_match(self):
        mention = _mention(authors_as_heard="Nikolaitis")
        description = "Nikolaidis PT (2019) - Effects of training volume - https://doi.org/10.1234/real-paper\n"
        overrides = {
            "ep-1": {
                quote_hash("the quote text"): {
                    "title": "Corrected Title",
                    "url": "https://example.com/corrected",
                    "doi": "10.9999/corrected",
                }
            }
        }
        result = resolve_source_mention(
            mention, quote="the quote text", episode_key="ep-1",
            description=description, bibliographic_candidates=[_candidate()],
            overrides=overrides,
        )
        self.assertEqual(result.status, STATUS_CORRECTED)
        self.assertEqual(result.title, "Corrected Title")

    def test_show_notes_wins_over_bibliographic_when_both_would_match(self):
        mention = _mention()  # "Nikolaidis", 2019 -- matches _candidate() exactly too
        description = "Nikolaidis PT (2019) - Effects of training volume - https://doi.org/10.1234/real-paper\n"
        result = resolve_source_mention(
            mention, quote="the quote text", episode_key="ep-1",
            description=description, bibliographic_candidates=[_candidate()],
        )
        # Both would independently resolve (show notes fuzzy-matches this
        # line; the bibliographic candidate also qualifies) -- show notes
        # takes precedence per design spec §5.3's stated order.
        self.assertEqual(result.status, STATUS_SHOW_NOTES)

    def test_bibliographic_used_when_no_show_notes_match(self):
        mention = _mention()
        result = resolve_source_mention(
            mention, quote="the quote text", episode_key="ep-1",
            description="Nothing relevant here.", bibliographic_candidates=[_candidate()],
        )
        self.assertEqual(result.status, STATUS_AUTO_MATCHED)

    def test_as_heard_when_nothing_matches(self):
        mention = _mention()
        result = resolve_source_mention(
            mention, quote="the quote text", episode_key="ep-1",
            description="Nothing relevant here.", bibliographic_candidates=[],
        )
        self.assertEqual(result.status, STATUS_AS_HEARD)
        self.assertEqual(result.title, mention["title_as_heard"])

    def test_empty_bibliographic_candidates_degrades_to_as_heard(self):
        # Simulates the caller having already degraded an HTTP failure to [].
        mention = _mention()
        result = resolve_source_mention(
            mention, quote="the quote text", episode_key="ep-1",
            description=None, bibliographic_candidates=[],
        )
        self.assertEqual(result.status, STATUS_AS_HEARD)

    def test_missing_episode_key_in_overrides_does_not_raise(self):
        result = resolve_source_mention(
            _mention(), quote="the quote text", episode_key="ep-unknown",
            description=None, bibliographic_candidates=[], overrides={"ep-1": {}},
        )
        self.assertEqual(result.status, STATUS_AS_HEARD)


def _ok_transport(payload):
    def transport(query, *, year, contact_email):
        return payload

    return transport


def _raising_transport(exc):
    def transport(query, *, year, contact_email):
        raise exc

    return transport


_CROSSREF_PAYLOAD = {
    "message": {
        "items": [
            {
                "title": ["Training volume and powerlifting performance"],
                "URL": "https://doi.org/10.1234/abc",
                "DOI": "10.1234/abc",
                "author": [{"family": "Nikolaidis"}],
                "issued": {"date-parts": [[2019]]},
                "score": 38.0,
            }
        ]
    }
}

_OPENALEX_PAYLOAD = {
    "results": [
        {
            "title": "Training volume and powerlifting performance",
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1234/abc",
            "publication_year": 2019,
            "authorships": [{"author": {"display_name": "P. Nikolaidis"}}],
            "relevance_score": 39.0,
        }
    ]
}


class SourceLookupClientTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.cache_dir = Path(self._tmpdir.name) / "cache"

    def test_merges_crossref_and_openalex_candidates(self):
        client = SourceLookupClient(
            cache_dir=self.cache_dir,
            crossref_transport=_ok_transport(_CROSSREF_PAYLOAD),
            openalex_transport=_ok_transport(_OPENALEX_PAYLOAD),
        )
        candidates = client.candidates(surnames=["nikolaidis"], year=2019, topic_words=["training", "volume"])
        self.assertEqual(len(candidates), 2)
        providers = {c["provider"] for c in candidates}
        self.assertEqual(providers, {"crossref", "openalex"})

    def test_identical_query_is_served_from_cache(self):
        client = SourceLookupClient(
            cache_dir=self.cache_dir,
            crossref_transport=_ok_transport(_CROSSREF_PAYLOAD),
            openalex_transport=_ok_transport(_OPENALEX_PAYLOAD),
        )
        client.candidates(surnames=["nikolaidis"], year=2019, topic_words=["training"])
        client.candidates(surnames=["nikolaidis"], year=2019, topic_words=["training"])
        self.assertEqual(client.fetch_count, 1)

    def test_a_different_query_is_a_cache_miss(self):
        client = SourceLookupClient(
            cache_dir=self.cache_dir,
            crossref_transport=_ok_transport(_CROSSREF_PAYLOAD),
            openalex_transport=_ok_transport(_OPENALEX_PAYLOAD),
        )
        client.candidates(surnames=["nikolaidis"], year=2019, topic_words=["training"])
        client.candidates(surnames=["someone-else"], year=2019, topic_words=["training"])
        self.assertEqual(client.fetch_count, 2)

    def test_cache_survives_across_client_instances(self):
        client_a = SourceLookupClient(
            cache_dir=self.cache_dir,
            crossref_transport=_ok_transport(_CROSSREF_PAYLOAD),
            openalex_transport=_ok_transport(_OPENALEX_PAYLOAD),
        )
        client_a.candidates(surnames=["nikolaidis"], year=2019, topic_words=["training"])

        client_b = SourceLookupClient(
            cache_dir=self.cache_dir,
            crossref_transport=_ok_transport(_CROSSREF_PAYLOAD),
            openalex_transport=_ok_transport(_OPENALEX_PAYLOAD),
        )
        candidates = client_b.candidates(surnames=["nikolaidis"], year=2019, topic_words=["training"])
        self.assertEqual(client_b.fetch_count, 0)
        self.assertEqual(len(candidates), 2)

    def test_crossref_failure_degrades_to_openalex_only(self):
        client = SourceLookupClient(
            cache_dir=self.cache_dir,
            crossref_transport=_raising_transport(requests.exceptions.Timeout("simulated")),
            openalex_transport=_ok_transport(_OPENALEX_PAYLOAD),
        )
        candidates = client.candidates(surnames=["nikolaidis"], year=2019, topic_words=["training"])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["provider"], "openalex")

    def test_both_providers_failing_returns_empty_list_without_raising(self):
        client = SourceLookupClient(
            cache_dir=self.cache_dir,
            crossref_transport=_raising_transport(requests.exceptions.ConnectionError("simulated")),
            openalex_transport=_raising_transport(RuntimeError("simulated")),
        )
        candidates = client.candidates(surnames=["nikolaidis"], year=2019, topic_words=["training"])
        self.assertEqual(candidates, [])


class LoadOverridesTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_missing_file_returns_empty_dict(self):
        path = Path(self._tmpdir.name) / "does-not-exist.json"
        self.assertEqual(load_overrides(path), {})

    def test_a_valid_file_is_loaded(self):
        path = Path(self._tmpdir.name) / "overrides.json"
        payload = {"ep-1": {"abc123": {"title": "Corrected", "url": "https://x", "doi": "10.1/x"}}}
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(load_overrides(path), payload)

    def test_malformed_file_degrades_to_empty_dict(self):
        path = Path(self._tmpdir.name) / "broken.json"
        path.write_text("not valid json {", encoding="utf-8")
        self.assertEqual(load_overrides(path), {})


if __name__ == "__main__":
    unittest.main()
