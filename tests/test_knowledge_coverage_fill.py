"""TASK-119: deterministic coverage detectors and the forced-decision fill."""

import json
import tempfile
import unittest
from pathlib import Path

from podcast_engine.knowledge.notes_v3 import coverage, fill

SHORT = (
    "Okay, welcome back. Today we talk about creatine and sleep. "
    "The 2024 study used 0.35 grams per kilogram and cognition was rescued by about ten to fifteen percent. "
    "For heavier guys Nordic curls are brutal, so I put a bench in front of me to catch my chest and shorten the range. "
    + "We chatted about the weather and our favourite shows for a long while. " * 7
    + "On the leg press I keep my feet high on the platform and use a full range of motion with a slow eccentric. "
    "I finished the diet at 89 kilos and I'm up to like 92 now. "
)
CHATTER = "We chatted about the weather and our favourite shows for a long while. " * 40
LONG = SHORT + CHATTER + "On the hack squat I keep a narrow stance and pause at the bottom for two seconds. "
TRANSCRIPT = LONG

def bullet(text, anchor, basis="none", hedged=False):
    return {"text": text, "basis": basis, "hedged": hedged, "anchor": anchor}


def note():
    return {
        "tldr": ["Creatine can offset some effects of sleep loss."],
        "sections": [
            {"title": "Creatine and sleep", "bottom_line": "", "protocols": [], "bullets": [
                bullet("The 2024 study used 0.35 g/kg and rescued cognition by about 10-15%.",
                       "the 2024 study used 0.35 grams per kilogram and cognition was rescued", "research"),
            ]},
            {"title": "Nordic curls", "bottom_line": "", "protocols": [], "bullets": [
                bullet("Casey puts a bench in front to catch his chest and shorten the range.",
                       "so I put a bench in front of me to catch my chest", "personal_experience"),
            ]},
        ],
        "research_discussed": [], "numbers": [], "takeaways": [], "sources_mentioned": [],
        "also_discussed": [], "follow_up": [], "topics": [], "people": [], "existing_tags": [],
        "new_tag_candidates": [],
    }


class LexiconTests(unittest.TestCase):
    def test_repository_lexicon_is_versioned_and_complete(self):
        lexicon = coverage.load_lexicon()
        self.assertEqual((len(lexicon.exercises), len(lexicon.muscles), len(lexicon.cues)), (329, 44, 98))
        self.assertIn("nordic", lexicon.exercises)
        self.assertTrue(lexicon.sha256.startswith("sha256:"))
        data = json.loads(coverage.LEXICON_PATH.read_text())
        self.assertEqual(data["schema_version"], coverage.LEXICON_SCHEMA_VERSION)
        self.assertTrue(any("CC BY-SA" in source for source in data["sources"]))

    def test_operator_terms_extend_the_lexicon_and_change_its_digest(self):
        base = coverage.load_lexicon()
        extended = coverage.load_lexicon(extra_terms={"exercises": ["Zottman Curl"]})
        self.assertIn("zottman curl", extended.exercises)
        self.assertNotEqual(base.sha256, extended.sha256)

    def test_wrong_schema_version_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lexicon.json"
            path.write_text(json.dumps({"schema_version": "other", "exercises": [], "muscles": [], "cues": [], "units": []}))
            with self.assertRaises(coverage.LexiconError):
                coverage.load_lexicon(path)


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = coverage.load_lexicon()

    def report(self, value=None, transcript=SHORT):
        return coverage.find_candidates(value or note(), transcript, self.lexicon, max_candidates=99)

    def test_uncovered_exercise_spot_is_found_and_covered_spot_is_not(self):
        spots = [c for c in self.report().candidates if c.type == "exercise_advice"]
        self.assertTrue(any("leg press" in c.terms for c in spots))
        self.assertFalse(any(c.terms == ["nordic"] for c in spots))
        self.assertTrue(all(c.excerpt for c in spots))

    def test_muscle_needs_a_cue_in_the_same_chunk(self):
        text = coverage.normalize("My glutes were sore yesterday after the trip. " * 3)
        self.assertEqual(coverage.exercise_spots(text, self.lexicon), [])
        text = coverage.normalize("Train glutes with a full range of motion and a slow eccentric every time.")
        self.assertEqual(len(coverage.exercise_spots(text, self.lexicon)), 1)

    def test_quantity_missing_from_the_note_is_a_candidate_and_decimals_stay_whole(self):
        details = " ".join(c.detail for c in self.report().candidates)
        self.assertIn("number 89", details)
        self.assertNotIn("number 0 ", details)
        self.assertNotIn("number 35", details)
        self.assertNotIn("number 2024", details)
        covered = note()
        covered["numbers"] = [{"topic": "Weight", "value": "89 kg and 92 kg", "basis": "personal_experience"}]
        self.assertNotIn("number 89", " ".join(c.detail for c in self.report(covered).candidates))

    def test_thousands_separator_is_read_back(self):
        from podcast_engine.knowledge.notes_v3.checks import normalize_with_index
        raw = "He walks 1,500 steps after dinner and takes 0.5 grams."
        text, index = normalize_with_index(raw)
        mentions = coverage.quantity_mentions(text, self.lexicon, raw, index)
        self.assertEqual([m["value"] for m in mentions], [1500.0, 0.5])

    def test_trigger_counts_spots_inside_an_uncovered_window(self):
        report = self.report(transcript=LONG)
        self.assertTrue(report.triggered)
        windows = [c for c in report.candidates if c.type == "uncovered_window"]
        self.assertTrue(windows)
        self.assertFalse(any(c.type == "exercise_advice" and "hack squat" in c.terms for c in report.candidates))
        self.assertGreaterEqual(report.raw_counts["exercise_advice"], 2)

    def test_uncovered_windows_alone_never_trigger(self):
        lonely = note()
        lonely["sections"] = [{"title": "Chat", "bottom_line": "", "protocols": [], "bullets": [
            bullet("They chatted about the weather for a while.", "we chatted about the weather and our favourite shows")]}]
        report = coverage.find_candidates(lonely, CHATTER * 2, self.lexicon)
        self.assertFalse(report.triggered)
        self.assertTrue(any(c.type == "uncovered_window" for c in report.candidates))

    def test_records_are_serialisable_with_excerpt_hashes(self):
        record = self.report().as_record()
        json.dumps(record)
        self.assertEqual(record["policy_version"], coverage.COVERAGE_POLICY_VERSION)
        self.assertTrue(all(c["excerpt_sha256"].startswith("sha256:") for c in record["candidates"]))


class FillTests(unittest.TestCase):
    def setUp(self):
        self.report = coverage.find_candidates(note(), TRANSCRIPT, coverage.load_lexicon(), max_candidates=99)
        self.ids = {c.type: c.id for c in self.report.candidates}

    def decision(self, cid, action, **kwargs):
        empty = bullet("", "")
        value = {"candidate_id": cid, "action": action, "section_title": "", "old_bullet_id": "",
                 "bullet": empty, "skip_reason": "not_applicable" if action != "skip" else "not_practical"}
        value.update(kwargs)
        return value

    def test_fill_input_carries_the_transcript_every_bullet_and_the_candidates(self):
        payload = fill.fill_input(note(), TRANSCRIPT, self.report, title="Ep 1")
        self.assertEqual(payload["transcript"], TRANSCRIPT)
        self.assertEqual(payload["episode_title"], "Ep 1")
        self.assertEqual(payload["section_titles"], ["Creatine and sleep", "Nordic curls"])
        self.assertEqual([b["bullet_id"] for b in payload["all_bullets"]], ["b001", "b002"])
        self.assertEqual(payload["all_bullets"][1]["section_title"], "Nordic curls")
        self.assertEqual({c["candidate_id"] for c in payload["candidates"]}, {c.id for c in self.report.candidates})
        self.assertTrue(all(len(c["transcript_excerpt"]) <= fill.MAX_EXCERPT_CHARS for c in payload["candidates"]))

    def test_parse_response_returns_decisions_and_refuses_truncation(self):
        content = json.dumps({"decisions": []})
        decisions, served = fill.parse_response({
            "model": "m", "provider": "p", "usage": {"cost": 0.05, "prompt_tokens": 9},
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        })
        self.assertEqual(decisions, [])
        self.assertEqual(served["usage"], {"cost": 0.05, "prompt_tokens": 9})
        with self.assertRaises(fill.FillTruncatedError):
            fill.parse_response({"choices": [{"finish_reason": "length", "message": {"content": content}}]})
        with self.assertRaises(fill.FillResponseError):
            fill.parse_response({"choices": []})

    def test_response_schema_is_strict_and_parse_rejects_malformed_content(self):
        self.assertTrue(fill.RESPONSE_FORMAT["json_schema"]["strict"])
        with self.assertRaises(fill.FillResponseError):
            fill.parse_decisions("not json")
        with self.assertRaises(fill.FillResponseError):
            fill.parse_decisions({"decisions": [{"candidate_id": "c01", "action": "maybe"}]})

    def test_add_extend_and_skip_are_applied_and_recorded(self):
        spot = self.ids["exercise_advice"]
        quantity = self.ids["quantity"]
        window = self.ids["uncovered_window"]
        decisions = [
            self.decision(spot, "add", section_title="Leg press", bullet=bullet(
                "On the leg press he keeps his feet high and uses a full range of motion with a slow eccentric.",
                "I keep my feet high on the platform and use a full range of motion", "personal_experience")),
            self.decision(quantity, "extend", old_bullet_id="b002", bullet=bullet(
                "Casey puts a bench in front to catch his chest and shorten the range; he finished the diet at 89 kg.",
                "I finished the diet at 89 kilos and I'm up to like 92 now", "personal_experience")),
            self.decision(window, "skip", skip_reason="off_scope_or_small_talk"),
        ]
        parsed = fill.parse_decisions({"decisions": decisions})
        updated, report = fill.apply_decisions(note(), TRANSCRIPT, self.report.candidates, parsed)
        self.assertEqual(report.applied, {"add": 1, "extend": 1, "skip": 1})
        self.assertEqual(updated["sections"][-1]["title"], "Leg press")
        self.assertIn("89 kg", updated["sections"][1]["bullets"][0]["text"])
        extend = next(d for d in report.decisions if d["action"] == "extend")
        self.assertEqual(extend["old_text"], note()["sections"][1]["bullets"][0]["text"])
        self.assertEqual(note()["sections"][1]["bullets"][0]["text"][:13], "Casey puts a ")
        json.dumps(report.as_record())

    def test_invalid_decisions_are_recorded_but_never_applied(self):
        spot = self.ids["exercise_advice"]
        quantity = self.ids["quantity"]
        window = self.ids["uncovered_window"]
        decisions = [
            self.decision(spot, "add", section_title="Leg press", bullet=bullet(
                "Feet high on the leg press platform with a full range and a slow eccentric.",
                "words that were never said anywhere in this episode at all")),
            self.decision(quantity, "extend", old_bullet_id="b999", bullet=bullet(
                "He finished his diet at 89 kg and is now about 92 kg.",
                "I finished the diet at 89 kilos and I'm up to like 92 now")),
            self.decision(window, "add", section_title="Sponsor", bullet=bullet(
                "Use code DEMO10 at the store for a discount on gear and equipment.",
                "we chatted about the weather and our favourite shows")),
            self.decision("c99", "skip"),
        ]
        updated, report = fill.apply_decisions(note(), TRANSCRIPT, self.report.candidates, decisions)
        self.assertEqual(updated, note())
        self.assertEqual(report.rejected, {"anchor_not_in_transcript": 1, "unknown_bullet_id": 1, "ad_or_sponsor": 1})
        self.assertEqual(report.unknown, ["c99"])

    def test_missing_and_duplicate_decisions_are_reported(self):
        first = self.report.candidates[0].id
        decisions = [self.decision(first, "skip"), self.decision(first, "skip")]
        _, report = fill.apply_decisions(note(), TRANSCRIPT, self.report.candidates, decisions)
        self.assertEqual(report.applied["skip"], 1)
        self.assertEqual(sorted(report.missing), sorted(c.id for c in self.report.candidates[1:]))

    def test_prompt_is_versioned_and_hashed(self):
        self.assertIn("note-fill-v1", fill.system_prompt())
        self.assertTrue(fill.prompt_sha256().startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
