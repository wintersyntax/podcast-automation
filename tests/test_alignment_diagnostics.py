import unittest

from compiler.alignment_diagnostics import (
    ReviewSpan,
    build_alignment_diagnostics_shadow,
    disabled_alignment_diagnostics,
    needleman_wunsch_opcodes,
)
from compiler.transcript import compile_transcripts


class AlignmentDiagnosticsTests(unittest.TestCase):
    def test_needleman_wunsch_keeps_simple_insertion_local(self):
        opcodes = needleman_wunsch_opcodes(
            ["creatine", "is", "very", "useful"],
            ["creatine", "is", "useful"],
        )
        self.assertEqual(opcodes[0], ("equal", 0, 2, 0, 2))
        self.assertIn(("delete", 2, 3, 2, 2), opcodes)
        self.assertEqual(opcodes[-1], ("equal", 3, 4, 2, 3))

    def test_large_insert_region_gets_shadow_comparison(self):
        apple = ["start", "finish"]
        inserted = [f"extra{i}" for i in range(12)]
        whisper = ["start", *inserted, "finish"]
        production = [
            ("equal", 0, 1, 0, 1),
            ("insert", 1, 1, 1, 13),
            ("equal", 1, 2, 13, 14),
        ]
        payload = build_alignment_diagnostics_shadow(
            apple,
            whisper,
            production,
            review_spans=[ReviewSpan(7, 1, 1, 1, 13, True)],
        )
        self.assertEqual(payload["mode"], "shadow")
        self.assertFalse(payload["authoritative"])
        self.assertEqual(payload["decision_effect"], "none")
        self.assertEqual(payload["suspicious_region_count"], 1)
        region = payload["regions"][0]
        self.assertIn("large_insert_span", region["reasons"])
        self.assertEqual(region["review_overlap_count"], 1)
        self.assertEqual(region["comparison"]["status"], "compared")

    def test_small_difference_is_not_flagged(self):
        payload = build_alignment_diagnostics_shadow(
            ["a", "b", "c"],
            ["a", "x", "c"],
            [
                ("equal", 0, 1, 0, 1),
                ("replace", 1, 2, 1, 2),
                ("equal", 2, 3, 2, 3),
            ],
        )
        self.assertEqual(payload["suspicious_region_count"], 0)
        self.assertEqual(payload["comparison_count"], 0)

    def test_disabled_contract_is_explicitly_non_authoritative(self):
        payload = disabled_alignment_diagnostics()
        self.assertEqual(payload["mode"], "disabled")
        self.assertFalse(payload["authoritative"])
        self.assertEqual(payload["decision_effect"], "none")

    def test_compile_shadow_toggle_does_not_change_canonical_decision_surface(self):
        inserted = " ".join(f"extra{i}" for i in range(12))
        apple = "Start finish."
        whisper = f"Start {inserted} finish."
        enabled = compile_transcripts(apple, whisper, alignment_diagnostics_shadow=True)
        disabled = compile_transcripts(apple, whisper, alignment_diagnostics_shadow=False)

        enabled_dict = enabled.to_dict()
        disabled_dict = disabled.to_dict()
        enabled_diag = enabled_dict.pop("alignment_diagnostics")
        disabled_diag = disabled_dict.pop("alignment_diagnostics")

        self.assertEqual(enabled.compiled_transcript, disabled.compiled_transcript)
        self.assertEqual(enabled_dict, disabled_dict)
        self.assertEqual(enabled_diag["mode"], "shadow")
        self.assertEqual(disabled_diag["mode"], "disabled")


if __name__ == "__main__":
    unittest.main()
