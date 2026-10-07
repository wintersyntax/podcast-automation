"""TASK-133: deterministic materiality rule for Human Review cards."""

from __future__ import annotations

import unittest

from compiler.materiality import materiality_inputs, rule_verdict
from compiler.materiality_choice import choose_reading, proposal


def card(left: str, apple: str, whisper: str, right: str, **extra) -> dict:
    return {
        "id": 1,
        "apple_text": apple,
        "whisper_text": whisper,
        "apple_context": f"{left} {apple} {right}".strip(),
        "whisper_context": f"{left} {whisper} {right}".strip(),
        **extra,
    }


def verdict(left: str, apple: str, whisper: str, right: str) -> tuple[str, str]:
    inputs = materiality_inputs(card(left, apple, whisper, right))
    assert inputs is not None
    return rule_verdict(inputs)


class MaterialityRuleTests(unittest.TestCase):
    def test_abandoned_negated_start_restarted_with_a_negation_is_immaterial(self):
        self.assertEqual(
            verdict("the earlier draft was rejected. So", "", "it's not, you know,", "it isn't finished yet. And there is"),
            ("immaterial", "fillers_restarts"),
        )

    def test_lone_no_interjection_is_immaterial(self):
        self.assertEqual(
            verdict("the idea was still unclear, but like", "", "No", ", I don't agree with that. And so"),
            ("immaterial", "fillers_restarts"),
        )

    def test_dropped_negation_that_flips_the_claim_goes_to_the_judge(self):
        self.assertEqual(
            verdict("the proposal is straightforward. I", "", "don't", "think there is anything unusual about doing"),
            ("judge", "negation"),
        )

    def test_numbers_always_stay_with_the_reviewer(self):
        self.assertEqual(
            verdict("for most people, you're in that probably", "200", "2", "to 400 units in the example"),
            ("reviewer", "number"),
        )
        self.assertEqual(
            verdict("the schedule might move around", "twenty twenty", "twenty twenty eight", "before launch"),
            ("reviewer", "number"),
        )

    def test_same_numbers_written_differently_are_immaterial(self):
        for apple, whisper in (
            ("am or 2am", "a.m. or 2 a.m"),
            ("PM and 10 PM", "p.m. and 10 p.m"),
            ("40s and 50s to the 60s and 70s", "forties and fifties to the sixties and seventies"),
            ("1,500", "a thousand five hundred"),
            ("twenty five percent", "25%"),
            ("a hundred and ten kilos", "110 kg"),
            ("7:30", "7.30"),
        ):
            with self.subTest(apple=apple):
                self.assertEqual(verdict("so", apple, whisper, "and then"), ("immaterial", "number_format"))

    def test_different_numbers_stay_with_the_reviewer(self):
        for apple, whisper in (
            ("2017", "17"),
            ("five sets", "two three sets"),
            ("7.30", "7, 30"),
            ("twenty seventeen", "2017"),
            ("110 kilo deadlift", "a hundred, 110 kilo deadlift"),
            ("we don't do 3 sets", "we do 3 sets"),
        ):
            with self.subTest(apple=apple):
                self.assertEqual(verdict("so", apple, whisper, "and then"), ("reviewer", "number"))

    def test_filler_and_function_word_differences_are_immaterial(self):
        self.assertEqual(
            verdict("you do not experience", "just a", "like, you know, just the", "a meaningful change"),
            ("immaterial", "function_words"),
        )

    def test_spacing_and_hyphenation_only_is_immaterial(self):
        self.assertEqual(
            verdict("we received an", "e-mail from the sender", "email from the sender", "that we had discussed"),
            ("immaterial", "same_letters"),
        )

    def test_repetition_of_the_next_words_is_immaterial(self):
        self.assertEqual(
            verdict("but it is not,", "", "it is not", "it is not part of the final plan"),
            ("immaterial", "fillers_restarts"),
        )

    def test_content_word_and_modal_differences_go_to_the_judge(self):
        self.assertEqual(
            verdict("the example mentions", "creatine", "creating", "in the sentence"),
            ("judge", "content"),
        )
        self.assertEqual(
            verdict("the bracket sits beside", "the frame. You can", "the frame and you have to", "attach the support"),
            ("judge", "content"),
        )

    def test_sponsor_read_in_one_source_keeps_the_other_reading(self):
        ad = ("This message is brought to you by ExampleCard. With ExampleCard, you earn unlimited example reward points "
              "on everyday purchases, like ordinary example purchases, anywhere it is accepted.")
        self.assertEqual(verdict("", ad, "", ". After the break, we return to the discussion"), ("advertisement", "use_whisper"))
        # A sponsor read with numbers is still recognised before the number rule.
        self.assertEqual(verdict("", "", ad + " Call 555 today.", "and welcome"), ("advertisement", "use_apple"))

    def test_long_stretch_only_one_source_has_stays_with_the_reviewer(self):
        apple = ("So sometime between seven and nine, depending upon the day and whether it's the weekend, "
                 "I'm having breakfast, which typically consists of some type of mix of vegetables, and or toast")
        self.assertEqual(verdict("The example schedule begins here.", apple, "", "and then it continues"), ("reviewer", "number"))
        apple = ("So we don't see really any net negative impact of that muscle losing some volume as it improves "
                 "its characteristics in how it handles glucose and insulin over the following months")
        self.assertEqual(verdict("The synthetic context starts here.", apple, "", "and ends here"), ("reviewer", "one_sided_long"))

    def test_partial_focus_scopes_the_readings(self):
        item = card("so", "we did approve the change", "we didn't approve the change", "before the next step")
        item["focus"] = {"scope": "partial", "apple_text": "did", "whisper_text": "didn't"}
        inputs = materiality_inputs(item)
        self.assertEqual((inputs["apple"], inputs["whisper"]), ("did", "didn't"))
        self.assertEqual(rule_verdict(inputs), ("judge", "negation"))

    def test_missing_reading_has_no_inputs(self):
        self.assertIsNone(materiality_inputs({"id": 1, "apple_text": "x"}))

    def test_inputs_without_third_asr_evidence_have_no_third_reading(self):
        self.assertIsNone(materiality_inputs(card("so", "a", "b", "c"))["third"])


def choice_inputs(apple: str, whisper: str, third: str | None = None) -> dict:
    return {"apple": apple, "whisper": whisper, "left": "", "right": "", "third": third}


class ReadingChoiceTests(unittest.TestCase):
    def test_no_source_is_preferred_without_evidence(self):
        self.assertEqual(choose_reading(choice_inputs("term alpha", "term beta"), reason="judge"), (None, "click_one"))

    def test_advertisement_keeps_the_reading_without_the_sponsor_read(self):
        self.assertEqual(choose_reading(choice_inputs("ad", ""), reason="use_whisper"), ("whisper", "advertisement"))

    def test_third_asr_must_equal_exactly_one_reading(self):
        inputs = choice_inputs("restore the target value", "Thank you", third="you know, restore the target value")
        self.assertEqual(choose_reading(inputs, reason="judge"), ("apple", "third_asr"))
        self.assertEqual(choose_reading(choice_inputs("a", "b", third="c"), reason="judge"), (None, "click_one"))

    def test_registry_term_matches_whole_words_only(self):
        self.assertEqual(choose_reading(choice_inputs("creating", "creatine"), reason="judge"),
                         ("whisper", "registry_term"))
        self.assertEqual(choose_reading(choice_inputs("a hurdle", "a hurl"), reason="judge"), (None, "click_one"))

    def test_unanimous_judge_vote_decides(self):
        inputs = choice_inputs("I play", "I played")
        self.assertEqual(choose_reading(inputs, reason="judge", votes=["whisper"] * 3), ("whisper", "judge"))
        self.assertEqual(choose_reading(inputs, reason="judge", votes=["whisper", "unclear", "whisper"]),
                         (None, "click_one"))

    def test_rule_settled_card_keeps_the_fuller_reading(self):
        inputs = choice_inputs("but it's not there", "but it's not, you know, it's not there")
        self.assertEqual(choose_reading(inputs, reason="fillers_restarts"), ("whisper", "fuller"))
        self.assertEqual(choose_reading(choice_inputs("e-mail", "email"), reason="same_letters"), (None, "either"))

    def test_proposal_needs_unanimous_votes_the_third_asr_does_not_contradict(self):
        inputs = choice_inputs("option alpha or option beta", "garbled option beta")
        self.assertEqual(proposal(inputs, reason="judge", votes=["apple"] * 3), ("apple", "judge"))
        contradicted = choice_inputs("option alpha or option beta", "garbled option beta", third="garbled option beta")
        self.assertEqual(proposal(contradicted, reason="judge", votes=["apple"] * 3), (None, None))
        self.assertEqual(proposal(inputs, reason="judge", votes=["apple", "unclear", "apple"]), (None, None))

    def test_number_card_proposes_the_only_reading_with_words(self):
        self.assertEqual(proposal(choice_inputs("50 units", "um"), reason="number"), ("apple", "number_gap"))
        self.assertEqual(proposal(choice_inputs("2017", "17"), reason="number"), (None, None))


if __name__ == "__main__":
    unittest.main()
