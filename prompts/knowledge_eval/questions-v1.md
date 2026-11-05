# Reference-question generation — prompt v1

TASK-106 Phase A, `questions-v1` schema version. Design authority:
`docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md`
§11.2 (reference questions).

You are writing an offline evaluation question set for one full episode of a
fitness/nutrition podcast. You see the **entire transcript** for this episode
plus its title and description for orientation. These questions will later be
asked of a knowledge-note pipeline you never see, and graded against your
reference answers — so every question must have one clear, checkable answer
that is actually stated in this transcript, and every answer must be
supported by an exact quote you can point to.

## What to produce

Write 10-15 practical questions a real listener might ask after hearing this
episode — the kind of thing someone would want to know and act on, not a
trivia question about incidental phrasing. For each question, give:

- `question` — a natural, self-contained question a listener could ask.
- `reference_answer` — the correct answer, as it would be stated in a
  knowledge note: complete, but no longer than necessary.
- `scope` — `core`, `life_support`, `sport_philosophy`, or `out_of_scope`
  (same categories as extraction; see below). Only write questions whose
  `scope` is **not** `out_of_scope` — do not ask questions about ads,
  sponsor reads, relationship/dating advice, or podcast-production talk.
- `quotes` — one or more supporting quotes, each an **exact, verbatim**
  substring of the transcript (do not paraphrase, fix disfluencies, or
  normalize punctuation) with its `quote_occurrence` (1 if it is the first
  place that exact text appears in the transcript, 2 for the second, and so
  on).

Spread your questions across the whole episode: some from early in the
conversation, some from the middle, some from near the end. Do not cluster
every question in one part of the transcript.

## Nuance-linking questions (at least 2 required)

At least **2** of your 10-15 questions must be "nuance-linking" questions: a
question whose correct answer depends on combining a qualifier or condition
stated in one part of the transcript with a claim or recommendation stated
in a **different, non-adjacent** part — the kind of question a system that
only reads one short chunk at a time could get wrong, because the caveat and
the claim are far apart.

For a nuance-linking question, give **two** supporting quotes in `quotes`:
one for the qualifying/conditioning statement, one for the claim or
recommendation it changes. Pick quotes that are genuinely far apart in the
transcript (not two sentences next to each other) — this is checked
mechanically against how far apart they actually are, so a pair that is too
close together will not count no matter how you phrase the question.

Example shape (content illustrative only): the speaker says early on "that
recommendation is really for people already at an advanced training age" and
much later gives a general dosing recommendation without repeating that
caveat — a good nuance-linking question asks whether that recommendation
applies to a beginner, with the reference answer reflecting the earlier
caveat.

## Scope categories (same as extraction)

- `core` — training, bodybuilding, hypertrophy, nutrition, supplements,
  recovery, injury, competition.
- `life_support` — sleep, work, stress, scheduling, travel, finances, but
  only as they concern living and training around the sport.
- `sport_philosophy` — identity, discipline, motivation, long-term
  relationship with lifting.
- `out_of_scope` — relationships/dating/family advice, content-creation or
  podcast-production talk, marketing, unrelated chat, ads, or sponsor reads.
  Do not write questions with this scope.

## Quote discipline

A quote must appear **character-for-character** in the transcript you were
given (aside from line-ending differences, which do not matter). If you
cannot find an exact substring that supports an answer, do not invent the
quote or approximate it — either find a different supporting passage that is
actually there, or drop that question. A question with a quote that fails to
locate exactly in the transcript will be rejected and does not count toward
the 10-15 total.

## If your previous attempt was insufficient

If you are told your previous response did not have enough valid questions,
enough nuance-linking questions, or was missing a position (beginning/
middle/end) of the episode, read the listed reasons and produce a corrected,
complete question set that addresses every reason given — do not just add a
few more questions on top of the previous set.

## Output

Return only the JSON object matching the `reference_questions_v1` schema:
`episode_key`, and `questions` (each with `question_id`, `question`,
`reference_answer`, `scope`, and `quotes`). No prose outside the JSON.
