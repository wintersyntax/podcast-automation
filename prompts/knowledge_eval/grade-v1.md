# Note grading — prompt v1

TASK-106 Phase A, `note-grade-v1` schema version. Design authority:
`docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md`
§11.3 (grading).

You are grading how well a piece of content ("the subject") answers a set
of practical questions, each with a reference answer. The subject is
either a finished podcast knowledge note (`subject_kind: "note"`) or a
list of verified, quote-anchored knowledge items (`subject_kind:
"items"`). You do **not** see the transcript and you do not see anything
beyond the given `subject` — grade only from what it actually contains.

## What to do for each question

For every entry in `questions` (each has `question_id`, `question`, and
`reference_answer`):

1. Find what the subject actually says that answers the question, if
   anything. Write this as `model_answer` — a faithful, concise summary of
   what the subject states, not your own outside knowledge. If the
   subject says nothing relevant, `model_answer` should say so plainly
   (e.g. "not addressed in the subject").
2. Compare `model_answer` against `reference_answer` and assign exactly
   one `grade`:
   - `correct` — the subject's content matches the reference answer's
     substance (numbers, conditions, and recommendation direction all
     agree; wording may differ).
   - `partial` — the subject addresses the question but is missing a
     material part of the reference answer (a qualifying condition, a
     number, a caveat) or is vaguer than the reference.
   - `missing` — the subject does not address the question at all.
   - `wrong` — the subject states something that contradicts the
     reference answer (a different number, a dropped or reversed
     condition, a recommendation the reference answer does not support).

A grade must be about content the subject actually contains — never grade
`correct` based on outside knowledge the subject itself does not state.

## Output

Return **only** JSON matching this shape — no prose before or after:

```json
{
  "episode_key": "<the episode_key from episode_context>",
  "answers": [
    {"question_id": "q-01", "model_answer": "Recommends 3-5 sets of 8-12 reps for hypertrophy.", "grade": "correct"},
    {"question_id": "q-02", "model_answer": "not addressed in the subject", "grade": "missing"}
  ]
}
```

Return exactly one answer per question you were given — never zero, never
two for the same `question_id`.
