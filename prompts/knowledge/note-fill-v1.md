# Podcast knowledge note — coverage fill prompt (note-fill-v1)

TASK-119. Design authority: `docs/KNOWLEDGE_GENERATION.md` (section 7).

You maintain a permanent knowledge note written from one podcast episode
transcript. The reader will never listen to the episode, so practical detail the
note misses is lost.

Software scanned the transcript deterministically and found **candidate
passages** the note may not cover:

- `exercise_advice`: exercise execution or technique (set-up, range of motion,
  depth, stance, grip, tempo, pauses, cues, equipment tricks, easier/harder
  variations, why a variation is chosen, what goes wrong and how it is fixed);
- `quantity`: a number with a unit (or a year) whose value appears nowhere in
  the note;
- `uncovered_window`: a stretch of transcript with no note point anchored in it.

You receive the episode title, every section title of the note, every bullet of
the note (`all_bullets`, each with its `bullet_id` and section), the whole
`transcript`, and for each candidate its `transcript_excerpt` and the detector's
hit. Use the transcript for context around an excerpt; decide only the listed
candidates.

## Decide every candidate

Return **exactly one decision per candidate**, in `decisions`:

- `extend`: a listed bullet covers the topic but misses this detail. Return the
  whole improved bullet in `bullet` and that bullet's `bullet_id` in
  `old_bullet_id`. Keep everything the old bullet said.
- `add`: the passage holds practical content the note lacks. Return a new
  `bullet` and the `section_title` it belongs to — an existing title exactly as
  listed, or a new short descriptive title when none fits.
- `skip`: nothing of practical value is missing. Set `skip_reason`:
  `ad_or_plug` (sponsor reads, discount codes, membership/merch/podcast plugs),
  `off_scope_or_small_talk` (relationships, content creation, chat, life story),
  `already_covered` (a bullet in `all_bullets` already states it),
  `not_practical` (no usable advice, number or finding).

For `add` and `extend` set `skip_reason` to `not_applicable`. For `skip` and
`add` set `old_bullet_id` to an empty string; for `skip` set `section_title` to
an empty string and fill `bullet` with empty strings, `basis` `none` and
`hedged` false.

## How to write a bullet

Follow the note's rules:

- Keep the claim with its conditions, reasons, alternatives, caveats and every
  number exactly as stated (keep both ends of ranges). Two or three sentences are
  fine when needed. Never write a label-only bullet.
- Never convert units. Write every non-SI quantity exactly as spoken inside
  double braces with digits: `{{5 lb}}`, `{{1 g/lb}}`, `{{6 feet}}`. Write kcal,
  g, kg, mg, ml, reps, sets, RIR, RPE, %, hours and minutes normally with digits.
- Never add facts, numbers or advice that are not in the excerpt; do not
  strengthen or weaken certainty; do not turn an anecdote into a general rule.
- `basis`: `research`, `coaching_experience`, `personal_experience`, `opinion`,
  or `none` for a purely descriptive bullet; `hedged: true` when the speaker was
  unsure.
- `anchor`: a **verbatim** span of 6–20 consecutive words copied exactly from the
  candidate's `transcript_excerpt` (ASR errors included) where the point is made.
  Software checks it; do not paraphrase.
- Write in English. Never mention ads, sponsors or plugs.

Return only the JSON object.
