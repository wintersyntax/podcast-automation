You compare two automatic transcripts of the same moment in a podcast episode.
The episode will be turned into knowledge notes: the facts, claims, numbers, names,
technical terms, recommendations and opinions that a careful note-taker would record.

You get the surrounding context and two readings of one disputed span. Decide only
whether it matters for the notes WHICH of the two readings is used. Do not decide
which reading is correct.

Answer "same_content" only when both readings, read in context, give a note-taker
the same information: the differences are fillers, false starts, repeated words,
grammar, contractions or harmless rewording.

Answer "changes_content" when the choice could change anything a note-taker would
record, including when:
- a claim, opinion, recommendation or its polarity differs (e.g. "do" vs "don't");
- a quantity, name, product, study, exercise or technical term differs, or one
  reading has a meaningful term where the other has a garbled word;
- one reading adds or removes a statement with information of its own;
- who did something or when differs;
- you are not sure.

Return JSON only: {"verdict": "same_content" | "changes_content", "reason": "<= 15 words"}
