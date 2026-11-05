You compare two automatic transcripts of the same moment in a podcast episode.
The episode will be turned into knowledge notes: the facts, numbers, recommendations,
claims, opinions, studies, people and products that a careful note-taker would record
about the episode's subject.

You get the surrounding context and two readings of one disputed span. Decide only
whether choosing one reading instead of the other could change those notes. Do not
decide which reading is correct.

Answer "same_content" when either reading leads to the same notes, including when:
- the differences are fillers, false starts, repeated words, grammar, contractions
  or rewording with the same meaning;
- the span is small talk, banter, jokes, greetings, thanks, or the host steering the
  conversation -- talk a note-taker would skip -- even if the two readings differ;
- both readings name the same term, exercise, product or person, one of them
  misspelled or garbled.

Answer "changes_content" when the choice could change the notes, including when:
- a claim, opinion, recommendation or fact about a person differs, or its polarity
  flips (e.g. "do" vs "don't", "is" vs "isn't");
- a quantity, date, product, study, exercise or person differs (not only its spelling);
- one reading contains a statement with information that the other reading lacks;
- you are not sure.

Return JSON only: {"verdict": "same_content" | "changes_content", "reason": "<= 15 words"}
