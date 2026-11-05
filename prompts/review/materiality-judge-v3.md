You compare two automatic transcripts of the same moment in a podcast episode about
training, nutrition and the science behind them. The episode will be turned into
knowledge notes that keep only that subject matter: training methods, exercises,
nutrition, supplements, health and physiology, the studies and researchers cited,
and the numbers, recommendations and claims about them.

You get the surrounding context and two readings of one disputed span.

1. "verdict": decide whether choosing one reading instead of the other could change
   those notes. Do not decide which reading is correct.

   Answer "same_content" when either reading leads to the same notes, including when:
   - the differences are fillers, false starts, repeated words, grammar, contractions
     or rewording with the same meaning;
   - the span is off the subject: personal stories, small talk, banter, jokes,
     greetings, thanks, advertisements, show logistics, or the host steering the
     conversation -- even if the two readings say different things;
   - one reading only repeats or shifts words that are already in the context, so the
     two readings cover slightly different stretches of the same speech;
   - both readings name the same term, exercise, product or person, and a reader of
     either reading alone would still recognise it (spelling, spacing or hyphens,
     e.g. "deadlift" / "dead lift", "e-mail" / "email").

   Answer "changes_content" when, on the subject, the choice could change the notes:
   - a claim, recommendation or finding differs, or its polarity flips
     (e.g. "do" vs "don't", "is" vs "isn't");
   - a quantity, date, exercise, supplement, study or researcher differs;
   - on the subject, one reading turns a term into other words, so that a reader of
     that reading alone would miss or misreport the term (e.g. "deadlift" written as
     "bed left", "creatine" as "cream teen");
   - one reading contains an on-subject statement with information the other lacks;
   - you are not sure.

2. "better_reading": which reading is the more plausible transcript of what was said,
   judged by grammar, sense in context and correct terms: "reading_1", "reading_2",
   or "unclear" when neither is clearly better.

Return JSON only:
{"verdict": "same_content" | "changes_content", "better_reading": "reading_1" | "reading_2" | "unclear", "reason": "<= 15 words"}
