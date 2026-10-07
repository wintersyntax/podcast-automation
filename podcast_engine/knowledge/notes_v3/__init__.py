"""Single-pass grounded knowledge notes (TASK-118).

One strong model call reads the whole canonical compiled transcript and
returns a structured note (``knowledge-note-v3``). ``writer`` makes that
single call; everything else in this package is deterministic Python with no
model calls, no network and no GCS access:

* ``schema``  -- the structured-output contract and its structural validation;
* ``units``   -- Python-owned SI conversion of ``{{5 lb}}`` markup;
* ``checks``  -- anchor verification, ad/sponsor and stub-bullet guards;
* ``render``  -- the Markdown body with the evidence-strength scale.

Design authority:
``docs/superpowers/specs/2026-09-29-single-pass-knowledge-note-design.md``.
"""
