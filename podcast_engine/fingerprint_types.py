"""Static-only labels for two persisted fingerprint kinds; both remain strings."""

from typing import NewType, TypedDict


SourceFingerprint = NewType("SourceFingerprint", str)
InputFingerprint = NewType("InputFingerprint", str)


class ReviewFingerprintFields(TypedDict):
    source_fingerprint: SourceFingerprint
    input_fingerprint: InputFingerprint
