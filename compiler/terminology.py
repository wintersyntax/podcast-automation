"""Structured terminology registry for transcript-compiler candidate evidence.

Terminology Registry V1 is intentionally a structural refactor of the historical
``DOMAIN_GLOSSARY``.  Only ``canonical`` and ``category`` participate in the
current compiler behavior.  Aliases, risk metadata, pronunciation hints, and
provenance are schema-ready metadata and must not gain decision authority without
an explicit, tested compiler-policy change.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


TERMINOLOGY_SCHEMA_VERSION = 1
TERMINOLOGY_POLICY_VERSION = "terminology-policy-v1"
TERMINOLOGY_PROVENANCE = "legacy-domain-glossary-v1"

ALLOWED_TERMINOLOGY_CATEGORIES = (
    "exercise_name",
    "supplement",
    "training_term",
    "scientific_medical_term",
    "citation",
)
ALLOWED_TERMINOLOGY_RISK_LEVELS = ("meaning_sensitive",)
ACTIVE_SEMANTIC_FIELDS = ("canonical", "category")

_SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class TerminologyEntry:
    """One validated terminology entry.

    V1 compiler semantics use only ``canonical`` and ``category``.  The other
    fields are deliberately non-authoritative metadata until a later policy
    explicitly activates them.
    """

    canonical: str
    category: str
    aliases: tuple[str, ...] = ()
    risk_level: str = "meaning_sensitive"
    pronunciation_hint: str | None = None
    provenance: str = TERMINOLOGY_PROVENANCE


def _normalized_key(value: str) -> str:
    return _SPACE_PATTERN.sub(" ", value.strip()).casefold()


def _require_clean_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be blank")
    normalized = _SPACE_PATTERN.sub(" ", value.strip())
    if value != normalized:
        raise ValueError(f"{field} must be trimmed and use single spacing: {value!r}")
    return value


def validate_terminology_registry(
    entries: Sequence[TerminologyEntry],
) -> tuple[TerminologyEntry, ...]:
    """Validate registry invariants and fail closed on ambiguous vocabulary.

    Validation is intentionally stricter than the current compiler requires so
    future growth does not create silent alias/canonical ambiguity.
    """

    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("terminology registry must be a sequence")
    if not entries:
        raise ValueError("terminology registry must not be empty")

    canonical_owners: dict[str, str] = {}
    alias_owners: dict[str, str] = {}
    validated: list[TerminologyEntry] = []

    for index, entry in enumerate(entries):
        if not isinstance(entry, TerminologyEntry):
            raise ValueError(f"entry {index} must be a TerminologyEntry")

        canonical = _require_clean_text(entry.canonical, field=f"entry {index} canonical")
        category = _require_clean_text(entry.category, field=f"entry {index} category")
        provenance = _require_clean_text(entry.provenance, field=f"entry {index} provenance")
        risk_level = _require_clean_text(entry.risk_level, field=f"entry {index} risk_level")

        if category not in ALLOWED_TERMINOLOGY_CATEGORIES:
            raise ValueError(f"entry {index} has unknown category: {category!r}")
        if risk_level not in ALLOWED_TERMINOLOGY_RISK_LEVELS:
            raise ValueError(f"entry {index} has unknown risk_level: {risk_level!r}")

        if entry.pronunciation_hint is not None:
            _require_clean_text(
                entry.pronunciation_hint,
                field=f"entry {index} pronunciation_hint",
            )

        if not isinstance(entry.aliases, tuple):
            raise ValueError(f"entry {index} aliases must be a tuple")

        canonical_key = _normalized_key(canonical)
        previous_canonical = canonical_owners.get(canonical_key)
        if previous_canonical is not None:
            raise ValueError(
                "duplicate canonical terminology form: "
                f"{canonical!r} conflicts with {previous_canonical!r}"
            )

        # A canonical form may not reuse an alias owned by an earlier entry.
        previous_alias_owner = alias_owners.get(canonical_key)
        if previous_alias_owner is not None:
            raise ValueError(
                "canonical terminology form collides with an alias: "
                f"{canonical!r} is already an alias of {previous_alias_owner!r}"
            )

        canonical_owners[canonical_key] = canonical

        local_aliases: set[str] = set()
        for alias_index, alias in enumerate(entry.aliases):
            alias = _require_clean_text(
                alias,
                field=f"entry {index} alias {alias_index}",
            )
            alias_key = _normalized_key(alias)
            if alias_key == canonical_key:
                raise ValueError(
                    f"alias {alias!r} duplicates its canonical form {canonical!r}"
                )
            if alias_key in local_aliases:
                raise ValueError(
                    f"duplicate alias {alias!r} within canonical form {canonical!r}"
                )
            if alias_key in canonical_owners:
                raise ValueError(
                    "alias collides with a canonical terminology form: "
                    f"{alias!r} conflicts with {canonical_owners[alias_key]!r}"
                )
            previous_owner = alias_owners.get(alias_key)
            if previous_owner is not None:
                raise ValueError(
                    "ambiguous alias shared by multiple terminology entries: "
                    f"{alias!r} belongs to both {previous_owner!r} and {canonical!r}"
                )
            local_aliases.add(alias_key)
            alias_owners[alias_key] = canonical

        validated.append(entry)

    return tuple(validated)


def _entry(category: str, canonical: str) -> TerminologyEntry:
    return TerminologyEntry(canonical=canonical, category=category)


# These 47 canonical forms exactly preserve the pre-V1 DOMAIN_GLOSSARY surface.
# Metadata-only fields intentionally remain at safe defaults in this refactor.
_RAW_TERMINOLOGY_REGISTRY = (
    _entry("exercise_name", "Bulgarian split squat"),
    _entry("exercise_name", "Romanian deadlift"),
    _entry("exercise_name", "RDL"),
    _entry("exercise_name", "hex bar deadlift"),
    _entry("exercise_name", "trap bar deadlift"),
    _entry("exercise_name", "leg press"),
    _entry("exercise_name", "hip thrust"),
    _entry("exercise_name", "lat pulldown"),
    _entry("exercise_name", "bench press"),
    _entry("exercise_name", "back squat"),
    _entry("exercise_name", "front squat"),
    _entry("exercise_name", "deadlift"),
    _entry("supplement", "creatine monohydrate"),
    _entry("supplement", "creatine"),
    _entry("supplement", "beta-alanine"),
    _entry("supplement", "caffeine"),
    _entry("supplement", "HMB"),
    _entry("supplement", "citrulline malate"),
    _entry("supplement", "whey protein"),
    _entry("supplement", "casein"),
    _entry("supplement", "electrolytes"),
    _entry("training_term", "RPE"),
    _entry("training_term", "RIR"),
    _entry("training_term", "1RM"),
    _entry("training_term", "progressive overload"),
    _entry("training_term", "volume landmarks"),
    _entry("training_term", "deload"),
    _entry("training_term", "hypertrophy"),
    _entry("training_term", "periodization"),
    _entry("scientific_medical_term", "cortisol"),
    _entry("scientific_medical_term", "diabetes"),
    _entry("scientific_medical_term", "estrogen"),
    _entry("scientific_medical_term", "glucose"),
    _entry("scientific_medical_term", "hypertension"),
    _entry("scientific_medical_term", "insulin"),
    _entry("scientific_medical_term", "metformin"),
    _entry("scientific_medical_term", "nocebo"),
    _entry("scientific_medical_term", "placebo"),
    _entry("scientific_medical_term", "semaglutide"),
    _entry("scientific_medical_term", "testosterone"),
    _entry("citation", "Brad Schoenfeld"),
    _entry("citation", "Stuart Phillips"),
    _entry("citation", "James Krieger"),
    _entry("citation", "Eric Helms"),
    _entry("citation", "Layne Norton"),
    _entry("citation", "Mike Israetel"),
    _entry("citation", "Alan Aragon"),
)

TERMINOLOGY_REGISTRY = validate_terminology_registry(_RAW_TERMINOLOGY_REGISTRY)


def terminology_by_category(
    entries: Iterable[TerminologyEntry] = TERMINOLOGY_REGISTRY,
) -> dict[str, tuple[str, ...]]:
    """Return the legacy category -> canonical tuple compatibility view.

    Aliases and metadata are deliberately excluded in V1 so compiler decisions
    remain byte-for-byte compatible with the historical DOMAIN_GLOSSARY.
    """

    grouped: dict[str, list[str]] = {
        category: [] for category in ALLOWED_TERMINOLOGY_CATEGORIES
    }
    for entry in entries:
        grouped[entry.category].append(entry.canonical)
    return {category: tuple(values) for category, values in grouped.items()}


DOMAIN_GLOSSARY = terminology_by_category()


def terminology_registry_payload(
    entries: Sequence[TerminologyEntry] = TERMINOLOGY_REGISTRY,
) -> dict:
    """Return a stable audit payload; it does not affect compiler decisions."""

    return {
        "schema_version": TERMINOLOGY_SCHEMA_VERSION,
        "policy_version": TERMINOLOGY_POLICY_VERSION,
        "active_semantic_fields": list(ACTIVE_SEMANTIC_FIELDS),
        "entries": [asdict(entry) for entry in entries],
    }


def terminology_registry_sha256(
    entries: Sequence[TerminologyEntry] = TERMINOLOGY_REGISTRY,
) -> str:
    """Fingerprint the complete validated registry for future provenance use."""

    payload = terminology_registry_payload(entries)
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


TERMINOLOGY_REGISTRY_SHA256 = terminology_registry_sha256()


def terminology_registry_manifest() -> dict[str, object]:
    """Return compact non-authoritative provenance for compiler reports."""

    return {
        "schema_version": TERMINOLOGY_SCHEMA_VERSION,
        "policy_version": TERMINOLOGY_POLICY_VERSION,
        "sha256": TERMINOLOGY_REGISTRY_SHA256,
        "entry_count": len(TERMINOLOGY_REGISTRY),
        "active_semantic_fields": list(ACTIVE_SEMANTIC_FIELDS),
        "aliases_active": False,
        "pronunciation_hints_active": False,
        "authoritative": False,
        "decision_effect": "legacy_domain_glossary_compatibility_only",
    }
