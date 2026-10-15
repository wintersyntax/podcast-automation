"""External curated terminology catalogs and candidate-registry matching.

External catalogs are deliberately *not* compiler policy. They may be large and
may contain terms that never occur in Podcast Worker. The bridge in this module
keeps them outside the mutable candidate registry until a term already observed
in the Podcast Worker corpus/Human Review data has an exact unambiguous catalog
match. Even then the result is only an ``external-curated`` candidate
observation; active compiler terminology still requires explicit policy review.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from .terminology_candidates import (
    CandidateObservation,
    CandidateRegistryStore,
    candidate_is_term_like,
    normalize_candidate_text,
    validate_candidate_registry,
)


EXTERNAL_CATALOG_SCHEMA_VERSION = 1
EXTERNAL_CATALOG_AUTHORITY = "none"
EXTERNAL_CATALOG_COMPILER_VISIBILITY = "none"

_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
_OWL = "{http://www.w3.org/2002/07/owl#}"
_SKOS = "{http://www.w3.org/2004/02/skos/core#}"
_OBO = "{http://www.geneontology.org/formats/oboInOwl#}"


@dataclass(frozen=True)
class ExternalTerm:
    canonical: str
    source_id: str
    aliases: tuple[str, ...] = ()


def _clean_term(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    if not candidate_is_term_like(cleaned):
        return None
    return cleaned


def _dedupe_terms(values: Iterable[str], *, exclude: str | None = None) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    excluded = normalize_candidate_text(exclude) if exclude else None
    for value in values:
        cleaned = _clean_term(value)
        if cleaned is None:
            continue
        key = normalize_candidate_text(cleaned)
        if key == excluded or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return tuple(output)


def parse_mesh_xml(path: Path) -> list[ExternalTerm]:
    """Parse an official MeSH Descriptor or Supplemental XML release file.

    The parser intentionally extracts only source identifiers, preferred labels,
    and term synonyms. Tree-number semantics are not used to make compiler
    decisions and category mapping remains a later human/corpus concern.
    """

    terms: list[ExternalTerm] = []
    for _event, element in ET.iterparse(path, events=("end",)):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in {"DescriptorRecord", "SupplementalRecord"}:
            continue
        ui = element.findtext("DescriptorUI") or element.findtext("SupplementalRecordUI")
        canonical = (
            element.findtext("./DescriptorName/String")
            or element.findtext("./SupplementalRecordName/String")
        )
        cleaned = _clean_term(canonical)
        if not ui or cleaned is None:
            element.clear()
            continue
        aliases = _dedupe_terms(
            (
                node.text or ""
                for node in element.findall(".//TermList/Term/String")
            ),
            exclude=cleaned,
        )
        terms.append(ExternalTerm(canonical=cleaned, source_id=ui.strip(), aliases=aliases))
        element.clear()
    return terms


def parse_rdf_xml(path: Path) -> list[ExternalTerm]:
    """Parse labels/synonyms from a standard RDF/XML ontology file.

    This intentionally supports only common label/synonym predicates and does
    not interpret ontology relationships as transcript evidence.
    """

    terms: list[ExternalTerm] = []
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag != _OWL + "Class":
            continue
        source_id = element.attrib.get(_RDF + "about") or element.attrib.get(_RDF + "ID")
        labels = [node.text or "" for node in element.findall(_RDFS + "label")]
        labels.extend(node.text or "" for node in element.findall(_SKOS + "prefLabel"))
        canonical = next((_clean_term(label) for label in labels if _clean_term(label)), None)
        if not source_id or canonical is None:
            element.clear()
            continue
        synonyms: list[str] = []
        for tag in (
            _OBO + "hasExactSynonym",
            _OBO + "hasRelatedSynonym",
            _OBO + "hasBroadSynonym",
            _OBO + "hasNarrowSynonym",
            _SKOS + "altLabel",
        ):
            synonyms.extend(node.text or "" for node in element.findall(tag))
        aliases = _dedupe_terms(synonyms, exclude=canonical)
        terms.append(ExternalTerm(canonical=canonical, source_id=source_id.strip(), aliases=aliases))
        element.clear()
    return terms


def build_external_catalog(
    *,
    source_name: str,
    source_version: str,
    source_note: str,
    terms: Iterable[ExternalTerm],
    source_url: str | None = None,
    license_note: str | None = None,
) -> dict[str, Any]:
    if not source_name.strip() or not source_version.strip() or not source_note.strip():
        raise ValueError("external catalog source name/version/note are required")
    deduped: dict[str, ExternalTerm] = {}
    for term in terms:
        canonical = _clean_term(term.canonical)
        if canonical is None or not term.source_id.strip():
            continue
        key = normalize_candidate_text(canonical)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = ExternalTerm(
                canonical=canonical,
                source_id=term.source_id.strip(),
                aliases=_dedupe_terms(term.aliases, exclude=canonical),
            )
            continue
        # Duplicate preferred labels are legal in some ontologies. Preserve a
        # stable first record; ambiguity is handled by the lookup layer because
        # aliases/source IDs remain source evidence, not compiler truth.
    payload = {
        "schema_version": EXTERNAL_CATALOG_SCHEMA_VERSION,
        "authority": EXTERNAL_CATALOG_AUTHORITY,
        "compiler_visibility": EXTERNAL_CATALOG_COMPILER_VISIBILITY,
        "source": {
            "name": source_name.strip(),
            "version": source_version.strip(),
            "note": source_note.strip(),
            "url": source_url.strip() if isinstance(source_url, str) and source_url.strip() else None,
            "license_note": (
                license_note.strip()
                if isinstance(license_note, str) and license_note.strip()
                else None
            ),
        },
        "terms": [
            {
                "canonical": term.canonical,
                "source_id": term.source_id,
                "aliases": list(term.aliases),
            }
            for _, term in sorted(deduped.items())
        ],
    }
    payload["sha256"] = external_catalog_sha256(payload)
    return validate_external_catalog(payload)


def external_catalog_sha256(payload: dict[str, Any]) -> str:
    material = {key: value for key, value in payload.items() if key != "sha256"}
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_external_catalog(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("external catalog must be an object")
    if payload.get("schema_version") != EXTERNAL_CATALOG_SCHEMA_VERSION:
        raise ValueError("external catalog schema is unsupported")
    if payload.get("authority") != EXTERNAL_CATALOG_AUTHORITY:
        raise ValueError("external catalog must remain non-authoritative")
    if payload.get("compiler_visibility") != EXTERNAL_CATALOG_COMPILER_VISIBILITY:
        raise ValueError("external catalog must remain compiler-invisible")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("external catalog source metadata is required")
    for field in ("name", "version", "note"):
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError(f"external catalog source {field} is required")
    terms = payload.get("terms")
    if not isinstance(terms, list):
        raise ValueError("external catalog terms must be a list")
    seen: set[str] = set()
    for index, term in enumerate(terms):
        if not isinstance(term, dict):
            raise ValueError(f"external catalog term {index} must be an object")
        canonical = _clean_term(term.get("canonical"))
        if canonical is None:
            raise ValueError(f"external catalog term {index} canonical is invalid")
        key = normalize_candidate_text(canonical)
        if key in seen:
            raise ValueError("external catalog contains duplicate canonical labels")
        seen.add(key)
        if not isinstance(term.get("source_id"), str) or not term["source_id"].strip():
            raise ValueError(f"external catalog term {index} source_id is required")
        aliases = term.get("aliases")
        if not isinstance(aliases, list):
            raise ValueError(f"external catalog term {index} aliases must be a list")
        _dedupe_terms(aliases, exclude=canonical)
    expected = external_catalog_sha256(payload)
    if payload.get("sha256") != expected:
        raise ValueError("external catalog fingerprint does not match content")
    return payload


def external_catalog_stats(payload: dict[str, Any]) -> dict[str, Any]:
    validate_external_catalog(payload)
    alias_count = sum(len(term["aliases"]) for term in payload["terms"])
    return {
        "source": payload["source"]["name"],
        "version": payload["source"]["version"],
        "term_count": len(payload["terms"]),
        "alias_count": alias_count,
        "sha256": payload["sha256"],
        "authority": payload["authority"],
        "compiler_visibility": payload["compiler_visibility"],
    }


def _catalog_lookup(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    validate_external_catalog(payload)
    lookup: dict[str, list[dict[str, Any]]] = {}
    for term in payload["terms"]:
        for form in [term["canonical"], *term["aliases"]]:
            cleaned = _clean_term(form)
            if cleaned is None:
                continue
            lookup.setdefault(normalize_candidate_text(cleaned), []).append(term)
    return lookup


def external_matches_for_registry(
    registry: dict[str, Any], catalog: dict[str, Any]
) -> list[CandidateObservation]:
    """Return idempotent external-curated observations for exact unique matches."""

    validate_candidate_registry(registry)
    validate_external_catalog(catalog)
    lookup = _catalog_lookup(catalog)
    catalog_sha = catalog["sha256"]
    source = catalog["source"]
    observations: list[CandidateObservation] = []
    for entry in registry["candidates"].values():
        if entry.get("status") != "candidate":
            continue
        already_matched = any(
            isinstance(sample, dict)
            and sample.get("external_catalog_sha256") == catalog_sha
            for sample in entry.get("evidence_samples", [])
        )
        if already_matched:
            continue
        matched_term: dict[str, Any] | None = None
        matched_form: str | None = None
        ambiguous = False
        forms = [entry["canonical"], *entry.get("observed_forms", [])]
        for form in forms:
            matches = lookup.get(normalize_candidate_text(form), [])
            unique_ids = {match["source_id"] for match in matches}
            if len(unique_ids) > 1:
                ambiguous = True
                break
            if len(matches) == 1:
                if matched_term is not None and matched_term["source_id"] != matches[0]["source_id"]:
                    ambiguous = True
                    break
                matched_term = matches[0]
                matched_form = form
        if ambiguous or matched_term is None or matched_form is None:
            continue
        categories = entry.get("suggested_categories", {})
        if not categories:
            continue
        category = max(categories, key=lambda name: (categories[name], name))
        observations.append(
            CandidateObservation(
                text=entry["canonical"],
                suggested_category=category,
                provenance="external-curated",
                observed_form=matched_form,
                evidence={
                    "external_source": source["name"],
                    "external_version": source["version"],
                    "external_source_id": matched_term["source_id"],
                    "external_canonical": matched_term["canonical"],
                    "external_catalog_sha256": catalog_sha,
                    "matched_form": matched_form,
                },
            )
        )
    return observations


def match_catalog_into_candidate_store(
    catalog: dict[str, Any], *, store: CandidateRegistryStore | None = None
) -> dict[str, Any]:
    store = store or CandidateRegistryStore()
    registry = store.load()
    observations = external_matches_for_registry(registry, catalog)
    result = store.record_many(observations)
    return {
        "catalog": external_catalog_stats(catalog),
        "matched_candidates": len(observations),
        "write_result": result,
    }
