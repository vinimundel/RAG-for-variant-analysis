"""Deterministic assignment of every BRAF literature record to one layer.

Three layers are kept, and no record is ever deleted:

``primary_evidence``
    Peer-reviewed primary articles eligible to support a mechanistic claim.
``context_reference``
    Reviews, meta-analyses and guidelines. Usable for context and for mining
    references, never as direct support for a mechanism or a disease link.
``excluded``
    Preprints, retracted records, unresolved expressions of concern, notices,
    duplicates, records without provenance and records with no verifiable link
    to the gene. Excluded records are retained for audit but never indexed.

Journal, citation count and open-access status are metadata and never enter the
decision.  Every assignment carries the rule that produced it, so the layered
corpus can be regenerated and diffed from the cached PubMed records alone.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from pydantic import BaseModel, Field

from src.literature.pubmed_records import PubMedRecord

Layer = Literal["primary_evidence", "context_reference", "excluded"]
# Layers that may be retrieved. ``excluded`` is never indexed: it exists to be
# audited, not to be searched.
INDEXED_LAYERS = ("primary_evidence", "context_reference")
# A record with no entry in the layered corpus. It is kept distinct from every
# real layer so an unlayered corpus can never be mistaken for a curated one.
UNLAYERED = "unlayered"

# Publication types are matched by their PubMed label, lowercased.  Labels are
# stable across releases, while UIs are not always present in every payload.
CONTEXT_TYPES = {
    "review", "systematic review", "meta-analysis", "scoping review",
    "practice guideline", "guideline", "consensus development conference",
    "consensus development conference, nih",
}
NOTICE_TYPES = {
    "editorial", "comment", "news", "newspaper article", "published erratum",
    "retraction of publication", "expression of concern", "biography",
    "interview", "address", "patient education handout", "video-audio media",
    "webcast", "bibliography", "directory", "dictionary",
}
PREPRINT_TYPES = {"preprint"}
RETRACTED_TYPES = {"retracted publication"}

_PLACEHOLDER_IDENTIFIERS = {"", "n/a", "na", "none", "unknown", "cached"}
_SCRAPING_JUNK = re.compile(
    r"(?:^|\b)(?:access denied|page not found|search results?|captcha|javascript|"
    r"cookie policy|internal server error|error loading|untitled)(?:\b|$)", re.I)

GeneLink = Literal[
    "title", "abstract", "mesh", "keyword", "absent", "unverifiable_no_text",
    "unverified_no_pubmed_record",
]


class CorpusAssignment(BaseModel):
    """The layer of one record plus the auditable reason behind it."""

    pmid: str
    layer: Layer
    rule_id: str
    exclusion_reason: str = ""
    gene_link: GeneLink
    publication_types: list[str] = Field(default_factory=list)
    duplicate_of: str = ""
    affecting_notices: list[str] = Field(default_factory=list)
    full_text_available: bool = False
    manual_review_status: Literal["not_reviewed", "overridden"] = "not_reviewed"
    manual_review_note: str = ""


# A symbol is routinely written fused to its variant, as in BRAFV600E or
# BrafV600MUT, so the trailing boundary also accepts an inline variant token.
INLINE_VARIANT = r"[ACDEFGHIKLMNPQRSTVWY]\d+(?:[ACDEFGHIKLMNPQRSTVWY]|MUT|MUTANT|WT)?"
# Only these prefixes may precede the symbol without a separator.  The leading
# boundary stays strict otherwise, which is what keeps dabrafenib, vemurafenib
# and encorafenib out of the corpus.
FUSED_PREFIXES = ("anti", "non", "pan", "mut")
_LEADING_BOUNDARY = "(?:(?<![A-Za-z0-9])" + "".join(
    f"|(?<={prefix})" for prefix in FUSED_PREFIXES) + ")"


def _alias_pattern(aliases: Iterable[str]) -> re.Pattern[str]:
    """Boundary-safe alternation over gene aliases, longest alias first."""
    ordered = sorted({str(alias).strip() for alias in aliases if str(alias).strip()},
                     key=len, reverse=True)
    if not ordered:
        raise ValueError("At least one gene alias is required to verify a gene link")
    body = "|".join(re.escape(alias) for alias in ordered)
    return re.compile(
        rf"{_LEADING_BOUNDARY}(?:{body})(?:{INLINE_VARIANT})?(?![A-Za-z0-9])", re.I)


def gene_link_status(record: PubMedRecord, aliases: Iterable[str],
                     mesh_terms: Iterable[str] = ()) -> GeneLink:
    """Where the gene is mentioned, or why the mention could not be verified.

    ``unverifiable_no_text`` is distinct from ``absent``: a record without title
    and abstract is missing evidence of a link, which is not evidence of no link.
    """
    pattern = _alias_pattern(aliases)
    if pattern.search(record.title):
        return "title"
    if pattern.search(record.abstract):
        return "abstract"
    wanted_mesh = {str(term).strip().lower() for term in mesh_terms if str(term).strip()}
    if any(term.lower() in wanted_mesh for term in record.mesh_terms):
        return "mesh"
    if any(pattern.search(keyword) for keyword in record.keywords):
        return "keyword"
    if not record.title.strip() and not record.abstract.strip():
        return "unverifiable_no_text"
    return "absent"


def gene_mention_count(record: PubMedRecord, aliases: Iterable[str]) -> int:
    """How often the gene is named in title and abstract.

    Reported as metadata, never as a layer rule.  A single peripheral mention
    still yields a verifiable link; whether the paper is *about* the gene is
    decided per claim, through ``variant_directness`` and ``support_type``.
    """
    pattern = _alias_pattern(aliases)
    return len(pattern.findall(f"{record.title}\n{record.abstract}"))


def _has_provenance(record: PubMedRecord) -> bool:
    """Require a traceable identifier and title before scientific classification."""
    pmid = str(record.pmid).strip().lower()
    doi = str(record.doi).strip().lower()
    return bool(record.title.strip()) and (
        pmid.isdigit() or doi not in _PLACEHOLDER_IDENTIFIERS
    )


def _is_scraping_junk(record: PubMedRecord) -> bool:
    """Recognize obvious retrieval pages without treating generic titles as junk."""
    title = record.title.strip()
    return bool(title and _SCRAPING_JUNK.search(title))


def assign_layer(record: PubMedRecord, aliases: Iterable[str],
                 mesh_terms: Iterable[str] = (), duplicate_of: str = "",
                 full_text_available: bool = False,
                 manual_override: dict | None = None) -> CorpusAssignment:
    """Assign one record to a layer under a fixed rule precedence.

    Precedence is identity first (duplicates), then integrity (retraction and
    expression of concern), then publication type, then the gene link.  An
    integrity problem therefore always outranks an attractive publication type.
    """
    types = {str(value).strip().lower() for value in record.publication_types}
    notices = [item.ref_type for item in record.affecting_notices]
    base = {
        "pmid": record.pmid, "gene_link": gene_link_status(record, aliases, mesh_terms),
        "publication_types": list(record.publication_types), "duplicate_of": duplicate_of,
        "affecting_notices": notices, "full_text_available": full_text_available,
    }

    def excluded(rule: str, reason: str) -> CorpusAssignment:
        return CorpusAssignment(layer="excluded", rule_id=rule, exclusion_reason=reason, **base)

    if not _has_provenance(record):
        reason = "scraping_junk" if _is_scraping_junk(record) else "missing_pmid_or_doi_or_title"
        rule = "scraping_junk" if reason == "scraping_junk" else "missing_provenance"
        return excluded(rule, reason)
    if manual_override:
        return CorpusAssignment(
            layer=manual_override["layer"], rule_id="manual_override",
            exclusion_reason=manual_override.get("reason", ""),
            manual_review_status="overridden",
            manual_review_note=manual_override.get("note", ""), **base,
        )
    if duplicate_of:
        return excluded("duplicate_record", "duplicate_of_kept_record")
    if types & RETRACTED_TYPES or "RetractionIn" in notices:
        return excluded("retracted", "retracted_publication")
    if "ExpressionOfConcernIn" in notices:
        # Resolution cannot be inferred from the notice itself; a reviewer must
        # record it explicitly through a manual override.
        return excluded("expression_of_concern", "expression_of_concern_unresolved")
    if types & NOTICE_TYPES:
        return excluded("non_research_publication_type",
                        f"publication_type:{sorted(types & NOTICE_TYPES)[0]}")
    if base["gene_link"] == "absent":
        return excluded("gene_link", "gene_absent_from_title_abstract_mesh")
    if base["gene_link"] == "unverifiable_no_text":
        return excluded("gene_link", "gene_link_unverifiable_no_text")
    if types & PREPRINT_TYPES:
        return excluded("preprint", "preprint_not_eligible_for_braf_corpus")
    if types & CONTEXT_TYPES:
        return CorpusAssignment(
            layer="context_reference", rule_id="review_or_guideline", **base)
    return CorpusAssignment(layer="primary_evidence", rule_id="primary_article", **base)


def load_layer_map(layers_csv: Path) -> dict[str, str]:
    """Read ``pmid -> layer`` from a built corpus; a missing corpus is empty.

    Callers must treat an empty map as "not curated yet" and fall back to
    ``UNLAYERED``, never to ``primary_evidence``.
    """
    path = Path(layers_csv)
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"pmid": str})
    return {
        str(row["pmid"]): str(row["layer"])
        for row in frame.to_dict("records")
        if str(row.get("pmid", "")).strip() and str(row.get("pmid", "")).lower() != "nan"
    }


def _duplicate_map(records: list[PubMedRecord]) -> dict[str, str]:
    """Map each duplicated PMID to the first record kept for the same DOI."""
    kept_by_doi: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for record in records:
        doi = record.doi.strip().lower()
        if not doi:
            continue
        if doi in kept_by_doi:
            duplicates[record.pmid] = kept_by_doi[doi]
        else:
            kept_by_doi[doi] = record.pmid
    return duplicates


def build_corpus_layers(records: Iterable[PubMedRecord], aliases: Iterable[str],
                        mesh_terms: Iterable[str] = (),
                        full_text_pmids: Iterable[str] = (),
                        manual_overrides: dict[str, dict] | None = None) -> pd.DataFrame:
    """Layer a whole corpus; the returned frame keeps one row per unique PMID."""
    unique: dict[str, PubMedRecord] = {}
    for record in records:
        unique.setdefault(record.pmid, record)
    ordered = sorted(unique.values(), key=lambda item: int(item.pmid))
    duplicates = _duplicate_map(ordered)
    with_text = {str(pmid) for pmid in full_text_pmids}
    overrides = manual_overrides or {}
    rows = []
    for record in ordered:
        assignment = assign_layer(
            record, aliases, mesh_terms,
            duplicate_of=duplicates.get(record.pmid, ""),
            full_text_available=record.pmid in with_text,
            manual_override=overrides.get(record.pmid),
        )
        rows.append({
            **assignment.model_dump(),
            "affecting_notices": ";".join(assignment.affecting_notices),
            "publication_types": ";".join(assignment.publication_types),
            "title": record.title, "journal": record.journal, "year": record.year,
            "doi": record.doi, "pmcid": record.pmcid,
            "has_abstract": bool(record.abstract.strip()),
            "gene_mentions": gene_mention_count(record, aliases),
            "section": "", "parser": "pubmed_record",
            "publication": record.journal,
            "evidence_type": assignment.layer,
        })
    return pd.DataFrame(rows)
