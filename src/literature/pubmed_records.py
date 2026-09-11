"""Full PubMed article records used to decide corpus layer and verifiability.

The abstract acquisition in :mod:`src.data.pubmed_abstracts` keeps only title,
abstract and MeSH descriptors, which is not enough to separate a primary study
from a review, a preprint or a retracted paper.  This module parses the same
EFetch XML into an auditable record: publication types, comments/corrections
relations, authors with affiliations, identifiers and dates.  Parsing is a pure
function of the payload so layer decisions are reproducible without network.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from lxml import etree
from pydantic import BaseModel, Field

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# RefTypes that describe how another record annuls or amends this one.  The
# "*In" direction points at the notice, so it marks the article as affected.
AFFECTING_REF_TYPES = {
    "RetractionIn", "ExpressionOfConcernIn", "ErratumIn",
    "CorrectedandRepublishedIn", "RepublishedIn",
}


class CommentCorrection(BaseModel):
    ref_type: str
    pmid: str = ""
    ref_source: str = ""


class PubMedRecord(BaseModel):
    """One PubMed article as returned by EFetch, without derived judgement."""

    pmid: str
    title: str = ""
    abstract: str = ""
    journal: str = ""
    journal_iso: str = ""
    doi: str = ""
    pmcid: str = ""
    publication_types: list[str] = Field(default_factory=list)
    publication_type_uis: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    mesh_descriptor_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    year: int | None = None
    pub_date: str = ""
    date_revised: str = ""
    comments_corrections: list[CommentCorrection] = Field(default_factory=list)

    @property
    def affecting_notices(self) -> list[CommentCorrection]:
        """Notices that retract, amend or question this article."""
        return [item for item in self.comments_corrections
                if item.ref_type in AFFECTING_REF_TYPES]


def _text(node, xpath: str) -> str:
    values = node.xpath(xpath)
    return " ".join("".join(value.itertext()) if hasattr(value, "itertext") else str(value)
                    for value in values).strip()


def _abstract(article) -> str:
    parts = []
    for part in article.xpath(".//Article/Abstract/AbstractText"):
        label, value = part.get("Label"), " ".join(part.itertext()).strip()
        if value:
            parts.append(f"{label}: {value}" if label else value)
    return "\n".join(parts)


def _authors(article) -> tuple[list[str], list[str]]:
    names, affiliations = [], []
    for author in article.xpath(".//Article/AuthorList/Author"):
        last, fore = _text(author, "./LastName"), _text(author, "./ForeName")
        collective = _text(author, "./CollectiveName")
        name = f"{last}, {fore}".strip(", ") or collective
        if name:
            names.append(name)
        affiliations.extend(
            value for value in (_text(node, ".") for node in
                                author.xpath("./AffiliationInfo/Affiliation"))
            if value
        )
    # Affiliations are deduplicated while preserving order so the count reflects
    # distinct institutions rather than repeated author-level annotations.
    return names, list(dict.fromkeys(affiliations))


def _pub_date(article) -> tuple[str, int | None]:
    node = article.xpath(".//Article/Journal/JournalIssue/PubDate")
    if not node:
        return "", None
    year, medline = _text(node[0], "./Year"), _text(node[0], "./MedlineDate")
    month, day = _text(node[0], "./Month"), _text(node[0], "./Day")
    raw = "-".join(part for part in (year, month, day) if part) or medline
    resolved = year or (medline[:4] if medline[:4].isdigit() else "")
    return raw, int(resolved) if resolved.isdigit() else None


def parse_pubmed_articles(payload: bytes) -> list[PubMedRecord]:
    """Parse an EFetch XML payload into records, ignoring entries without a PMID."""
    root = etree.fromstring(payload)
    records = []
    for article in root.xpath(".//PubmedArticle"):
        pmid = _text(article, ".//MedlineCitation/PMID[1]")
        if not pmid:
            continue
        identifiers = {str(node.get("IdType", "")).lower(): "".join(node.itertext()).strip()
                       for node in article.xpath(".//PubmedData/ArticleIdList/ArticleId")}
        authors, affiliations = _authors(article)
        pub_date, year = _pub_date(article)
        records.append(PubMedRecord(
            pmid=pmid,
            title=_text(article, ".//Article/ArticleTitle[1]"),
            abstract=_abstract(article),
            journal=_text(article, ".//Article/Journal/Title[1]"),
            journal_iso=_text(article, ".//Article/Journal/ISOAbbreviation[1]"),
            doi=identifiers.get("doi", ""),
            pmcid=identifiers.get("pmc", ""),
            publication_types=[t.strip() for t in article.xpath(
                ".//Article/PublicationTypeList/PublicationType/text()") if t.strip()],
            publication_type_uis=[str(node.get("UI", "")) for node in article.xpath(
                ".//Article/PublicationTypeList/PublicationType")],
            mesh_terms=[t.strip() for t in article.xpath(
                ".//MeshHeading/DescriptorName/text()") if t.strip()],
            mesh_descriptor_ids=[str(value) for value in article.xpath(
                ".//MeshHeading/DescriptorName/@UI")],
            keywords=[t.strip() for t in article.xpath(".//KeywordList/Keyword/text()")
                      if t.strip()],
            authors=authors, affiliations=affiliations,
            year=year, pub_date=pub_date,
            date_revised="-".join(part for part in (
                _text(article, ".//MedlineCitation/DateRevised/Year"),
                _text(article, ".//MedlineCitation/DateRevised/Month"),
                _text(article, ".//MedlineCitation/DateRevised/Day")) if part),
            comments_corrections=[CommentCorrection(
                ref_type=str(node.get("RefType", "")),
                pmid=_text(node, "./PMID"),
                ref_source=_text(node, "./RefSource"),
            ) for node in article.xpath(".//CommentsCorrectionsList/CommentsCorrections")],
        ))
    return records


def load_cache(cache_path: Path) -> dict[str, PubMedRecord]:
    """Read the JSONL record cache; a missing cache is an empty cache."""
    if not Path(cache_path).exists():
        return {}
    records = {}
    for line in Path(cache_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = PubMedRecord.model_validate_json(line)
            records[record.pmid] = record
    return records


def write_cache(cache_path: Path, records: dict[str, PubMedRecord]) -> Path:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for pmid in sorted(records, key=int):
            handle.write(records[pmid].model_dump_json() + "\n")
    return cache_path


def fetch_records(pmids: list[str], cache_path: Path, batch_size: int = 150,
                  retries: int = 4, gene: str = "UNKNOWN") -> dict[str, PubMedRecord]:
    """Fetch missing PMIDs through EFetch, persisting after every batch.

    PMIDs already cached are never re-requested, so a resumed run costs nothing.
    A batch that keeps failing raises instead of returning a partial corpus.
    """
    wanted = [str(pmid).strip().removesuffix(".0") for pmid in pmids]
    wanted = sorted({pmid for pmid in wanted if pmid.isdigit()}, key=int)
    records = load_cache(cache_path)
    pending = [pmid for pmid in wanted if pmid not in records]
    session, api_key = requests.Session(), os.getenv("NCBI_API_KEY")
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        data = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml",
                "tool": f"{gene.lower()}_structural_variant_ranker",
                "email": "researcher@epistasis-predictor.local"}
        if api_key:
            data["api_key"] = api_key
        error = None
        for attempt in range(retries):
            try:
                response = session.post(EFETCH_URL, data=data, timeout=90)
                response.raise_for_status()
                for record in parse_pubmed_articles(response.content):
                    records[record.pmid] = record
                error = None
                break
            except Exception as exc:
                error = exc
                time.sleep(min(8, 2 ** attempt))
        if error is not None:
            raise RuntimeError(f"PubMed EFetch failed for batch beginning {batch[0]}") from error
        write_cache(cache_path, records)
        time.sleep(0.11 if api_key else 0.34)
        print(f"PubMed records: {min(start + len(batch), len(pending))}/{len(pending)}")
    write_cache(cache_path, records)
    missing = [pmid for pmid in wanted if pmid not in records]
    if missing:
        # PubMed silently drops deleted or merged PMIDs; they must stay visible.
        (Path(cache_path).with_name(Path(cache_path).stem + "_unretrieved.json")).write_text(
            json.dumps({"unretrieved_pmids": missing}, indent=2), encoding="utf-8"
        )
    return records
