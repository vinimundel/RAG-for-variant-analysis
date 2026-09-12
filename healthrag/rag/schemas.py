"""Evidence provenance shared by retrieval, generation and evaluation."""

from typing import Literal
from pydantic import BaseModel, Field

EvidenceScope = Literal["exact_variant", "same_residue_analogy", "functional_region", "general_context"]

class LiteratureEvidence(BaseModel):
    evidence_id: str
    pmid: str
    article_title: str | None = None
    doi: str | None = None
    publication: str | None = None
    publication_types: list[str] = Field(default_factory=list)
    publication_type: str | None = None
    passage: str = Field(max_length=2500)
    retrieval_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
    page_number: int | None = None
    section: str | None = None
    source_file: str
    parser: str | None = None
    source_collection: str = "literature_csv"
    source_type: Literal["grobid_full_text", "pubmed_abstract"] = "grobid_full_text"
    full_text_available: bool = False
    provenance: str = ""
    evidence_type: Literal[
        "primary_evidence", "context_reference", "excluded", "unlayered",
    ] = "unlayered"
    # The curated layer of the source, carried so a review can never be read as
    # primary evidence downstream. See src/literature/corpus_layers.py.
    corpus_layer: Literal[
        "primary_evidence", "context_reference", "excluded", "unlayered",
    ] = "unlayered"
    retrieval_channels: list[str] = Field(default_factory=list)
    evidence_scope: EvidenceScope = "general_context"
