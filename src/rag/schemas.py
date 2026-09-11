"""Structured contracts for evidence and mechanistic literature hypotheses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GenerationMode = Literal["deterministic", "llm_validated", "biophysical_rule"]
EvidenceScope = Literal[
    "exact_variant", "same_residue_analogy", "functional_region", "general_context",
]
InferenceStatus = Literal[
    "llm_validated", "deterministic_fallback", "no_evidence", "technical_llm_failure",
]
MechanisticEvidenceStatus = Literal[
    "mechanistic_evidence_retrieved", "no_mechanistic_evidence_retrieved",
]


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


class MechanisticLiteratureInference(BaseModel):
    variant: str
    gene: str
    mechanism: str = Field(description="Specific proposed mechanism, for example oligomerization")
    verdict: Literal["supported", "contradicted", "insufficient_evidence"]
    predicted_direction: Literal["decrease", "increase", "unclear"]
    rationale: str
    cited_evidence_ids: list[str]
    uncertainty: str
    requires_experimental_validation: bool = True


class MechanisticQuery(BaseModel):
    gene: str
    variant: str
    mechanism: str
    structural_observation: str
    k: int = Field(default=5, ge=1, le=10)


class MechanisticRAGResult(BaseModel):
    inference: MechanisticLiteratureInference
    evidence: list[LiteratureEvidence]
    retrieval_query: str
    retrieval_version: str = "bm25_medcpt_qdrant_rrf_crossencoder_v2"
    llm_contract_status: Literal["passed", "abstained_after_invalid_output"] = "passed"


class DiseaseAssociation(BaseModel):
    status: Literal["associated", "cooccurrence", "none"] = "none"
    disease_context: str = ""
    rationale: str = ""
    cited_evidence_ids: list[str] = []
    generation_mode: GenerationMode = "llm_validated"


class DiscoveryHypothesis(BaseModel):
    mechanism: str
    target: str
    predicted_direction: str
    conformational_state: str = "not_specified"
    classification: Literal[
        "literature_reported", "literature_supported_hypothesis",
        "biophysical_hypothesis", "context_only",
    ]
    rationale: str
    cited_evidence_ids: list[str] = []
    rule_ids: list[str] = []
    evidence_conflict: bool = False
    requires_experimental_validation: bool = True
    generation_mode: GenerationMode = "llm_validated"
    evidence_scope: EvidenceScope = "general_context"


class DiscoveryInference(BaseModel):
    variant: str
    gene: str
    disease_association: DiseaseAssociation = DiseaseAssociation()
    hypotheses: list[DiscoveryHypothesis] = Field(default_factory=list, max_length=3)
    evidence_conflict_flag: bool = False
    uncertainty: str


class DiscoveryQuery(BaseModel):
    gene: str
    variant: str
    position: int
    functional_region: str
    ppi_partner: str | None = None
    structural_observation: str
    biophysical_hypotheses_json: str
    condition_profile: str = "general_gsdmd"
    condition_terms: list[str] = Field(default_factory=list)
    query_version: str = "gsdmd_modular_v1"
    # Pharmacological evidence is opt-in and must be requested explicitly, so a
    # mechanistic question is never answered with therapeutic response data.
    pharmacology_requested: bool = False
    k: int = Field(default=10, ge=1, le=10)


class DiscoveryRAGResult(BaseModel):
    inference: DiscoveryInference
    evidence: list[LiteratureEvidence]
    # Explicit blocks prevent general pathway/therapy context from being
    # mistaken for a mechanistic top-k result. ``evidence`` remains the
    # compatibility union used by older consumers and is never the benchmark
    # denominator for V6.
    mechanistic_evidence: list[LiteratureEvidence] = Field(default_factory=list)
    context_reference: list[LiteratureEvidence] = Field(default_factory=list)
    mechanistic_evidence_status: MechanisticEvidenceStatus = (
        "no_mechanistic_evidence_retrieved"
    )
    retrieval_queries: list[str]
    retrieval_version: str = "discovery_v3"
    condition_profile: str = "general_gsdmd"
    condition_terms: list[str] = Field(default_factory=list)
    query_version: str = "gsdmd_modular_v1"
    llm_contract_status: Literal["passed", "deterministic_fallback"] = "passed"
    inference_status: InferenceStatus = "llm_validated"
    evidence_status: Literal["exact", "indirect", "none", "technical_failure"] = "none"
    llm_retry_count: int = 0
    llm_failure_reason: str | None = None
    retrieval_index_fingerprint: str | None = None
    retrieval_stack_fingerprint: str | None = None
    retrieval_stack_manifest_sha256: str | None = None
