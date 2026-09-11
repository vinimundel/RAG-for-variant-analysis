"""Ordinal evidence grade for one literature claim, derived from auditable dimensions.

The grade is a function of what a claim actually demonstrates, never of journal,
impact factor, citation count or open-access status.  It is reported alongside a
prediction as a separate axis and is never summed into, nor used as a feature
of, any functional probability.

Grades
------
``A``  exact variant, primary controlled study, replicated independently or by
       orthogonal assays.
``B``  exact variant in one controlled primary study with verified full text, or
       the same residue supported by independent primary studies.
``C``  regional, computational, abstract-only or indirect mechanistic transfer.
``D``  general context, review-only support, insufficient support, or an
       unresolved conflict between sources.

Only ``A`` and ``B`` may be labelled ``literature_supported``.  ``C`` yields
``mechanistic_hypothesis`` and ``D`` yields ``context``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

VariantDirectness = Literal["exact_variant", "same_residue", "functional_region", "general"]
SupportType = Literal[
    "structural_biophysical", "biochemical", "cell_functional", "in_vivo",
    "human_association", "computational", "context",
]
ControlsReported = Literal["yes", "no", "unknown"]
ConflictStatus = Literal["none", "resolved", "unresolved"]
Grade = Literal["A", "B", "C", "D"]
Label = Literal["literature_supported", "mechanistic_hypothesis", "context"]
GenerationMode = Literal["deterministic", "llm_validated", "biophysical_rule"]

MECHANISTIC_SUPPORT = {"structural_biophysical", "biochemical", "cell_functional", "in_vivo"}
ASSOCIATION_SUPPORT = {"human_association"}
# Two distinct assay modalities are the minimum that makes support orthogonal.
MIN_ORTHOGONAL_ASSAYS = 2
MIN_INDEPENDENT_PAPERS = 2

LABEL_BY_GRADE: dict[Grade, Label] = {
    "A": "literature_supported", "B": "literature_supported",
    "C": "mechanistic_hypothesis", "D": "context",
}


class ClaimEvidence(BaseModel):
    """Dimensions recorded for one claim before any grade is computed."""

    claim_id: str
    variant: str
    claim_text: str = ""
    variant_directness: VariantDirectness
    support_type: SupportType
    controls_reported: ControlsReported = "unknown"
    orthogonal_assays: int = Field(default=0, ge=0)
    independent_papers: int = Field(default=0, ge=0)
    full_text_verified: bool = False
    conflict_status: ConflictStatus = "none"
    # PMIDs are carried so a grade can always be traced back to its sources.
    primary_evidence_pmids: list[str] = Field(default_factory=list)
    context_reference_pmids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    verifiable_full_text_dimensions: list[str] = Field(default_factory=list)
    generation_mode: GenerationMode = "llm_validated"
    manual_review_status: Literal["not_reviewed", "reviewed", "overridden"] = "not_reviewed"


class EvidenceGrade(BaseModel):
    claim_id: str
    grade: Grade
    label: Label
    rule_id: str
    reasons: list[str] = Field(default_factory=list)


def grade_claim(claim: ClaimEvidence) -> EvidenceGrade:
    """Grade one claim deterministically; identical dimensions always agree.

    Precedence is integrity first: an unresolved conflict or the absence of any
    primary source caps the claim at ``D`` regardless of how direct it looks.
    """
    reasons: list[str] = []

    def result(grade: Grade, rule: str) -> EvidenceGrade:
        return EvidenceGrade(claim_id=claim.claim_id, grade=grade,
                             label=LABEL_BY_GRADE[grade], rule_id=rule, reasons=reasons)

    if claim.conflict_status == "unresolved":
        reasons.append("conflicting sources were not resolved")
        return result("D", "unresolved_conflict")
    if not claim.primary_evidence_pmids:
        reasons.append("no source in the primary_evidence layer supports this claim")
        return result("D", "no_primary_source")
    if claim.support_type == "context":
        reasons.append("support type is contextual only")
        return result("D", "context_only_support")

    mechanistic = claim.support_type in MECHANISTIC_SUPPORT
    association = claim.support_type in ASSOCIATION_SUPPORT
    controlled = claim.controls_reported == "yes"
    replicated = claim.independent_papers >= MIN_INDEPENDENT_PAPERS
    orthogonal = claim.orthogonal_assays >= MIN_ORTHOGONAL_ASSAYS

    if not claim.full_text_verified:
        # Full text is verifiability, not scientific quality; an unverified
        # claim still stands as a hypothesis, it just cannot be called supported.
        reasons.append("claim was not verified against full text")
        return result("C", "abstract_only")
    if claim.variant_directness == "exact_variant" and mechanistic and controlled:
        if replicated or orthogonal:
            reasons.append("exact variant, controlled primary study, "
                           + ("independent replication" if replicated else "orthogonal assays"))
            return result("A", "exact_controlled_replicated")
        reasons.append("exact variant in a single controlled primary study")
        return result("B", "exact_controlled_single_study")
    if claim.variant_directness == "same_residue" and mechanistic and replicated:
        reasons.append("same residue supported by independent primary studies")
        return result("B", "same_residue_replicated")
    if claim.variant_directness == "exact_variant" and association and controlled and replicated:
        reasons.append("exact variant with replicated controlled human association")
        return result("B", "exact_association_replicated")
    if claim.variant_directness in {"functional_region", "general"}:
        reasons.append("support is regional rather than variant-specific")
        return result("C", "regional_transfer")
    if not mechanistic and not association:
        reasons.append("support is computational rather than experimental")
        return result("C", "computational_support")
    reasons.append("primary support present but insufficient for an exact controlled claim")
    return result("C", "insufficient_for_supported")


def grade_frame(claims: list[ClaimEvidence]) -> list[EvidenceGrade]:
    return [grade_claim(claim) for claim in claims]
