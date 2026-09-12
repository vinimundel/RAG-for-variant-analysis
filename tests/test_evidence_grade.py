"""Contract tests for the ordinal evidence grade."""

from __future__ import annotations

import pytest

from healthrag.literature.evidence_grade import ClaimEvidence, grade_claim

BASE = {
    "claim_id": "c1", "variant": "V600E", "variant_directness": "exact_variant",
    "support_type": "biochemical", "controls_reported": "yes",
    "full_text_verified": True, "primary_evidence_pmids": ["1"],
}


def claim(**fields) -> ClaimEvidence:
    return ClaimEvidence(**{**BASE, **fields})


def test_exact_controlled_and_replicated_claim_is_grade_a():
    result = grade_claim(claim(independent_papers=2, primary_evidence_pmids=["1", "2"]))
    assert (result.grade, result.label) == ("A", "literature_supported")


def test_orthogonal_assays_substitute_for_independent_replication():
    assert grade_claim(claim(orthogonal_assays=2)).grade == "A"
    assert grade_claim(claim(orthogonal_assays=1)).grade == "B"


def test_single_controlled_study_is_grade_b():
    result = grade_claim(claim())
    assert (result.grade, result.label) == ("B", "literature_supported")


def test_same_residue_needs_independent_studies_for_grade_b():
    replicated = claim(variant_directness="same_residue", independent_papers=2)
    single = claim(variant_directness="same_residue", independent_papers=1)
    assert grade_claim(replicated).grade == "B"
    assert grade_claim(single).grade == "C"


def test_regional_support_cannot_exceed_grade_c():
    for directness in ("functional_region", "general"):
        result = grade_claim(claim(variant_directness=directness, independent_papers=5,
                                   orthogonal_assays=5))
        assert (result.grade, result.label) == ("C", "mechanistic_hypothesis")


def test_computational_support_cannot_exceed_grade_c():
    assert grade_claim(claim(support_type="computational", independent_papers=5)).grade == "C"


def test_abstract_only_claim_is_capped_at_c():
    result = grade_claim(claim(full_text_verified=False, independent_papers=5,
                               orthogonal_assays=5))
    assert (result.grade, result.rule_id) == ("C", "abstract_only")


def test_uncontrolled_study_cannot_be_literature_supported():
    for controls in ("no", "unknown"):
        assert grade_claim(claim(controls_reported=controls)).label == "mechanistic_hypothesis"


def test_unresolved_conflict_outranks_every_other_dimension():
    result = grade_claim(claim(conflict_status="unresolved", independent_papers=9,
                               orthogonal_assays=9))
    assert (result.grade, result.label, result.rule_id) == ("D", "context", "unresolved_conflict")


def test_claim_without_a_primary_source_is_grade_d():
    result = grade_claim(claim(primary_evidence_pmids=[],
                               context_reference_pmids=["1", "2", "3"]))
    assert (result.grade, result.rule_id) == ("D", "no_primary_source")


def test_reviews_alone_can_never_produce_literature_supported():
    result = grade_claim(claim(support_type="context", primary_evidence_pmids=["1"]))
    assert result.label == "context"


@pytest.mark.parametrize("grade,label", [("A", "literature_supported"),
                                         ("B", "literature_supported"),
                                         ("C", "mechanistic_hypothesis"),
                                         ("D", "context")])
def test_label_is_a_pure_function_of_the_grade(grade, label):
    from healthrag.literature.evidence_grade import LABEL_BY_GRADE
    assert LABEL_BY_GRADE[grade] == label


def test_grading_is_deterministic():
    sample = claim(independent_papers=2)
    assert grade_claim(sample) == grade_claim(sample)


def test_grade_carries_its_reason():
    assert grade_claim(claim()).reasons
