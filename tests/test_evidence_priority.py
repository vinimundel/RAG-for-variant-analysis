import pandas as pd

from src.model.evidence_priority import add_discovery_priority, add_evidence_priority


def test_literature_priority_never_overwrites_structural_score():
    frame = pd.DataFrame({"stability_score": [5.0, 90.0],
                          "literature_verdict": ["supported", "insufficient_evidence"],
                          "literature_specificity": ["exact_variant", "none"]})
    result = add_evidence_priority(frame)
    assert result["stability_score"].tolist() == [5.0, 90.0]
    assert result["evidence_adjusted_priority_score"].tolist() == [25.0, 90.0]


def test_missing_structure_and_no_literature_support_remains_unknown():
    frame = pd.DataFrame({
        "stability_score": [float("nan"), float("nan")],
        "literature_verdict": ["insufficient_evidence", "supported"],
        "literature_specificity": ["none", "regional"],
    })
    result = add_evidence_priority(frame)
    assert pd.isna(result.loc[0, "evidence_adjusted_priority_score"])
    assert pd.isna(result.loc[0, "evidence_adjusted_priority_rank"])
    assert result.loc[1, "evidence_adjusted_priority_score"] == 7.5


def test_discovery_priority_caps_and_missing_structure():
    frame = pd.DataFrame({
        "stability_score": [100.0, float("nan")],
        "literature_mechanism_specificity": ["exact_variant", "functional_region"],
        "literature_mechanism_source_type": ["grobid_full_text", "pubmed_abstract"],
        "disease_association_status": ["associated", "cooccurrence"],
        "disease_association_source_type": ["grobid_full_text", "pubmed_abstract"],
        "biophysical_component": [99.0, 12.0],
    })
    result = add_discovery_priority(frame)
    assert result.loc[0, "variant_priority_score"] == 100.0
    assert result.loc[0, "structural_component"] == 35.0
    assert result.loc[0, "mechanism_component"] == 35.0
    assert result.loc[0, "disease_component"] == 10.0
    assert result.loc[0, "biophysical_component"] == 20.0
    assert result.loc[1, "variant_priority_score"] == 31.0
    assert bool(result.loc[1, "structural_component_missing"])
