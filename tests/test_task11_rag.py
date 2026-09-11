"""Contracts introduced by the curated Task 11 retrieval rebuild."""

from __future__ import annotations

import pandas as pd

import pandas as pd

from src.literature.benchmark import benchmark_gate, citation_accuracy
from src.literature.evidence_grade import ClaimEvidence, grade_claim
from src.rag.schemas import DiscoveryHypothesis, DiscoveryInference, DiscoveryRAGResult
from pipeline.steps.run_literature_reranking import _normalize_generation_modes


def test_task11_gate_blocks_an_unannotated_blind_sheet():
    columns = ["item_id", "mutation", "pmid", "corpus_layer", "relevance",
               "specificity", "entailment", "cited_in_claim"]
    sheet = pd.DataFrame([{
        "item_id": "I001", "mutation": "V600E", "pmid": "1",
        "corpus_layer": "primary_evidence", "relevance": "",
        "specificity": "", "entailment": "", "cited_in_claim": "",
    }], columns=columns)
    repeats = sheet.rename(columns={"item_id": "repeat_item_id"})
    repeats["item_id"] = "R001"
    result = benchmark_gate(sheet, repeats)
    assert result["status"] == "blocked"
    assert result["promotion_allowed"] is False
    assert "human annotations are incomplete" in result["reasons"]


def test_claim_grade_records_full_text_dimensions_and_stays_deterministic():
    claim = ClaimEvidence(
        claim_id="V600E:H01", variant="V600E", claim_text="activation",
        variant_directness="exact_variant", support_type="biochemical",
        controls_reported="yes", independent_papers=1, full_text_verified=True,
        primary_evidence_pmids=["1"], evidence_ids=["E01"],
        verifiable_full_text_dimensions=["direction", "controls"],
    )
    result = grade_claim(claim)
    assert result.grade == "B"
    assert result.label == "literature_supported"


def test_task11_citation_accuracy_returns_json_null_value_when_unreferenced():
    result = citation_accuracy(pd.DataFrame({
        "cited_in_claim": ["no"], "entailment": ["neutral"],
        "specificity": ["general"],
    }))
    assert result["citation_accuracy"] is None
    assert result["exact_claim_accuracy"] is None


def test_old_fallback_cache_is_upgraded_with_generation_modes():
    inference = DiscoveryInference(
        variant="V600E", gene="BRAF", uncertainty="test",
        disease_association={"status": "associated", "cited_evidence_ids": ["E01"]},
        hypotheses=[
            DiscoveryHypothesis(
                mechanism="literature", target="BRAF", predicted_direction="unclear",
                classification="literature_reported", rationale="test",
            ),
            DiscoveryHypothesis(
                mechanism="rule", target="BRAF", predicted_direction="unclear",
                classification="biophysical_hypothesis", rationale="test", rule_ids=["r1"],
            ),
        ],
    )
    result = DiscoveryRAGResult(
        inference=inference, evidence=[], retrieval_queries=[],
        llm_contract_status="deterministic_fallback",
    )
    upgraded = _normalize_generation_modes(result)
    assert [item.generation_mode for item in upgraded.inference.hypotheses] == [
        "deterministic", "biophysical_rule"
    ]
    assert upgraded.inference.disease_association.generation_mode == "deterministic"
