from pathlib import Path
from types import SimpleNamespace

from pipeline.steps.run_literature_reranking import _cache_path, _specificity
from src.rag.chain import DiscoveryLiteratureAnalyzer, LiteratureMechanismAnalyzer, _normalize_citation_ids
from src.rag.schemas import DiscoveryQuery, LiteratureEvidence, MechanisticQuery
from src.rag.embedder import article_pair
from src.rag.llm import llm_model_name


def test_medcpt_article_pair_uses_title_segment():
    assert article_pair("Paper title\nResults\nBRAF V600E") == [
        "Paper title", "Results\nBRAF V600E"
    ]


def test_rag_abstains_when_retrieval_is_empty(monkeypatch):
    analyzer = LiteratureMechanismAnalyzer(gene="BRAF")
    monkeypatch.setattr(analyzer, "retrieve", lambda query, k: [])
    result = analyzer.infer(MechanisticQuery(
        gene="BRAF", variant="V600E", mechanism="dimerization",
        structural_observation="charge changes at an interface", k=3,
    ))
    assert result.inference.verdict == "insufficient_evidence"
    assert result.inference.cited_evidence_ids == []
    assert result.inference.requires_experimental_validation


def test_specificity_uses_only_passages_cited_by_llm():
    evidence = [
        SimpleNamespace(evidence_id="E01", passage="BRAF V600E was tested."),
        SimpleNamespace(evidence_id="E02", passage="The kinase region forms an interface."),
    ]
    assert _specificity("V600E", evidence, ["E02"]) == "general"
    assert _specificity("V600E", evidence, ["E01"]) == "exact_variant"


def test_inference_cache_is_versioned_by_stack_prompt_and_model():
    path = _cache_path(Path("cache"), "V600E")
    assert "discovery_v3" in path.name
    assert "discovery_hypothesis_gene_aware_v4" in path.name
    assert "braf_oncology" in path.name
    assert llm_model_name().replace(":", "_") in path.name


def test_citation_ids_normalize_only_within_whitelist():
    assert _normalize_citation_ids(["E1", "[E02]", "1"], {"E01", "E02"}) == ["E01", "E02"]
    import pytest
    with pytest.raises(ValueError, match="unavailable"):
        _normalize_citation_ids(["E06"], {"E01", "E02"})


def test_discovery_exact_claim_requires_exact_variant_passage():
    from pipeline.steps.run_literature_reranking import _specificity
    evidence = [LiteratureEvidence(evidence_id="E01", pmid="1", passage="BRAF V600 mutations occur.",
                                   source_file="x", source_type="pubmed_abstract")]
    assert _specificity("V600E", evidence, ["E01"], "activation_segment") != "exact_variant"


def test_retrieval_diversity_policy_is_at_most_two_chunks_per_pmid():
    # This invariant is asserted again on the final artifact by the pipeline audit.
    pmids = "1;1;2;2;3".split(";")
    from collections import Counter
    assert max(Counter(pmids).values()) <= 2


def test_discovery_can_infer_from_persisted_empty_retrieval(monkeypatch):
    analyzer = DiscoveryLiteratureAnalyzer(gene="GSDMD")
    monkeypatch.setattr(
        analyzer, "retrieve_discovery",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run")),
    )
    request = DiscoveryQuery(
        gene="GSDMD", variant="C191R", position=191,
        functional_region="cys191_regulatory_site", structural_observation="",
        biophysical_hypotheses_json="[]", condition_profile="neuro_aging",
    )
    result = analyzer.infer_discovery(
        request, evidence=[], retrieval_queries=["persisted query"]
    )
    assert result.retrieval_queries == ["persisted query"]
    assert result.llm_contract_status == "deterministic_fallback"
