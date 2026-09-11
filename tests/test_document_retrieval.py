"""Fusion must rank documents, not reward documents for being long."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.rag.chain import (
    DOCUMENT_CANDIDATES, MECHANISTIC_CHANNELS, PHARMACOLOGY_TERMS, QUERY_CHANNELS,
    DiscoveryLiteratureAnalyzer,
)
from src.rag.document_retrieval import (
    MAX_PASSAGES_PER_DOCUMENT, document_ranking, retrieval_diagnostics,
    rrf_over_documents, select_passages,
)
from src.rag.schemas import DiscoveryQuery


def chunk(pmid: str, index: int) -> Document:
    return Document(page_content=f"{pmid} chunk {index}",
                    metadata={"pmid": pmid, "key": f"{pmid}-{index}"})


def key_of(doc: Document) -> str:
    return doc.metadata["key"]


def pmid_of(doc: Document) -> str:
    return doc.metadata["pmid"]


# A long paper contributing five of the top seven chunks; two shorter papers.
LONG_PAPER_RESULTS = [
    (chunk("111", 1), 0.99), (chunk("111", 2), 0.98), (chunk("111", 3), 0.97),
    (chunk("222", 1), 0.96), (chunk("111", 4), 0.95), (chunk("111", 5), 0.94),
    (chunk("333", 1), 0.93),
]


def test_a_method_contributes_one_entry_per_document():
    ranking = document_ranking(LONG_PAPER_RESULTS, key_of, pmid_of)
    assert [pmid for pmid, _, _ in ranking] == ["111", "222", "333"]


def test_the_kept_chunk_is_that_methods_best_for_the_document():
    ranking = document_ranking(LONG_PAPER_RESULTS, key_of, pmid_of)
    assert ranking[0][1] == "111-1"
    assert ranking[0][2] == pytest.approx(0.99)


def test_document_ranking_of_an_empty_result_is_empty():
    assert document_ranking([], key_of, pmid_of) == []


def test_length_no_longer_buys_extra_rrf_mass():
    """Under chunk-level fusion the long paper accrued five contributions."""
    ranking = document_ranking(LONG_PAPER_RESULTS, key_of, pmid_of)
    fused, contributions = rrf_over_documents({"structural_mechanism:q:dense": ranking})
    assert len(contributions["111"]) == 1
    assert fused["111"] > fused["222"] > fused["333"]


def test_rrf_accumulates_across_channels_and_methods():
    ranking = document_ranking(LONG_PAPER_RESULTS, key_of, pmid_of)
    single, _ = rrf_over_documents({"a:q:dense": ranking})
    doubled, contributions = rrf_over_documents({"a:q:dense": ranking, "b:q:bm25": ranking})
    assert doubled["111"] == pytest.approx(2 * single["111"])
    assert set(contributions["111"]) == {"a:q:dense", "b:q:bm25"}


def test_rrf_records_the_rank_each_ranking_gave_a_document():
    ranking = document_ranking(LONG_PAPER_RESULTS, key_of, pmid_of)
    _, contributions = rrf_over_documents({"a:q:dense": ranking})
    assert contributions["333"]["a:q:dense"] == 3


def test_no_more_than_two_passages_come_from_one_document():
    chunks = {"111": [chunk("111", i) for i in range(1, 6)],
              "222": [chunk("222", 1)], "333": [chunk("333", 1)]}
    selected = select_passages(["111", "222", "333"], chunks, k=5)
    counts = {}
    for doc in selected:
        counts[pmid_of(doc)] = counts.get(pmid_of(doc), 0) + 1
    assert counts["111"] == MAX_PASSAGES_PER_DOCUMENT
    assert set(counts) == {"111", "222", "333"}


def test_selection_follows_the_document_order():
    chunks = {"111": [chunk("111", 1)], "222": [chunk("222", 1)]}
    assert [pmid_of(d) for d in select_passages(["222", "111"], chunks, k=2)] == ["222", "111"]


def test_a_short_result_is_never_padded_past_the_cap():
    """Two documents and a cap of two cannot honestly produce five passages."""
    chunks = {"111": [chunk("111", i) for i in range(1, 9)],
              "222": [chunk("222", i) for i in range(1, 9)]}
    assert len(select_passages(["111", "222"], chunks, k=5)) == 4


def test_diagnostics_expose_document_size_bias():
    diagnostics = retrieval_diagnostics(
        selected_pmids=["111", "111", "222"],
        candidate_pmids=["111", "222", "333"],
        contributions={"111": {"structural_mechanism:q:dense": 1},
                       "222": {"structural_mechanism:q:bm25": 2}},
        corpus_chunk_counts={"111": 100, "222": 10, "333": 10},
    )
    assert diagnostics["selected_documents"] == 2
    assert diagnostics["selected_passages"] == 3
    assert diagnostics["max_passages_from_one_document"] == 2
    assert diagnostics["passages_per_document"] == {"111": 2, "222": 1}
    assert diagnostics["document_size_bias"] > 1


def test_diagnostics_separate_lexical_and_dense_contribution():
    diagnostics = retrieval_diagnostics(
        selected_pmids=["111", "222", "333"], candidate_pmids=["111", "222", "333"],
        contributions={"111": {"a:q:dense": 1, "a:q:bm25": 2},
                       "222": {"a:q:bm25": 1}, "333": {"a:q:dense": 1}},
        corpus_chunk_counts={"111": 5, "222": 5, "333": 5},
    )
    assert diagnostics["documents_found_by_both_methods"] == 1
    assert diagnostics["documents_found_by_lexical_only"] == 1
    assert diagnostics["documents_found_by_dense_only"] == 1
    assert diagnostics["document_size_bias"] == 1.0


def test_diagnostics_record_which_channels_surfaced_each_document():
    diagnostics = retrieval_diagnostics(
        selected_pmids=["111"], candidate_pmids=["111"],
        contributions={"111": {"structural_mechanism:q:dense": 1, "disease:q:bm25": 3}},
        corpus_chunk_counts={"111": 5},
    )
    assert diagnostics["channels_per_selected_document"]["111"] == ["disease", "structural_mechanism"]


def request(mechanism: str = "", observation: str = "buried residue") -> DiscoveryQuery:
    return DiscoveryQuery(gene="BRAF", variant="V600E", position=600,
                          functional_region="activation_segment",
                          structural_observation=observation,
                          biophysical_hypotheses_json="[]",
                          condition_profile="braf_oncology")


def analyzer() -> DiscoveryLiteratureAnalyzer:
    from pathlib import Path
    return DiscoveryLiteratureAnalyzer("BRAF", rag_dir=Path("unused"))


def test_a_mechanistic_request_excludes_the_pharmacology_channel():
    channels = analyzer()._channel_queries(request())
    assert "pharmacology" not in channels
    assert set(channels) <= set(MECHANISTIC_CHANNELS)


def test_a_mechanistic_request_carries_no_therapeutic_language():
    joined = " ".join(analyzer()._queries(request())).lower()
    assert not any(term in joined for term in ("vemurafenib", "dabrafenib", "drug", "inhibitor"))


def test_a_structural_binding_site_annotation_does_not_open_pharmacology():
    """binding_site_context is a nucleotide site, not a drug; it opened the
    channel for 21 of 171 BRAF variants before the gate was made explicit."""
    observation = ("delta_charge=0.0; secondary_structure=sheet; "
                   "flags=buried_residue;binding_site_context;active_site_context")
    assert "pharmacology" not in analyzer()._channel_queries(request(observation=observation))


def test_pharmacology_opens_when_requested_explicitly():
    asked = request()
    asked = asked.model_copy(update={"pharmacology_requested": True})
    assert "pharmacology" in analyzer()._channel_queries(asked)


@pytest.mark.parametrize("term", PHARMACOLOGY_TERMS)
def test_a_mechanism_stated_in_drug_terms_opens_pharmacology(term):
    asked = request().model_copy(update={"mechanism": f"variant drives {term}"})
    assert "pharmacology" in analyzer()._channel_queries(asked)


@pytest.mark.parametrize("mechanism", ["binding site occupancy", "ATP binding pocket",
                                       "dimerization interface"])
def test_a_structural_mechanism_never_opens_pharmacology(mechanism):
    asked = request().model_copy(update={"mechanism": mechanism})
    assert "pharmacology" not in analyzer()._channel_queries(asked)


def test_channels_can_be_requested_explicitly():
    channels = analyzer()._channel_queries(request(), channels=("structural_mechanism",))
    assert list(channels) == ["structural_mechanism"]


def test_declared_channels_match_the_ones_that_can_be_built():
    built = analyzer()._channel_queries(request(), channels=QUERY_CHANNELS)
    assert set(built) == set(QUERY_CHANNELS)


def test_flattened_queries_preserve_channel_order():
    handle = analyzer()
    by_channel = handle._channel_queries(request())
    assert handle._queries(request()) == [q for items in by_channel.values() for q in items]


def test_documents_are_reranked_before_passages_are_chosen():
    """The cap only means something if it applies after document ranking."""
    assert DOCUMENT_CANDIDATES >= 10
    assert MAX_PASSAGES_PER_DOCUMENT == 2
