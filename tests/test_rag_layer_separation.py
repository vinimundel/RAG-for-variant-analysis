"""A mechanistic query must never be able to reach a review.

The guarantee is structural: layers are separate Qdrant collections and separate
BM25 corpora, so reaching a review requires loading a different index, not
forgetting a metadata filter.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from pipeline.steps.build_rag_index import group_by_layer, indexable_layers
from pipeline.steps.run_literature_reranking import resolve_retrieval_layer
from src.literature.corpus_layers import INDEXED_LAYERS, UNLAYERED, load_layer_map
from src.rag.bm25_index import PersistentBM25, bm25_corpus_path
from src.rag.chain import CONTEXT_LAYER, EVIDENCE_LAYER, LiteratureMechanismAnalyzer
from src.rag.schemas import LiteratureEvidence
from src.rag.vector_store import collection_name


def chunk(pmid: str, layer: str, text: str = "BRAF V600E dimerization") -> Document:
    return Document(page_content=text, metadata={"pmid": pmid, "gene": "BRAF",
                                                 "corpus_layer": layer,
                                                 "source_sha256": pmid, "section": "Results"})


def test_each_layer_gets_its_own_collection_name():
    names = {collection_name("BRAF", layer) for layer in INDEXED_LAYERS}
    assert len(names) == len(INDEXED_LAYERS)
    assert collection_name("BRAF", EVIDENCE_LAYER) != collection_name("BRAF", CONTEXT_LAYER)


def test_pre_layer_collection_name_is_preserved():
    assert collection_name("BRAF") == "literature_braf_discovery_v3"
    assert collection_name("BRAF", EVIDENCE_LAYER).startswith("literature_braf_discovery_v3_")


def test_each_layer_gets_its_own_bm25_corpus(tmp_path):
    paths = {bm25_corpus_path(tmp_path, layer) for layer in INDEXED_LAYERS}
    assert len(paths) == len(INDEXED_LAYERS)
    assert bm25_corpus_path(tmp_path) not in paths


def test_excluded_chunks_are_never_indexed():
    grouped = group_by_layer([chunk("1", "primary_evidence"), chunk("2", "excluded"),
                              chunk("3", "context_reference")])
    assert "excluded" not in indexable_layers(grouped)
    assert set(indexable_layers(grouped)) == {"primary_evidence", "context_reference"}


def test_an_uncurated_corpus_builds_the_pre_layer_index_instead_of_being_relabelled():
    grouped = group_by_layer([chunk("1", UNLAYERED), chunk("2", UNLAYERED)])
    assert indexable_layers(grouped) == [None]


def test_a_curated_corpus_never_falls_back_to_the_pre_layer_index():
    grouped = group_by_layer([chunk("1", "primary_evidence"), chunk("2", UNLAYERED)])
    assert indexable_layers(grouped) == ["primary_evidence"]


def test_an_entirely_excluded_corpus_fails_the_build():
    with pytest.raises(RuntimeError, match="excluded layer"):
        indexable_layers(group_by_layer([chunk("1", "excluded")]))


def test_bm25_corpora_are_disjoint_across_layers(tmp_path):
    # BM25Okapi IDF is log((N - n + 0.5) / (n + 0.5)), which is zero for a term
    # in one of two documents, so each corpus needs at least three.
    def corpus(layer: str, hit: str, *misses: str) -> list:
        return [chunk(hit, layer, "V600E increases dimerization"),
                *(chunk(pmid, layer, "unrelated cytoskeleton assay") for pmid in misses)]

    PersistentBM25(corpus("primary_evidence", "1", "91", "92")).save(
        bm25_corpus_path(tmp_path, EVIDENCE_LAYER))
    PersistentBM25(corpus("context_reference", "2", "81", "82")).save(
        bm25_corpus_path(tmp_path, CONTEXT_LAYER))
    evidence = PersistentBM25.load(bm25_corpus_path(tmp_path, EVIDENCE_LAYER))
    hits = evidence.search("V600E dimerization", k=10)
    assert {doc.metadata["pmid"] for doc, _ in hits} == {"1"}
    context = PersistentBM25.load(bm25_corpus_path(tmp_path, CONTEXT_LAYER))
    assert {doc.metadata["pmid"] for doc, _ in context.search("V600E dimerization", k=10)} == {"2"}


class _Analyzer:
    """Exercises the layer guard without loading MedCPT or Qdrant."""

    layer = EVIDENCE_LAYER
    _assert_evidence_layer = __import__(
        "src.rag.chain", fromlist=["LiteratureMechanismAnalyzer"]
    ).LiteratureMechanismAnalyzer._assert_evidence_layer


def evidence(layer: str) -> LiteratureEvidence:
    return LiteratureEvidence(evidence_id="E01", pmid="1", passage="text",
                              source_file="f.pdf", corpus_layer=layer)


def test_primary_passages_pass_the_guard():
    result = _Analyzer()._assert_evidence_layer([evidence(EVIDENCE_LAYER)])
    assert len(result) == 1


def test_braf_mechanistic_analyzer_cannot_be_bound_to_context_or_legacy_layers():
    with pytest.raises(ValueError, match="primary_evidence"):
        LiteratureMechanismAnalyzer("BRAF", layer=CONTEXT_LAYER)
    with pytest.raises(ValueError, match="primary_evidence"):
        LiteratureMechanismAnalyzer("BRAF", layer=None)


@pytest.mark.parametrize("layer", [CONTEXT_LAYER, "excluded", UNLAYERED])
def test_a_non_primary_passage_fails_loudly_instead_of_being_dropped(layer):
    with pytest.raises(RuntimeError, match="rebuild the layered index"):
        _Analyzer()._assert_evidence_layer([evidence(EVIDENCE_LAYER), evidence(layer)])


def test_evidence_defaults_to_unlayered_never_to_primary():
    assert LiteratureEvidence(evidence_id="E01", pmid="1", passage="t",
                              source_file="f.pdf").corpus_layer == "unlayered"


def test_retrieval_layer_is_resolved_from_the_built_index(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"indexed_layers": {
        "primary_evidence": {}, "context_reference": {}}}), encoding="utf-8")
    assert resolve_retrieval_layer(manifest) == EVIDENCE_LAYER


def test_a_pre_layer_index_is_resolved_explicitly(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"indexed_layers": {"None": {}}}), encoding="utf-8")
    assert resolve_retrieval_layer(manifest) is None


def test_an_index_without_a_primary_layer_refuses_to_run(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"indexed_layers": {"context_reference": {}}}),
                        encoding="utf-8")
    with pytest.raises(RuntimeError, match="has no 'primary_evidence' collection"):
        resolve_retrieval_layer(manifest)


def test_layer_map_of_an_uncurated_gene_is_empty_not_primary(tmp_path):
    assert load_layer_map(tmp_path / "absent.csv") == {}
