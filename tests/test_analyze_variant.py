import json
from pathlib import Path

import pytest

from pipeline.steps import analyze_variant
from src.rag.schemas import DiscoveryQuery


@pytest.fixture
def query():
    path = Path(__file__).resolve().parents[1] / 'examples/braf_v600e_query.json'
    return DiscoveryQuery.model_validate_json(path.read_text())


@pytest.fixture
def index(tmp_path):
    (tmp_path / 'braf_rag_index_manifest.json').write_text(json.dumps({
        'indexed_layers': {'primary_evidence': {'documents': 1}}
    }))
    return tmp_path


def test_literature_only_analysis_uses_query_context_and_releases_retrieval(query, index, monkeypatch):
    calls = []
    original = analyze_variant.DiscoveryLiteratureAnalyzer

    class Analyzer(original):
        def retrieve_discovery(self, queries, k, **kwargs):
            calls.append(('retrieve', queries, k, kwargs))
            return []

        def release_retrieval_models(self):
            calls.append(('release',))

    monkeypatch.setattr(analyze_variant, 'DiscoveryLiteratureAnalyzer', Analyzer)
    result = analyze_variant.analyze(query, index)
    assert calls[0][0] == 'retrieve'
    assert isinstance(calls[0][1], dict)
    assert calls[0][3]['variant'] == 'V600E'
    assert calls[0][3]['functional_region'] == 'kinase domain'
    assert calls[1] == ('release',)
    assert result.inference.variant == 'V600E'
    assert result.inference_status == 'no_evidence'
    assert result.retrieval_index_fingerprint
    assert result.evidence == []


def test_missing_index_is_an_error(query, tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_variant.analyze(query, tmp_path)


def test_retrieval_failure_propagates_and_releases_models(query, index, monkeypatch):
    released = []
    original = analyze_variant.DiscoveryLiteratureAnalyzer

    class Analyzer(original):
        def retrieve_discovery(self, *args, **kwargs):
            raise RuntimeError('retrieval unavailable')

        def release_retrieval_models(self):
            released.append(True)

    monkeypatch.setattr(analyze_variant, 'DiscoveryLiteratureAnalyzer', Analyzer)
    with pytest.raises(RuntimeError, match='retrieval unavailable'):
        analyze_variant.analyze(query, index)
    assert released == [True]
