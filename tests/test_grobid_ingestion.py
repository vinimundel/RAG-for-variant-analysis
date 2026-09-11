"""PDF extraction must use GROBID and expose service/parser failures."""

import json
from types import SimpleNamespace

import pytest
from lxml import etree

from src.rag import ingestor


@pytest.fixture
def corpus(tmp_path):
    pdf_dir = tmp_path / 'data/input/BRAF/evidence/pdfs'
    pdf_dir.mkdir(parents=True)
    pdf = pdf_dir / 'PMID_123.pdf'
    pdf.write_bytes(b'%PDF-1.4\n' + b' synthetic fixture ' * 100)
    tei = tmp_path / 'data/output/BRAF/rag/tei/PMID_123.tei.xml'
    tei.parent.mkdir(parents=True)
    text = 'Synthetic BRAF V600E mechanistic evidence in the kinase domain. ' * 15
    tei.write_text(f'''<TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader><fileDesc><titleStmt><title>Synthetic study</title></titleStmt>
      </fileDesc></teiHeader><text><body><div><head>Results</head><p>{text}</p>
      </div></body></text></TEI>''')
    return tmp_path, pdf, tei


def test_successful_extraction_preserves_grobid_provenance(corpus, monkeypatch):
    root, pdf, tei = corpus
    content = tei.read_bytes()
    tei.unlink()
    monkeypatch.setattr(ingestor.requests, 'get', lambda *a, **k:
                        SimpleNamespace(status_code=200, text='true'))
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        assert kwargs['files']['input'][1].read().startswith(b'%PDF')
        return SimpleNamespace(status_code=200, content=content)

    monkeypatch.setattr(ingestor.requests, 'post', post)
    docs = ingestor.load_pdfs_for_gene(root, 'BRAF', workers=1)
    assert calls == ['http://localhost:8070/api/processFulltextDocument']
    assert docs and tei.exists()
    assert {d.metadata['parser'] for d in docs} == {'grobid_tei'}
    assert all(d.metadata['pmid'] == '123' for d in docs)
    assert docs[0].metadata['section'] == 'Results'
    assert docs[0].metadata['source_sha256']


def test_unhealthy_grobid_stops_ingestion(corpus, monkeypatch):
    monkeypatch.setattr(ingestor.requests, 'get', lambda *a, **k:
                        SimpleNamespace(status_code=503, text='unavailable'))
    with pytest.raises(RuntimeError, match='GROBID is unavailable'):
        ingestor.load_pdfs_for_gene(corpus[0], 'BRAF')


def test_failed_pdf_extraction_propagates(corpus, monkeypatch):
    root, _, tei = corpus
    tei.unlink()
    monkeypatch.setattr(ingestor.requests, 'get', lambda *a, **k:
                        SimpleNamespace(status_code=200, text='true'))
    monkeypatch.setattr(ingestor.requests, 'post', lambda *a, **k:
                        SimpleNamespace(status_code=500, content=b'failed'))
    with pytest.raises(RuntimeError, match='GROBID failed.*HTTP 500'):
        ingestor.load_pdfs_for_gene(root, 'BRAF', workers=1)
    assert not tei.exists()


def test_invalid_tei_propagates_parser_error(corpus, monkeypatch):
    corpus[2].write_text('<invalid')
    monkeypatch.setattr(ingestor, 'grobid_healthcheck', lambda *a: None)
    with pytest.raises(etree.XMLSyntaxError):
        ingestor.load_pdfs_for_gene(corpus[0], 'BRAF', workers=1)


def test_cached_tei_needs_no_grobid_and_abstracts_remain_separate(corpus, monkeypatch):
    root, _, _ = corpus
    abstracts = root / 'data/input/BRAF/evidence/abstracts/pubmed_abstracts.jsonl'
    abstracts.parent.mkdir(parents=True)
    abstracts.write_text('\n'.join(json.dumps({
        'pmid': pmid, 'title': 'Synthetic abstract', 'abstract': 'BRAF study abstract.'
    }) for pmid in ['123', '456']))

    def forbidden(*args, **kwargs):
        pytest.fail('Cached TEI loading must not contact GROBID')

    monkeypatch.setattr(ingestor.requests, 'get', forbidden)
    monkeypatch.setattr(ingestor.requests, 'post', forbidden)
    docs = ingestor.load_cached_tei_for_gene(root, 'BRAF')
    assert {d.metadata['parser'] for d in docs} == {'grobid_tei', 'pubmed_xml'}
    abstract_docs = [d for d in docs if d.metadata['parser'] == 'pubmed_xml']
    assert {d.metadata['pmid'] for d in abstract_docs} == {'456'}
    assert all(not d.metadata['full_text_available'] for d in abstract_docs)
