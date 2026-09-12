from langchain_core.documents import Document

from healthrag.rag.bm25_index import PersistentBM25, tokenize


def test_bm25_preserves_variant_tokens_and_roundtrips(tmp_path):
    assert "v600e" in tokenize("BRAF V600E-driven dimerization")
    documents = [Document(page_content="BRAF V600E dimer interface", metadata={"pmid": "1"}),
                 Document(page_content="unrelated control", metadata={"pmid": "2"}),
                 Document(page_content="second unrelated experiment", metadata={"pmid": "3"})]
    index = PersistentBM25(documents)
    path = tmp_path / "bm25.jsonl"
    index.save(path)
    result = PersistentBM25.load(path).search("V600E interface", 1)
    assert result[0][0].metadata["pmid"] == "1"
