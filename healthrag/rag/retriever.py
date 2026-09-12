"""BM25 + MedCPT fusion with explicit layer isolation and document diversity."""
import gc
import os
from pathlib import Path

from healthrag.rag.bm25_index import PersistentBM25, bm25_corpus_path
from healthrag.rag.reranker import MedCPTReranker
from healthrag.rag.schemas import LiteratureEvidence
from healthrag.rag.vector_store import load_vector_store, retrieve_with_scores
from healthrag.runtime import require_memory


def chunk_key(document) -> str:
    metadata = document.metadata
    return '|'.join(str(metadata.get(name, '')) for name in
                    ('pmid', 'source_sha256', 'section', 'section_chunk_index'))


class HybridRetriever:
    def __init__(self, collection: str, directory: Path, candidates: int = 20):
        self.collection, self.directory, self.candidates = collection, directory, candidates

    def retrieve(self, query: str, k: int = 4) -> list[LiteratureEvidence]:
        require_memory("Hybrid retrieval", 2.5)
        store = load_vector_store(self.collection, local_path=self.directory / 'qdrant',
                                  layer='primary_evidence')
        reranker = MedCPTReranker()
        try:
            bm25 = PersistentBM25.load(bm25_corpus_path(self.directory, 'primary_evidence'))
            dense = retrieve_with_scores(store, query, self.candidates)
            lexical = bm25.search(query, self.candidates)
            documents, fused, scores = {}, {}, {}
            for method, results in [('dense', dense), ('bm25', lexical)]:
                for rank, (document, score) in enumerate(results, 1):
                    if document.metadata.get('corpus_layer') != 'primary_evidence':
                        raise ValueError('A non-primary document reached the primary index')
                    key = chunk_key(document)
                    documents[key] = document
                    fused[key] = fused.get(key, 0.0) + 1 / (60 + rank)
                    scores.setdefault(key, {})[method] = float(score)
            ordered = sorted(fused, key=fused.get, reverse=True)[:self.candidates]
            ranked = reranker.rerank(query, [documents[key] for key in ordered], top_k=len(ordered))
            evidence, per_paper = [], {}
            for document, rank_score in ranked:
                metadata, key = document.metadata, chunk_key(document)
                pmid = str(metadata['pmid'])
                if per_paper.get(pmid, 0) >= 2:
                    continue
                per_paper[pmid] = per_paper.get(pmid, 0) + 1
                evidence.append(LiteratureEvidence(
                    evidence_id=f'E{len(evidence)+1:02d}', pmid=pmid,
                    article_title=metadata.get('title'), doi=metadata.get('doi'),
                    publication=metadata.get('publication'), passage=document.page_content[:2500],
                    retrieval_score=scores[key].get('dense'), bm25_score=scores[key].get('bm25'),
                    rrf_score=fused[key], reranker_score=float(rank_score), section=metadata.get('section'),
                    source_file=metadata.get('source_file', ''), parser=metadata.get('parser'),
                    source_type=metadata.get('source_type', 'grobid_full_text'),
                    full_text_available=metadata.get('full_text_available', False),
                    corpus_layer='primary_evidence', evidence_type='primary_evidence'))
                if len(evidence) >= k:
                    break
            return evidence
        finally:
            store.client.close()
            # Release model references before loading Ollama on small workstations.
            del store, reranker
            gc.collect()
            if os.getenv("MEDCPT_DEVICE", "cpu").startswith("cuda"):
                import torch
                torch.cuda.empty_cache()
