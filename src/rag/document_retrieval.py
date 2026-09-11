"""Document-level fusion for literature retrieval.

Fusing at chunk level lets a long paper occupy many slots at once: every one of
its chunks competes independently, so the ranking rewards document length rather
than relevance.  Here each method reduces its own ranking to one entry per PMID
first, RRF runs over documents, and passages are only chosen after the documents
have been reranked.

Every function is a pure transformation of ranked results, so the fusion policy
is testable without loading an encoder or a vector store.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Sequence

RRF_K = 60
# At most two passages from one paper: enough for a claim and its context,
# not enough for one document to fill the evidence set.
MAX_PASSAGES_PER_DOCUMENT = 2


def document_ranking(results: Sequence[tuple], key_of: Callable, pmid_of: Callable) -> list[tuple]:
    """Reduce one method's ranked chunks to its best chunk per document.

    Results must already be in descending score order, so the first chunk seen
    for a PMID is that method's best evidence from that document.
    """
    best, ordered = {}, []
    for document, score in results:
        pmid = str(pmid_of(document))
        if pmid in best:
            continue
        best[pmid] = (key_of(document), float(score))
        ordered.append((pmid, key_of(document), float(score)))
    return ordered


def rrf_over_documents(rankings: dict[str, Sequence[tuple]], rrf_k: int = RRF_K
                       ) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    """Fuse per-method document rankings, keeping who contributed and at what rank.

    Returns the fused score per PMID and, for auditing, the rank each labelled
    ranking gave it.  A document that only ever appears through one channel or
    one method is visible as such.
    """
    fused: dict[str, float] = {}
    contributions: dict[str, dict[str, int]] = {}
    for label, ranking in rankings.items():
        for rank, (pmid, _key, _score) in enumerate(ranking, 1):
            fused[pmid] = fused.get(pmid, 0.0) + 1.0 / (rrf_k + rank)
            contributions.setdefault(pmid, {})[label] = rank
    return fused, contributions


def select_passages(document_order: Sequence[str], chunks_by_document: dict[str, Sequence],
                    k: int, max_per_document: int = MAX_PASSAGES_PER_DOCUMENT) -> list:
    """Walk reranked documents in order, taking at most ``max_per_document`` each.

    A second pass is deliberately absent: if the best documents cannot supply
    ``k`` passages within the cap, the result is short rather than padded with a
    third passage from the top document.
    """
    selected = []
    for pmid in document_order:
        for chunk in list(chunks_by_document.get(pmid, ()))[:max_per_document]:
            selected.append(chunk)
            if len(selected) >= k:
                return selected
    return selected


def retrieval_diagnostics(selected_pmids: Sequence[str], candidate_pmids: Iterable[str],
                          contributions: dict[str, dict[str, int]],
                          corpus_chunk_counts: dict[str, int]) -> dict:
    """Report the bias surface of one retrieval: concentration, size and channel.

    ``document_size_bias`` compares the corpus chunk count of the documents that
    were selected against the corpus as a whole.  A value well above 1 means the
    ranking is favouring long documents, which is the failure this module exists
    to prevent.
    """
    selected = [str(pmid) for pmid in selected_pmids]
    candidates = sorted({str(pmid) for pmid in candidate_pmids})
    sizes = [corpus_chunk_counts.get(pmid, 0) for pmid in dict.fromkeys(selected)]
    corpus_mean = (sum(corpus_chunk_counts.values()) / len(corpus_chunk_counts)
                   if corpus_chunk_counts else 0.0)
    selected_mean = sum(sizes) / len(sizes) if sizes else 0.0
    method_of = lambda label: label.rsplit(":", 1)[-1]
    channel_of = lambda label: label.split(":", 1)[0]
    per_document_methods = {pmid: {method_of(label) for label in labels}
                            for pmid, labels in contributions.items()}
    return {
        "selected_documents": len(dict.fromkeys(selected)),
        "selected_passages": len(selected),
        "candidate_documents": len(candidates),
        "passages_per_document": dict(Counter(selected)),
        "max_passages_from_one_document": max(Counter(selected).values(), default=0),
        "corpus_mean_chunks_per_document": round(corpus_mean, 3),
        "selected_mean_chunks_per_document": round(selected_mean, 3),
        "document_size_bias": (round(selected_mean / corpus_mean, 3) if corpus_mean else None),
        "documents_found_by_both_methods": sum(
            1 for pmid in selected if len(per_document_methods.get(pmid, ())) > 1),
        "documents_found_by_lexical_only": sum(
            1 for pmid in selected if per_document_methods.get(pmid) == {"bm25"}),
        "documents_found_by_dense_only": sum(
            1 for pmid in selected if per_document_methods.get(pmid) == {"dense"}),
        "channels_per_selected_document": {
            pmid: sorted({channel_of(label) for label in contributions.get(pmid, {})})
            for pmid in dict.fromkeys(selected)
        },
    }
