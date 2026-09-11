"""Qdrant persistence and score-preserving biomedical retrieval."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.rag.embedder import get_biomedical_embedder

DEFAULT_HOST = os.getenv("QDRANT_HOST", "localhost")
DEFAULT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def collection_name(gene: str, layer: str | None = None) -> str:
    """Name one collection per gene and corpus layer.

    A mechanistic query must never be able to reach a review, so layers live in
    physically separate collections rather than behind a metadata filter that a
    caller could forget to pass.  ``layer=None`` keeps the pre-layer name and
    exists only for corpora built before curation.
    """
    version = "gsdmd_modular_v1" if gene.upper() == "GSDMD" else "discovery_v3"
    base = f"literature_{gene.lower()}_{version}"
    return base if layer is None else f"{base}_{layer}"


def build_vector_store(documents: list[Document], gene_symbol: str, host: str = DEFAULT_HOST,
                       port: int = DEFAULT_PORT, overwrite: bool = False,
                       local_path: Path | None = None,
                       layer: str | None = None) -> QdrantVectorStore:
    if not documents:
        raise ValueError("Cannot build a vector store with zero documents")
    connection = {"path": str(local_path)} if local_path else {"url": f"http://{host}:{port}"}
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=get_biomedical_embedder(),
        collection_name=collection_name(gene_symbol, layer),
        **connection,
        force_recreate=overwrite,
    )


def load_vector_store(gene_symbol: str, host: str = DEFAULT_HOST,
                      port: int = DEFAULT_PORT, local_path: Path | None = None,
                      layer: str | None = None) -> QdrantVectorStore:
    client = QdrantClient(path=str(local_path)) if local_path else QdrantClient(host=host, port=port)
    name = collection_name(gene_symbol, layer)
    if not client.collection_exists(name):
        raise ValueError(f"Collection {name!r} is absent; build the literature index first")
    return QdrantVectorStore(client=client, collection_name=name, embedding=get_biomedical_embedder())


def retrieve_with_scores(store: QdrantVectorStore, query: str, candidates: int = 20) -> list[tuple[Document, float]]:
    """Return the real Qdrant cosine score; never manufacture relevance values."""
    return [(doc, float(score)) for doc, score in store.similarity_search_with_score(query, k=candidates)]


def retrieve_many_with_scores(
    store: QdrantVectorStore, queries: list[str], candidates: int = 20
) -> list[list[tuple[Document, float]]]:
    """Batch MedCPT query encoding, preserving one Qdrant result list per query."""
    embedder = store.embeddings
    if embedder is None or not hasattr(embedder, "embed_queries"):
        return [retrieve_with_scores(store, query, candidates) for query in queries]
    vectors = embedder.embed_queries(queries)
    return [
        [(doc, float(score)) for doc, score in store.similarity_search_with_score_by_vector(vector, k=candidates)]
        for vector in vectors
    ]


def list_gene_collections(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> list[str]:
    client = QdrantClient(host=host, port=port)
    prefix, suffix = "literature_", "_medcpt_v1"
    return [c.name[len(prefix):-len(suffix)].upper() for c in client.get_collections().collections
            if c.name.startswith(prefix) and c.name.endswith(suffix)]
