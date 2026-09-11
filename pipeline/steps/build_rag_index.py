"""Build and smoke-test the per-gene MedCPT literature index."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.literature.corpus_layers import INDEXED_LAYERS, UNLAYERED
from src.rag.ingestor import extract_tei_for_gene, load_cached_tei_for_gene, load_pdfs_for_gene
from src.rag.bm25_index import PersistentBM25, bm25_corpus_path
from src.rag.embedder import DOCUMENT_MODEL, EMBEDDING_DIM, QUERY_MODEL
from src.rag.reranker import CROSS_ENCODER_MODEL
from src.rag.vector_store import (
    build_vector_store, collection_name, load_vector_store, retrieve_with_scores,
)
from src.rag.query_profiles import QUERY_VERSION

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_by_layer(documents) -> dict[str, list]:
    """Split chunks by curated corpus layer, preserving order inside each layer."""
    grouped: dict[str, list] = {}
    for document in documents:
        grouped.setdefault(str(document.metadata.get("corpus_layer", UNLAYERED)), []).append(document)
    return grouped


def indexable_layers(grouped: dict[str, list]) -> list[str | None]:
    """Decide which collections to build.

    A curated corpus produces one collection per indexed layer and never indexes
    ``excluded``.  A corpus with no curation at all falls back to the single
    pre-layer collection, so an uncurated gene stays usable without silently
    being relabelled as primary evidence.
    """
    curated = [layer for layer in INDEXED_LAYERS if grouped.get(layer)]
    if curated:
        return list(curated)
    if grouped.get(UNLAYERED):
        return [None]
    raise RuntimeError("No indexable documents: every chunk is in the excluded layer")


def close_store(store) -> None:
    """Release the embedded Qdrant file lock so the next layer can be built.

    The local engine locks its directory per client, so two layers cannot hold
    open clients on the same path at the same time.
    """
    close = getattr(getattr(store, "client", None), "close", None)
    if callable(close):
        close()


def _write_manifest(documents, layer_stats: dict, rag_dir: Path, gene: str, mode: str) -> Path:
    grouped = group_by_layer(documents)
    counts = {
        str(layer): {**stats, "bm25_corpus_sha256": _sha256(bm25_corpus_path(rag_dir, layer))}
        for layer, stats in layer_stats.items()
    }
    indexed_keys = {layer or UNLAYERED for layer in layer_stats}
    documents = [doc for layer, chunks in grouped.items() if layer in indexed_keys
                 for doc in chunks]
    if any(doc.metadata.get("corpus_layer") == "excluded" for doc in documents):
        raise RuntimeError("Index build attempted to include an excluded document")
    if any(doc.metadata.get("corpus_layer") == "discovery_only" for doc in documents):
        raise RuntimeError("Index build attempted to include a discovery-only document")
    papers = {str(doc.metadata.get("pmid")) for doc in documents}
    pdf_inventory = (ROOT / "data" / "input" / gene.upper() / "evidence" /
                     f"{gene.lower()}_literature_evidence.csv")
    extraction_manifest = rag_dir / f"{gene.lower()}_grobid_extraction_manifest.json"
    layer_manifest = (ROOT / "data" / "input" / gene.upper() / "evidence" /
                      f"{gene.lower()}_corpus_layers.manifest.json")
    legacy_bm25 = rag_dir / "archive" / "pre_curation" / "bm25_corpus.jsonl"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gene": gene.upper(), "storage_mode": mode,
        "indexed_layers": counts,
        "indexed_documents": len(documents), "indexed_papers": len(papers),
        "layer_chunk_counts": {str(layer): len(chunks) for layer, chunks in sorted(grouped.items())},
        # Excluded chunks are counted so a shrinking index is never a surprise.
        "excluded_chunks_not_indexed": len(grouped.get("excluded", [])),
        "excluded_papers_not_indexed": len(
            {str(doc.metadata.get("pmid")) for doc in grouped.get("excluded", [])}),
        "parsers": sorted({str(doc.metadata.get("parser")) for doc in documents}),
        "embedding_dimension": EMBEDDING_DIM,
        "document_encoder": DOCUMENT_MODEL, "query_encoder": QUERY_MODEL,
        "cross_encoder": CROSS_ENCODER_MODEL,
        "pdf_inventory": str(pdf_inventory.relative_to(ROOT)) if pdf_inventory.exists() else None,
        "pdf_inventory_sha256": _sha256(pdf_inventory),
        "corpus_layers_manifest": str(layer_manifest.relative_to(ROOT))
        if layer_manifest.exists() else None,
        "corpus_layers_manifest_sha256": _sha256(layer_manifest),
        "grobid_extraction_manifest_sha256": _sha256(extraction_manifest),
        "legacy_unlayered_index": {
            "status": "archived_not_active",
            "path": str(legacy_bm25.relative_to(ROOT)) if legacy_bm25.exists() else None,
            "sha256": _sha256(legacy_bm25),
            "excluded_never_indexed_in_active_corpus": True,
        },
        "source_type_counts": {
            str(kind): sum(doc.metadata.get("source_type") == kind for doc in documents)
            for kind in sorted({str(doc.metadata.get("source_type")) for doc in documents})
        },
        "retrieval_version": "gsdmd_discovery_v1" if gene.upper() == "GSDMD" else "discovery_v3",
        "query_version": QUERY_VERSION if gene.upper() == "GSDMD" else "discovery_v3",
        "source_collection_counts": {
            str(kind): sum(doc.metadata.get("source_collection") == kind for doc in documents)
            for kind in sorted({str(doc.metadata.get("source_collection", "unknown")) for doc in documents})
        },
        "scientific_boundary": "Retrieval relevance scores are not biological confidence.",
        "layer_policy": {
            "indexed": ["primary_evidence", "context_reference"],
            "excluded_never_indexed": True,
            "reviews_context_only": True,
            "preprints": "excluded",
        },
        "preserved_provenance_fields": [
            "pmid", "doi", "title", "section", "parser", "publication",
            "publication_types", "source_type", "evidence_type",
        ],
    }
    output = rag_dir / f"{gene.lower()}_rag_index_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--grobid-workers", type=int, default=1)
    parser.add_argument("--qdrant-server", action="store_true",
                        help="Use external Qdrant instead of the default embedded persistent engine")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract-only", action="store_true")
    mode.add_argument("--index-only", action="store_true")
    mode.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rag_dir = ROOT / "data" / "output" / args.gene.upper() / "rag"
    if args.extract_only:
        summary = extract_tei_for_gene(
            ROOT, args.gene, grobid_url=args.grobid_url, workers=args.grobid_workers
        )
        print(f"GROBID cache complete: {summary['tei_available']}/{summary['pdfs']} TEIs; "
              "stop GROBID before --index-only")
        return
    documents = (load_cached_tei_for_gene(ROOT, args.gene)
                 if (args.index_only or args.manifest_only)
                 else load_pdfs_for_gene(ROOT, args.gene, grobid_url=args.grobid_url,
                                         workers=args.grobid_workers))
    local_path = None if args.qdrant_server else rag_dir / "qdrant"
    grouped = group_by_layer(documents)
    layers = indexable_layers(grouped)
    # The layer mechanistic queries will actually use; its smoke test decides
    # whether the build succeeded, so an empty primary index cannot pass quietly.
    primary = "primary_evidence" if "primary_evidence" in layers else layers[0]
    layer_stats, smoke = {}, None
    for layer in layers:
        chunks = grouped[layer or UNLAYERED]
        if args.manifest_only:
            store = load_vector_store(args.gene, args.qdrant_host, args.qdrant_port,
                                      local_path=local_path, layer=layer)
        else:
            PersistentBM25(chunks).save(bm25_corpus_path(rag_dir, layer))
            store = build_vector_store(chunks, args.gene, args.qdrant_host, args.qdrant_port,
                                       args.overwrite, local_path=local_path, layer=layer)
        name = collection_name(args.gene, layer)
        layer_stats[layer] = {
            "collection": name,
            "qdrant_points": int(store.client.count(name, exact=True).count),
            "documents": len(chunks),
            "papers": len({str(doc.metadata.get("pmid")) for doc in chunks}),
        }
        if layer == primary:
            smoke = retrieve_with_scores(
                store, f"{args.gene} protein interface structural mechanism", 3)
        close_store(store)
        print(f"Layer {layer or UNLAYERED}: indexed {len(chunks)} chunks", flush=True)
    if not smoke:
        raise RuntimeError(f"Layer {primary!r} was built but returned zero smoke-test candidates")
    manifest = _write_manifest(
        documents, layer_stats, rag_dir, args.gene, "server" if args.qdrant_server else "embedded"
    )
    indexed = sum(len(grouped[layer or UNLAYERED]) for layer in layers)
    print(f"Indexed {indexed} chunks across {len(layers)} layer(s); "
          f"{len(grouped.get('excluded', []))} excluded chunks were not indexed; "
          f"smoke-test scores: {[round(s, 4) for _, s in smoke]}; manifest: {manifest}")


if __name__ == "__main__":
    main()
