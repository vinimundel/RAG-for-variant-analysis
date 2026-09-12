"""Extract with GROBID or build a hybrid index from existing TEI artifacts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from healthrag.pipeline.config import ROOT, paths_for, validate_collection
from healthrag.literature.corpus_layers import INDEXED_LAYERS, UNLAYERED
from healthrag.rag.bm25_index import PersistentBM25, bm25_corpus_path
from healthrag.rag.embedder import DOCUMENT_MODEL, QUERY_MODEL, EMBEDDING_DIM
from healthrag.rag.ingestor import (
    extract_tei_for_collection,
    load_cached_tei_for_collection,
    load_pdfs_for_collection,
)
from healthrag.rag.reranker import CROSS_ENCODER_MODEL
from healthrag.rag.vector_store import build_vector_store, collection_name, retrieve_with_scores
from healthrag.runtime import require_memory


def group_by_layer(documents):
    grouped = {}
    for document in documents:
        grouped.setdefault(document.metadata.get('corpus_layer', UNLAYERED), []).append(document)
    return grouped


def indexable_layers(grouped):
    layers = [layer for layer in INDEXED_LAYERS if grouped.get(layer)]
    if not layers:
        raise ValueError('Curate primary_evidence/context_reference layers before indexing')
    return layers


def build(collection: str, documents: list, overwrite: bool = False) -> dict:
    require_memory("Index build", 3.0)
    collection = validate_collection(collection)
    directory = paths_for(collection)['rag']
    directory.mkdir(parents=True, exist_ok=True)
    grouped = group_by_layer(documents)
    stats = {}
    for layer in indexable_layers(grouped):
        chunks = grouped[layer]
        store = build_vector_store(chunks, collection, overwrite=overwrite,
                                   local_path=directory / 'qdrant', layer=layer)
        try:
            smoke = retrieve_with_scores(store, f'{collection} biomedical research', 1)
            if not smoke:
                raise RuntimeError(f'No candidates returned from {layer}')
            name = collection_name(collection, layer)
            stats[layer] = {'collection': name, 'chunks': len(chunks),
                            'qdrant_points': store.client.count(name, exact=True).count,
                            'papers': len({d.metadata['pmid'] for d in chunks})}
        finally:
            store.client.close()
        PersistentBM25(chunks).save(bm25_corpus_path(directory, layer))
    corpus_hash = hashlib.sha256(''.join(
        hashlib.sha256(bm25_corpus_path(directory, layer).read_bytes()).hexdigest()
        for layer in sorted(stats)).encode()).hexdigest()
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'collection': collection,
                'indexed_layers': stats, 'corpus_sha256': corpus_hash,
                'parsers': sorted({d.metadata.get('parser', 'unknown') for d in documents}),
                'document_encoder': DOCUMENT_MODEL, 'query_encoder': QUERY_MODEL,
                'cross_encoder': CROSS_ENCODER_MODEL, 'embedding_dimension': EMBEDDING_DIM,
                'excluded_chunks': len(grouped.get('excluded', [])),
                'unlayered_chunks': len(grouped.get(UNLAYERED, []))}
    (directory / f'{collection.lower()}_rag_index_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', required=True, type=validate_collection)
    parser.add_argument('--grobid-url', default='http://localhost:8070')
    parser.add_argument('--grobid-workers', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--extract-only', action='store_true')
    mode.add_argument('--index-only', action='store_true')
    args = parser.parse_args()
    if args.extract_only:
        result = extract_tei_for_collection(ROOT, args.collection, grobid_url=args.grobid_url,
                                            workers=args.grobid_workers)
    else:
        documents = (load_cached_tei_for_collection(ROOT, args.collection) if args.index_only else
                     load_pdfs_for_collection(ROOT, args.collection, grobid_url=args.grobid_url,
                                              workers=args.grobid_workers))
        result = build(args.collection, documents, args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
