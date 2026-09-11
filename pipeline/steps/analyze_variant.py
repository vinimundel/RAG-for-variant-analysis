"""Analyze one explicit variant query against a locally prepared RAG index."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pipeline.config import get_gene_config, paths_for
from pipeline.steps.run_literature_reranking import resolve_retrieval_layer
from src.rag.chain import DiscoveryLiteratureAnalyzer
from src.rag.query_profiles import get_condition_profile
from src.rag.schemas import DiscoveryQuery, DiscoveryRAGResult


def analyze(request: DiscoveryQuery, rag_dir: Path | None = None) -> DiscoveryRAGResult:
    """Retrieve and infer without requiring a structural ranking table.

    The request supplies the scientific context explicitly. Missing index manifests,
    unsupported gene profiles, and retrieval failures propagate to the caller.
    """
    request = request.model_copy(update={'gene': request.gene.upper()})
    get_gene_config(request.gene)
    get_condition_profile(request.condition_profile)
    if request.gene != 'GSDMD' and request.condition_profile != 'braf_oncology':
        raise ValueError('BRAF queries require the braf_oncology condition profile')
    directory = Path(rag_dir) if rag_dir is not None else paths_for(request.gene)['rag']
    manifest = directory / f'{request.gene.lower()}_rag_index_manifest.json'
    fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
    layer = resolve_retrieval_layer(manifest)
    analyzer = DiscoveryLiteratureAnalyzer(request.gene, candidate_k=30,
                                           rag_dir=directory, layer=layer)
    channel_queries = analyzer._channel_queries(request)
    queries = [query for channel in channel_queries.values() for query in channel]
    try:
        evidence = analyzer.retrieve_discovery(
            channel_queries, request.k, variant=request.variant,
            functional_region=request.functional_region,
            reranker_query=f'{request.gene} {request.variant} mechanistic effect',
        )
    finally:
        analyzer.release_retrieval_models()
    result = analyzer.infer_discovery(request, evidence=evidence, retrieval_queries=queries)
    result.retrieval_index_fingerprint = fingerprint
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', type=Path, required=True,
                        help='JSON matching src.rag.schemas.DiscoveryQuery')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rag-dir', type=Path, help='Override the local gene index directory')
    args = parser.parse_args()
    request = DiscoveryQuery.model_validate_json(args.query.read_text(encoding='utf-8'))
    result = analyze(request, args.rag_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
