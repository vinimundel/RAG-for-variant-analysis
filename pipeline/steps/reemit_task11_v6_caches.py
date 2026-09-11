"""Re-emit V6 caches by separating archived V5 evidence blocks.

No embeddings, reranker calls, LLM calls, or human annotation files are read.
The archived V5 retrieval payload is partitioned by its already-recorded scope;
the new cache namespace and fingerprints make the transformation auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline.config import ROOT, paths_for
from pipeline.steps.run_literature_reranking import _cache_path, _retrieval_cache_path
from src.literature.task11 import RETRIEVAL_STACK_VERSION, sha256
from src.rag.retrieval_v5 import MECHANISTIC_SCOPES
from src.rag.schemas import DiscoveryRAGResult


def reemit(gene: str = "BRAF") -> Path:
    gene = gene.upper()
    paths = paths_for(gene)
    rag = paths["rag"]
    archive = rag / "archive" / "task11_stack_v5_context_separation"
    old_retrieval = archive / "retrieval_cache"
    old_inference = archive / "inference_cache"
    retrieval_paths = sorted(old_retrieval.glob("*__discovery_v4__*.json"))
    inference_paths = sorted(old_inference.glob("*__discovery_v4__*.json"))
    if len(retrieval_paths) != 171 or len(inference_paths) != 171:
        raise RuntimeError("archived V5 cache set is not exactly 171 + 171")
    index_manifest = rag / f"{gene.lower()}_rag_index_manifest.json"
    index_fingerprint = hashlib.sha256(index_manifest.read_bytes()).hexdigest()[:12]
    stack_manifest = rag / f"{gene.lower()}_rag_stack_manifest.json"
    stack_fingerprint = json.loads(stack_manifest.read_text(encoding="utf-8")).get("stack_fingerprint")
    stack_hash = sha256(stack_manifest)
    retrieval_dir = rag / "retrieval_cache"
    inference_dir = rag / "inference_cache"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)
    retrieval_by_variant = {}
    for old_path in retrieval_paths:
        payload = json.loads(old_path.read_text(encoding="utf-8"))
        variant = str(payload["variant"])
        evidence = list(payload.get("evidence", []))
        mechanistic = [item for item in evidence
                       if str(item.get("evidence_scope")) in MECHANISTIC_SCOPES]
        context = [item for item in evidence
                   if str(item.get("evidence_scope")) == "general_context"]
        if set(id(item) for item in mechanistic) & set(id(item) for item in context):
            raise RuntimeError(f"block overlap in {old_path.name}")
        diagnostics = dict(payload.get("retrieval_diagnostics", {}))
        diagnostics.update({
            "retrieval_stack_version": RETRIEVAL_STACK_VERSION,
            "mechanistic_evidence_count": len(mechanistic),
            "context_reference_count": len(context),
            "mechanistic_evidence_status": (
                "mechanistic_evidence_retrieved" if mechanistic
                else "no_mechanistic_evidence_retrieved"
            ),
            "context_can_fill_mechanistic_top_k": False,
            "reemit_source": "archived_v5_retrieval_cache",
        })
        updated = {
            **payload,
            "retrieval_version": "discovery_v5",
            "query_version": "discovery_v5",
            "stack_fingerprint": stack_fingerprint,
            "stack_manifest_sha256": stack_hash,
            "retrieval_stack_version": RETRIEVAL_STACK_VERSION,
            "retrieval_diagnostics": diagnostics,
            "evidence": mechanistic + context,
            "mechanistic_evidence": mechanistic,
            "context_reference": context,
            "mechanistic_evidence_status": (
                "mechanistic_evidence_retrieved" if mechanistic
                else "no_mechanistic_evidence_retrieved"
            ),
            "reemit_source": "archived_v5_retrieval_cache",
            "retrieval_recalculated": False,
        }
        output = _retrieval_cache_path(
            retrieval_dir, variant, str(payload.get("condition_profile", "braf_oncology")),
            tuple(payload.get("condition_terms", [])), "discovery_v5", index_fingerprint,
        )
        output.write_text(json.dumps(updated, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        retrieval_by_variant[variant] = updated

    for old_path in inference_paths:
        old = DiscoveryRAGResult.model_validate_json(old_path.read_text(encoding="utf-8"))
        payload = retrieval_by_variant.get(old.inference.variant)
        if payload is None:
            raise RuntimeError(f"no V6 retrieval payload for {old.inference.variant}")
        mechanistic = [type_item for type_item in old.evidence
                       if str(type_item.evidence_scope) in MECHANISTIC_SCOPES]
        context = [type_item for type_item in old.evidence
                   if str(type_item.evidence_scope) == "general_context"]
        updated = old.model_copy(update={
            "evidence": mechanistic + context,
            "mechanistic_evidence": mechanistic,
            "context_reference": context,
            "mechanistic_evidence_status": (
                "mechanistic_evidence_retrieved" if mechanistic
                else "no_mechanistic_evidence_retrieved"
            ),
            "retrieval_version": "discovery_v5",
            "query_version": "discovery_v5",
            "retrieval_index_fingerprint": index_fingerprint,
            "retrieval_stack_fingerprint": stack_fingerprint,
            "retrieval_stack_manifest_sha256": stack_hash,
        })
        output = _cache_path(
            inference_dir, old.inference.variant, old.condition_profile,
            tuple(old.condition_terms), "discovery_v5", index_fingerprint,
        )
        output.write_text(updated.model_dump_json(indent=2), encoding="utf-8")

    manifest = {
        "gene": gene, "condition_profile": "braf_oncology", "query_version": "discovery_v5",
        "retrieval_stack_version": RETRIEVAL_STACK_VERSION,
        "retrieval_stack_manifest": stack_manifest.name,
        "retrieval_stack_manifest_sha256": stack_hash,
        "retrieval_stack_fingerprint": stack_fingerprint,
        "index_fingerprint": index_fingerprint, "variants_total": 171,
        "retrieval_caches_available": len(list(retrieval_dir.glob("*__discovery_v5__*.json"))),
        "inference_caches_available": len(list(inference_dir.glob("*__discovery_v5__*.json"))),
        "retrieval_failures": {}, "source": "archived_v5_caches_partitioned_without_model_calls",
        "embeddings_rebuilt": False, "reranker_reexecuted": False, "llm_reexecuted": False,
    }
    output = rag / f"{gene.lower()}_retrieval_phase_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(reemit(args.gene))
