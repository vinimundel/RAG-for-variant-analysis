"""Build a fresh, unannotated Task 11F confirmatory benchmark.

This command never reads a filled annotation sheet.  It selects variants from
the structural catalog, excludes the previous 24 variant identities only to
ensure a new sample, retrieves five passages, and freezes the rubric and
thresholds before any human labels exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.config import ROOT, get_gene_config, paths_for
from pipeline.steps.build_rag_benchmark import primary_corpus_passages
from pipeline.steps.run_literature_reranking import (
    _cache_path, _discovery_request, _retrieval_cache_path, resolve_retrieval_layer,
)
from src.features.discovery_hypotheses import annotate_discovery_hypotheses
from src.literature.benchmark import (
    BENCHMARK_VARIANTS, HUMAN_REQUIRED_COLUMNS, PASSAGES_PER_VARIANT,
    STRATA, corpus_mentions, motif_positions, sample_variants,
)
from src.literature.task11 import sha256
from src.rag.bm25_index import bm25_corpus_path
from src.rag.chain import DiscoveryLiteratureAnalyzer
from src.rag.schemas import DiscoveryRAGResult, LiteratureEvidence


CONFIRMATORY_VERSION = "task11f1_confirmatory_v5"
CONFIRMATORY_SEED = "braf_rag_confirmatory_v2_frozen"
THRESHOLDS = {
    "precision_at_5": 0.70, "citation_precision": 1.00,
    "exact_claim_precision": 1.00, "intra_rater_kappa": 0.60,
}
VOCABULARIES = {
    "relevance": ["irrelevant", "background", "relevant", "directly_on_point"],
    "specificity": ["exact_variant", "same_residue", "functional_region", "general"],
    "entailment": ["supports", "neutral", "contradicts"],
    "support_type": ["structural_biophysical", "biochemical", "cell_functional", "in_vivo",
                      "human_association", "computational", "context"],
    "quality": ["controlled", "uncontrolled", "unclear"],
    "conflict": ["yes", "no"],
}


def _citation_key(variant: str, pmid: str, passage: str) -> str:
    return "SC-" + hashlib.sha256(f"{variant}|{pmid}|{passage}".encode()).hexdigest()[:16]


def _question(variant: str, region: str) -> str:
    return (
        f"Does this primary-literature passage directly support a mechanistic or functional "
        f"claim about BRAF {variant} ({region}), or is it only analogous/contextual evidence?"
    )


def build(gene: str = "BRAF", total: int = BENCHMARK_VARIANTS,
          passages: int = PASSAGES_PER_VARIANT, reuse_caches: bool = False) -> Path:
    gene = gene.upper()
    if gene != "BRAF":
        raise ValueError("Task 11F is currently defined for BRAF")
    cfg, paths = get_gene_config(gene), paths_for(gene)
    rag = paths["rag"]
    index_manifest = rag / f"{gene.lower()}_rag_index_manifest.json"
    layer = resolve_retrieval_layer(index_manifest)
    corpus = primary_corpus_passages(rag, layer)
    by_position = corpus_mentions(corpus)
    frame = pd.read_parquet(paths["scores"] / f"{gene.lower()}_structural_variant_ranking.parquet")
    frame = annotate_discovery_hypotheses(frame, cfg["regions"], cfg.get("functional_motifs", {}), gene=gene)
    old_sheet = rag / "benchmark" / f"{gene.lower()}_rag_benchmark_sheet.csv"
    previous = pd.read_csv(old_sheet) if old_sheet.exists() else pd.DataFrame(columns=["mutation"])
    previous_mutations = set(previous.get("mutation", pd.Series(dtype=str)).astype(str))
    candidates = frame.loc[~frame["mutation"].astype(str).isin(previous_mutations)].copy()
    selected, design = sample_variants(
        candidates, by_position, motif_positions(cfg.get("functional_motifs", {})),
        seed=CONFIRMATORY_SEED, total=total,
    )
    if len(selected) != total:
        raise RuntimeError(f"confirmatory sample has {len(selected)} variants; required {total}")

    analyzer = DiscoveryLiteratureAnalyzer(
        gene, candidate_k=30, rag_dir=rag, layer=layer,
    )
    index_fingerprint = hashlib.sha256(index_manifest.read_bytes()).hexdigest()[:12]
    rows = []
    inference_records = []
    for row in selected.to_dict("records"):
        variant = str(row["mutation"])
        request = _discovery_request(row, gene, "braf_oncology", (), "discovery_v4")
        retrieval_cache = _retrieval_cache_path(
            rag / "retrieval_cache", variant, "braf_oncology", (), "discovery_v4", index_fingerprint,
        )
        inference_cache = _cache_path(
            rag / "inference_cache", variant, "braf_oncology", (), "discovery_v4", index_fingerprint,
        )
        if reuse_caches and retrieval_cache.exists() and inference_cache.exists():
            retrieval_payload = json.loads(retrieval_cache.read_text(encoding="utf-8"))
            channel_queries = retrieval_payload.get("retrieval_channels", {})
            retrieval_diagnostics = retrieval_payload.get("retrieval_diagnostics", {})
            evidence = [LiteratureEvidence.model_validate(item)
                        for item in retrieval_payload["evidence"][:passages]]
            selected_channels = retrieval_payload.get("retrieval_diagnostics", {}).get(
                "channels_per_selected_document", {}
            )
            evidence = [item.model_copy(update={
                "retrieval_channels": list(selected_channels.get(str(item.pmid), []))
            }) for item in evidence]
            result = DiscoveryRAGResult.model_validate_json(inference_cache.read_text(encoding="utf-8"))
            retrieval_source = "existing_variant_caches_reused_without_annotations"
        else:
            channel_queries = analyzer._channel_queries(request)
            evidence = analyzer.retrieve_discovery(
                channel_queries, passages, variant=variant,
                functional_region=request.functional_region,
                reranker_query=(f"{gene} {variant} mechanistic question: "
                                "Does the evidence directly support a mechanistic effect?"),
            )
            result = analyzer.infer_discovery(
                request, evidence=evidence,
                retrieval_queries=[query for values in channel_queries.values() for query in values],
            )
            retrieval_diagnostics = analyzer.last_retrieval_diagnostics
            retrieval_source = "fresh_retrieval"
        inference_records.append({
            "variant": variant, "result": result.model_dump(mode="json"),
            "retrieval_diagnostics": retrieval_diagnostics,
        })
        cited = {
            evidence_id for hypothesis in result.inference.hypotheses
            for evidence_id in hypothesis.cited_evidence_ids
        }
        cited.update(result.inference.disease_association.cited_evidence_ids)
        for rank, item in enumerate(evidence, 1):
            rows.append({
                "item_id": f"C{len(rows) + 1:03d}", "mutation": variant,
                "claim_question": _question(variant, str(row["functional_region"])),
                "stratum": str(row["stratum"]), "rank": rank, "pmid": str(item.pmid),
                "article_title": item.article_title, "section": item.section,
                "corpus_layer": item.corpus_layer, "reranker_score": item.reranker_score,
                "passage": item.passage, "cited_in_claim": (
                    "yes" if item.evidence_id in cited else "no"
                ), "system_citation_key": _citation_key(variant, item.pmid, item.passage),
                **{column: "" for column in HUMAN_REQUIRED_COLUMNS},
            })
    sheet = pd.DataFrame(rows)
    if len(sheet) != total * passages:
        raise RuntimeError(f"confirmatory sheet has {len(sheet)} rows; required {total * passages}")

    out_dir = rag / "benchmark_v5_confirmatory"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_sheet.csv"
    sheet.to_csv(sheet_path, index=False)
    template_path = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_sheet_template.csv"
    sheet.to_csv(template_path, index=False)

    # The repeat contains the mutation/question/passage but none of the mapping
    # or document metadata that could reveal its original row.
    repeat_rows = []
    repeat_key_rows = []
    for index, row in sheet.sample(frac=1, random_state=211).head(total).iterrows():
        repeat_item_id = f"R{len(repeat_rows) + 1:03d}"
        repeat_rows.append({
            "repeat_item_id": repeat_item_id,
            "mutation": row["mutation"], "claim_question": row["claim_question"],
            "passage": row["passage"],
            **{column: "" for column in HUMAN_REQUIRED_COLUMNS},
        })
        repeat_key_rows.append({"repeat_item_id": repeat_item_id, "original_item_id": row["item_id"]})
    repeat = pd.DataFrame(repeat_rows)
    repeat_path = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_blind_repeats.csv"
    repeat.to_csv(repeat_path, index=False)
    repeat_template_path = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_blind_repeats_template.csv"
    repeat.to_csv(repeat_template_path, index=False)
    repeat_key_path = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_repeat_key.json"
    repeat_key_path.write_text(json.dumps({
        "status": "sealed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "mapping": repeat_key_rows,
    }, indent=2), encoding="utf-8")
    os.chmod(repeat_key_path, 0o600)
    inference_path = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_inference.json"
    inference_path.write_text(json.dumps(inference_records, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "manifest_version": CONFIRMATORY_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "gene": gene, "status": "frozen_before_annotation", "confirmatory": True,
        "seed": CONFIRMATORY_SEED, "variants": total, "passages_per_variant": passages,
        "passages_total": len(sheet), "blind_repeats": len(repeat),
        "sampling_design": design, "strata": STRATA,
        "previous_variant_identities_excluded": sorted(previous_mutations),
        "development_filled_annotations_read": False,
        "development_filled_annotations_excluded": [
            "data/output/BRAF/rag/benchmark/braf_rag_benchmark_filled.csv",
            "data/output/BRAF/rag/benchmark/braf_rag_benchmark_blind_filled.csv",
        ],
        "thresholds_frozen_before_annotation": THRESHOLDS,
        "vocabularies_frozen_before_annotation": VOCABULARIES,
        "rubric": {
            "same_residue": "explicit analogy; never an exact-variant claim",
            "same_domain": "context only; cannot materially change a score",
            "therapy_context": "cannot support a mechanistic claim without direct evidence",
            "citation": "the cited passage must support the exact wording of the claim",
        },
        "full_sheet": str(sheet_path.relative_to(ROOT)),
        "full_sheet_template": str(template_path.relative_to(ROOT)),
        "blind_repeat_sheet": str(repeat_path.relative_to(ROOT)),
        "blind_repeat_sheet_template": str(repeat_template_path.relative_to(ROOT)),
        "blind_repeat_key": str(repeat_key_path.relative_to(ROOT)),
        "inference_artifact": str(inference_path.relative_to(ROOT)),
        "retrieval_source": retrieval_source,
        "retrieval_stack_version": "rag_stack_v5",
        "sha256": {
            "full_sheet": sha256(sheet_path), "full_sheet_template": sha256(template_path),
            "blind_repeat_sheet": sha256(repeat_path), "blind_repeat_sheet_template": sha256(repeat_template_path),
            "blind_repeat_key": sha256(repeat_key_path),
            "inference_artifact": sha256(inference_path),
            "index_manifest": sha256(index_manifest),
        },
        "blind_visibility": {
            "shown": ["mutation", "claim_question", "passage"],
            "hidden": ["original_item_id", "rank", "pmid", "article_title", "section",
                       "corpus_layer", "reranker_score", "system_citation_key", "first_evaluation_link"],
            "mapping_blinded": True,
        },
        "promotion_status": "blocked_until_human_annotation_and_confirmatory_gate",
    }
    manifest_path = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    parser.add_argument("--variants", type=int, default=BENCHMARK_VARIANTS)
    parser.add_argument("--passages", type=int, default=PASSAGES_PER_VARIANT)
    parser.add_argument("--reuse-caches", action="store_true")
    args = parser.parse_args()
    print(build(args.gene, args.variants, args.passages, args.reuse_caches))
