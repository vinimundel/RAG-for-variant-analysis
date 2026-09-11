"""Freeze a confirmatory benchmark with separate evidence blocks.

This command consumes only completed V6 retrieval/inference caches.  It never
re-runs retrieval or inference and never reads a filled annotation file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

from pipeline.config import ROOT, get_gene_config, paths_for
from pipeline.steps.build_rag_benchmark import primary_corpus_passages
from pipeline.steps.run_literature_reranking import (
    _cache_path, _discovery_request, _retrieval_cache_path, resolve_retrieval_layer,
)
from src.features.discovery_hypotheses import annotate_discovery_hypotheses
from src.literature.benchmark import (
    BENCHMARK_VARIANTS, HUMAN_REQUIRED_COLUMNS, STRATA, corpus_mentions,
    motif_positions, sample_variants,
)
from src.literature.task11 import sha256
from src.rag.schemas import DiscoveryRAGResult, LiteratureEvidence


CURRENT_TAG = "v6"
PREVIOUS_TAG = "v5"
VERSION = "task11f2_confirmatory_v6"
SEED = "braf_rag_confirmatory_v3_mechanistic_only"
PASSAGES_PER_VARIANT = 5
THRESHOLDS = {
    "mechanistic_precision_at_5": 0.70,
    "variant_evidence_coverage": 0.50,
    "correct_abstention_rate": 0.80,
    "citation_precision": 1.00,
    "exact_claim_precision": 1.00,
    "scope_precision": 1.00,
    "relevance_kappa": 0.60,
    "abstention_kappa": 0.60,
}
ABSTENTION_VOCABULARY = ["correct_abstention", "missed_evidence"]


def _citation_key(variant: str, pmid: str, passage: str) -> str:
    return "SC-" + hashlib.sha256(f"{variant}|{pmid}|{passage}".encode()).hexdigest()[:16]


def _question(variant: str, region: str) -> str:
    return (
        f"Does this primary-literature passage directly support a mechanistic or functional "
        f"claim about BRAF {variant} ({region}), or is it only analogous evidence?"
    )


def _previous_variants(rag) -> set[str]:
    paths = [
        rag / f"benchmark_{PREVIOUS_TAG}_confirmatory" /
        f"braf_rag_benchmark_{PREVIOUS_TAG}_confirmatory_sheet.csv",
        rag / "benchmark" / "braf_rag_benchmark_sheet.csv",
    ]
    values = set()
    for path in paths:
        if path.exists():
            values.update(pd.read_csv(path).get("mutation", pd.Series(dtype=str)).astype(str))
    return values


def build(gene: str = "BRAF", total: int = BENCHMARK_VARIANTS) -> str:
    gene = gene.upper()
    if gene != "BRAF":
        raise ValueError("Task 11F.3 is currently defined for BRAF")
    cfg, paths = get_gene_config(gene), paths_for(gene)
    rag = paths["rag"]
    index_manifest = rag / f"{gene.lower()}_rag_index_manifest.json"
    layer = resolve_retrieval_layer(index_manifest)
    corpus = primary_corpus_passages(rag, layer)
    frame = pd.read_parquet(paths["scores"] / f"{gene.lower()}_structural_variant_ranking.parquet")
    frame = annotate_discovery_hypotheses(
        frame, cfg["regions"], cfg.get("functional_motifs", {}), gene=gene
    )
    frame = frame.loc[~frame["mutation"].astype(str).isin(_previous_variants(rag))].copy()
    selected, design = sample_variants(
        frame, corpus_mentions(corpus), motif_positions(cfg.get("functional_motifs", {})),
        seed=SEED, total=total,
    )
    if len(selected) != total:
        raise RuntimeError(f"{CURRENT_TAG.upper()} sample has {len(selected)} variants; required {total}")

    condition_profile = str(cfg["discovery_profile"])
    query_version = "discovery_v5"
    index_fingerprint = hashlib.sha256(index_manifest.read_bytes()).hexdigest()[:12]
    rows, abstention_rows, records = [], [], []
    for selected_row in selected.to_dict("records"):
        variant = str(selected_row["mutation"])
        request = _discovery_request(selected_row, gene, condition_profile, (), query_version)
        retrieval_path = _retrieval_cache_path(
            rag / "retrieval_cache", variant, condition_profile, (), query_version, index_fingerprint,
        )
        inference_path = _cache_path(
            rag / "inference_cache", variant, condition_profile, (), query_version, index_fingerprint,
        )
        if not retrieval_path.exists() or not inference_path.exists():
            raise RuntimeError(f"missing completed {CURRENT_TAG.upper()} cache for {variant}")
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        inference = DiscoveryRAGResult.model_validate_json(inference_path.read_text(encoding="utf-8"))
        mechanistic = [
            LiteratureEvidence.model_validate(item)
            for item in retrieval.get("mechanistic_evidence", [])
        ][:PASSAGES_PER_VARIANT]
        context_count = len(retrieval.get("context_reference", []))
        status = str(retrieval.get("mechanistic_evidence_status", ""))
        if bool(mechanistic) != (status == "mechanistic_evidence_retrieved"):
            raise RuntimeError(f"invalid {CURRENT_TAG.upper()} abstention status for {variant}")
        records.append({
            "variant": variant, "retrieval_cache": str(retrieval_path.relative_to(ROOT)),
            "inference_cache": str(inference_path.relative_to(ROOT)),
            "retrieval_cache_sha256": sha256(retrieval_path),
            "inference_cache_sha256": sha256(inference_path),
            "mechanistic_evidence_count": len(mechanistic),
            "context_reference_count": context_count,
            "mechanistic_evidence_status": status,
        })
        if not mechanistic:
            abstention_rows.append({
                "item_id": f"A{len(abstention_rows) + 1:03d}", "mutation": variant,
                "claim_question": _question(variant, str(selected_row["functional_region"])),
                "stratum": selected_row["stratum"],
                "mechanistic_evidence_status": "no_mechanistic_evidence_retrieved",
                "mechanistic_evidence_count": 0,
                "abstention_judgment": "",
            })
        cited = {
            evidence_id for hypothesis in inference.inference.hypotheses
            for evidence_id in hypothesis.cited_evidence_ids
        }
        cited.update(inference.inference.disease_association.cited_evidence_ids)
        for item in mechanistic:
            variant_rank = sum(1 for existing in rows if existing["mutation"] == variant) + 1
            rows.append({
                "item_id": f"M{len(rows) + 1:03d}", "mutation": variant,
                "claim_question": _question(variant, str(selected_row["functional_region"])),
                "stratum": selected_row["stratum"], "rank": variant_rank,
                "pmid": item.pmid, "article_title": item.article_title, "section": item.section,
                "corpus_layer": item.corpus_layer, "reranker_score": item.reranker_score,
                "passage": item.passage, "automatic_evidence_scope": item.evidence_scope,
                "mechanistic_evidence_status": status,
                "context_reference_count": context_count,
                "cited_in_claim": "yes" if item.evidence_id in cited else "no",
                "system_citation_key": _citation_key(variant, item.pmid, item.passage),
                **{column: "" for column in HUMAN_REQUIRED_COLUMNS},
            })

    out_dir = rag / f"benchmark_{CURRENT_TAG}_confirmatory"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = pd.DataFrame(rows)
    if rows:
        sheet = sheet.sort_values(["mutation", "rank"]).reset_index(drop=True)
    sheet_path = out_dir / f"{gene.lower()}_rag_benchmark_{CURRENT_TAG}_confirmatory_sheet.csv"
    template_path = out_dir / f"{gene.lower()}_rag_benchmark_{CURRENT_TAG}_confirmatory_sheet_template.csv"
    sheet.to_csv(sheet_path, index=False)
    sheet.to_csv(template_path, index=False)
    abstention = pd.DataFrame(abstention_rows)
    abstention_path = out_dir / f"{gene.lower()}_rag_benchmark_{CURRENT_TAG}_confirmatory_abstention_sheet.csv"
    abstention_template = out_dir / f"{gene.lower()}_rag_benchmark_{CURRENT_TAG}_confirmatory_abstention_sheet_template.csv"
    abstention.to_csv(abstention_path, index=False)
    abstention.to_csv(abstention_template, index=False)

    # One repeat per sampled variant, including an explicit empty passage for
    # variants where the system abstained. The mapping remains sealed.
    first_by_variant = {}
    for row in rows:
        first_by_variant.setdefault(str(row["mutation"]), row)
    repeat_rows, mapping = [], []
    abstention_by_variant = {str(row["mutation"]): row for row in abstention_rows}
    for index, selected_row in enumerate(selected.to_dict("records"), 1):
        variant = str(selected_row["mutation"])
        source = first_by_variant.get(variant)
        repeat_id = f"R{index:03d}"
        repeat_rows.append({
            "repeat_item_id": repeat_id, "mutation": variant,
            "claim_question": _question(variant, str(selected_row["functional_region"])),
            "passage": source["passage"] if source else "",
            "mechanistic_evidence_status": (
                "mechanistic_evidence_retrieved" if source
                else "no_mechanistic_evidence_retrieved"
            ),
            "abstention_judgment": "" if source else "",
            **{column: "" for column in HUMAN_REQUIRED_COLUMNS},
        })
        mapping.append({
            "repeat_item_id": repeat_id,
            "original_item_id": source["item_id"] if source else abstention_by_variant[variant]["item_id"],
            "annotation_domain": "mechanistic" if source else "abstention",
        })
    repeat_path = out_dir / f"{gene.lower()}_rag_benchmark_{CURRENT_TAG}_confirmatory_blind_repeats.csv"
    repeat_template = out_dir / f"{gene.lower()}_rag_benchmark_{CURRENT_TAG}_confirmatory_blind_repeats_template.csv"
    pd.DataFrame(repeat_rows).to_csv(repeat_path, index=False)
    pd.DataFrame(repeat_rows).to_csv(repeat_template, index=False)
    key_path = out_dir / f"{gene.lower()}_rag_benchmark_{CURRENT_TAG}_confirmatory_repeat_key.json"
    key_path.write_text(json.dumps({"status": "sealed", "mapping": mapping}, indent=2), encoding="utf-8")
    os.chmod(key_path, 0o600)

    inference_path = out_dir / f"{gene.lower()}_rag_benchmark_{CURRENT_TAG}_confirmatory_inference.json"
    inference_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    controls_path = rag / f"{gene.lower()}_task11f2_v6_control_manifest.json"
    stack_path = rag / f"{gene.lower()}_rag_stack_manifest.json"
    pool_path = out_dir / f"{gene.lower()}_rag_benchmark_{CURRENT_TAG}_abstention_adjudication_pool.csv"
    manifest = {
        "manifest_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "gene": gene, "status": "sealed", "confirmatory": True, "seed": SEED,
        "variants": total, "mechanistic_passages_total": len(sheet),
        "passages_per_variant_max": PASSAGES_PER_VARIANT,
        "variants_with_mechanistic_evidence": int(sum(bool(first_by_variant.get(str(v))) for v in selected["mutation"])),
        "variants_without_mechanistic_evidence": len(abstention),
        "abstention_rows": len(abstention), "strata": list(STRATA), "sampling_design": design,
        "previous_variant_identities_excluded": sorted(_previous_variants(rag)),
        "development_filled_annotations_read": False,
        "thresholds_frozen_before_annotation": THRESHOLDS,
        "vocabularies_frozen_before_annotation": {
            "human": {
                "relevance": ["irrelevant", "background", "relevant", "directly_on_point"],
                "specificity": ["exact_variant", "same_residue", "functional_region", "general"],
                "entailment": ["supports", "neutral", "contradicts"],
                "support_type": ["structural_biophysical", "biochemical", "cell_functional", "in_vivo",
                                 "human_association", "computational", "context"],
                "quality": ["controlled", "uncontrolled", "unclear"], "conflict": ["yes", "no"],
            }, "abstention_judgment": ABSTENTION_VOCABULARY,
        },
        "rubric": {
            "mechanistic_evidence": "exact_variant, same_residue_analogy, or functional_region only",
            "context_reference": "general_context only; excluded from mechanistic Precision@5 and literature score",
            "no_evidence": "no_mechanistic_evidence_retrieved is an explicit outcome, never top-5 padding",
        },
        "full_sheet": str(sheet_path.relative_to(ROOT)),
        "full_sheet_template": str(template_path.relative_to(ROOT)),
        "abstention_sheet": str(abstention_path.relative_to(ROOT)),
        "abstention_sheet_template": str(abstention_template.relative_to(ROOT)),
        "blind_repeat_sheet": str(repeat_path.relative_to(ROOT)),
        "blind_repeat_sheet_template": str(repeat_template.relative_to(ROOT)),
        "blind_repeat_key": str(key_path.relative_to(ROOT)),
        "inference_artifact": str(inference_path.relative_to(ROOT)),
        "retrieval_stack_version": "rag_stack_v6",
        "retrieval_source": "completed_v6_caches_reused_without_retrieval_or_inference",
        "frozen_hashes": {
            "full_sheet_template": sha256(template_path),
            "abstention_sheet_template": sha256(abstention_template),
            "blind_repeat_template": sha256(repeat_template),
            "blind_repeat_key": sha256(key_path),
        "inference_artifact": sha256(inference_path),
        **({"abstention_pool": sha256(pool_path)} if pool_path.exists() else {}),
        "control_manifest": sha256(controls_path),
            "stack_manifest": sha256(stack_path),
            "index_manifest": sha256(index_manifest),
        },
        "promotion_status": "blocked_until_human_annotation_and_confirmatory_gate",
        "abstention_adjudication_pool": (
            str(pool_path.relative_to(ROOT)) if pool_path.exists() else None
        ),
    }
    manifest_path = out_dir / f"{gene.lower()}_rag_benchmark_{CURRENT_TAG}_confirmatory_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    parser.add_argument("--variants", type=int, default=BENCHMARK_VARIANTS)
    args = parser.parse_args()
    print(build(args.gene, args.variants))
