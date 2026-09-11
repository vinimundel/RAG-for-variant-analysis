"""Generate the blinded annotation sheet for the human evaluation of the RAG.

Retrieval runs through the production discovery path against the curated
``primary_evidence`` layer, so the sheet measures the retrieval that actually
produces claims, not a re-implementation of it.

Two files are written and must stay apart: the sheet the reviewer annotates,
which carries only opaque item ids, and the key, which maps repeats back to
their originals.  Reading the key before annotating destroys the washout.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import hashlib

from pipeline.config import ROOT, get_gene_config, paths_for
from pipeline.steps.run_literature_reranking import (
    _discovery_request, _retrieval_cache_path, resolve_retrieval_layer,
)
from src.literature.benchmark import (
    BENCHMARK_VARIANTS, PASSAGES_PER_VARIANT, blind_repeat_plan, corpus_mentions,
    motif_positions, sample_variants,
)
from src.rag.bm25_index import bm25_corpus_path
from src.rag.chain import EVIDENCE_LAYER, QUERY_CHANNELS, DiscoveryLiteratureAnalyzer
from src.rag.schemas import LiteratureEvidence
from src.literature.task11 import RETRIEVAL_STACK_VERSION, task11_provisional_dir, sha256


def primary_corpus_passages(rag_dir: Path, layer: str | None) -> list[str]:
    """Every chunk of the layer claims are drawn from, for stratification."""
    path = bm25_corpus_path(rag_dir, layer)
    if not path.exists():
        raise FileNotFoundError(f"Lexical corpus absent, build the index first: {path}")
    return [json.loads(line)["page_content"]
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

# Columns the reviewer fills in. Empty on purpose: a pre-filled sheet anchors.
HUMAN_ANNOTATION_COLUMNS = {
    "relevance": "irrelevant | background | relevant | directly_on_point",
    "specificity": "exact_variant | same_residue | functional_region | general",
    "entailment": "supports | neutral | contradicts",
    "support_type": "structural_biophysical | biochemical | cell_functional | in_vivo | "
                    "human_association | computational | context",
    "quality": "controlled | uncontrolled | unclear",
    "conflict": "yes | no",
    "reviewer_note": "",
}
SYSTEM_ANNOTATION_COLUMNS = {
    "cited_in_claim": "yes | no (system-generated)",
    "system_citation_key": "system-generated opaque key",
}
ANNOTATION_COLUMNS = {**HUMAN_ANNOTATION_COLUMNS, **SYSTEM_ANNOTATION_COLUMNS}


def _system_citation_keys(paths, variant: str) -> set[tuple[str, str]]:
    """Return PMID/passage pairs cited by the current inference cache."""
    cache_files = sorted((paths["rag"] / "inference_cache").glob(f"{variant}__*.json"))
    if not cache_files:
        return set()
    payload = json.loads(cache_files[-1].read_text(encoding="utf-8"))
    result = payload.get("inference", {})
    cited_ids = set()
    for hypothesis in result.get("hypotheses", []):
        cited_ids.update(str(value) for value in hypothesis.get("cited_evidence_ids", []))
    cited_ids.update(str(value) for value in
                     result.get("disease_association", {}).get("cited_evidence_ids", []))
    return {
        (str(item.get("pmid")), str(item.get("passage", "")))
        for item in payload.get("evidence", [])
        if str(item.get("evidence_id")) in cited_ids
    }


def _citation_key(variant: str, pmid: str, passage: str) -> str:
    value = f"{variant}|{pmid}|{passage}".encode()
    return f"SC-{hashlib.sha256(value).hexdigest()[:16]}"


def _diversity_controlled_panel(rows: list[dict], max_per_pmid: int = 10) -> list[dict]:
    """Take one candidate per variant/rank round under a global PMID cap."""
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(str(row["mutation"]), []).append(row)
    for values in by_variant.values():
        values.sort(key=lambda item: int(item["rank"]))
    selected, used = [], {}
    target = max((len(values) for values in by_variant.values()), default=0)
    variants = sorted(by_variant)
    for rank in range(target):
        for variant in variants:
            values = by_variant[variant]
            for row in values:
                if row in selected or int(row["rank"]) != rank + 1:
                    continue
                pmid = str(row["pmid"])
                if used.get(pmid, 0) >= max_per_pmid:
                    continue
                selected.append({**row, "panel": "diversity_controlled"})
                used[pmid] = used.get(pmid, 0) + 1
                break
    return selected


def build(gene: str, seed: str = "braf_rag_benchmark_v1",
          variants: int = BENCHMARK_VARIANTS, k: int = PASSAGES_PER_VARIANT) -> Path:
    gene = gene.upper()
    cfg, paths = get_gene_config(gene), paths_for(gene)
    provisional = task11_provisional_dir(paths["output"])
    priority_path = provisional / "scores" / f"{gene.lower()}_evidence_adjusted_variant_priority.parquet"
    if not priority_path.exists():
        raise FileNotFoundError(f"Run the discovery pipeline first: {priority_path}")
    index_manifest = paths["rag"] / f"{gene.lower()}_rag_index_manifest.json"
    layer = resolve_retrieval_layer(index_manifest)

    frame = pd.read_parquet(priority_path)
    # Strata come from what the curated corpus actually mentions, not from a
    # previous LLM pass over a corpus that has since been re-layered.
    by_position = corpus_mentions(primary_corpus_passages(paths["rag"], layer))
    selected, design = sample_variants(
        frame, by_position, motif_positions(cfg.get("functional_motifs", {})),
        seed=seed, total=variants,
    )
    analyzer = DiscoveryLiteratureAnalyzer(gene, candidate_k=30, rag_dir=paths["rag"],
                                           layer=layer)
    condition_profile = str(cfg["discovery_profile"])
    query_version = "discovery_v4"
    index_fingerprint = hashlib.sha256(index_manifest.read_bytes()).hexdigest()[:12]

    rows, candidate_rows, diagnostics, item_index = [], [], [], 0
    for record in selected.to_dict("records"):
        request = _discovery_request(record, gene, condition_profile, (), query_version)
        # Channels stay separate through fusion so contamination is auditable.
        # The benchmark exercises every declared route independently.  The
        # production inference path may still opt out of pharmacology when it
        # is not requested, but the blind retrieval benchmark measures the
        # complete Task 11 channel contract.
        queries = analyzer._channel_queries(request, channels=QUERY_CHANNELS)
        # Retrieve the top-10 candidate pool once.  The first five passages are
        # the production panel; the same stored pool feeds the global diversity
        # panel without a second MedCPT/Qdrant pass.
        retrieval_cache = _retrieval_cache_path(
            paths["rag"] / "retrieval_cache", str(record["mutation"]), condition_profile,
            (), query_version, index_fingerprint,
        )
        if retrieval_cache.exists():
            cache_payload = json.loads(retrieval_cache.read_text(encoding="utf-8"))
            evidence = [LiteratureEvidence.model_validate(item)
                        for item in cache_payload["evidence"][:max(k, 10)]]
        else:
            evidence = analyzer.retrieve_discovery(
                queries, max(k, 10), variant=str(record["mutation"]),
                functional_region=str(record["functional_region"]),
                reranker_query=(f"{gene} {record['mutation']} mechanistic question: "
                                "Does the evidence directly support a mechanistic effect?"),
            )
        diagnostics.append({"mutation": str(record["mutation"]), "stratum": record["stratum"],
                            **analyzer.last_retrieval_diagnostics})
        cited_pairs = _system_citation_keys(paths, str(record["mutation"]))
        for rank, item in enumerate(evidence, 1):
            item_index += 1
            system_key = _citation_key(str(record["mutation"]), item.pmid, item.passage)
            candidate = {
                "item_id": f"I{item_index:03d}", "mutation": str(record["mutation"]),
                "stratum": record["stratum"], "rank": rank,
                "pmid": item.pmid, "article_title": item.article_title,
                "section": item.section, "corpus_layer": item.corpus_layer,
                "reranker_score": item.reranker_score, "passage": item.passage,
                "panel": "production_top5" if rank <= k else "candidate_top10",
                "cited_in_claim": "yes" if (item.pmid, item.passage) in cited_pairs else "no",
                "system_citation_key": system_key,
                **{column: "" for column in HUMAN_ANNOTATION_COLUMNS},
            }
            candidate_rows.append(candidate)
            if rank <= k:
                rows.append(candidate)
        print(f"Benchmark retrieval {len(rows)}/{variants * k}: "
              f"{record['mutation']} -> {len(evidence)} passages", flush=True)
    analyzer.release_retrieval_models()

    sheet = pd.DataFrame(rows)
    repeats = blind_repeat_plan(sheet["item_id"].tolist(), seed=seed)
    repeat_sheet = (repeats.merge(sheet, left_on="original_item_id", right_on="item_id")
                    .drop(columns=["item_id", "original_item_id"])
                    .rename(columns={"repeat_item_id": "item_id"}))
    # The repeat block carries no mapping identifiers or rank.  The passage
    # itself may still contain a PMID/variant, so this is mapping-blinded, not
    # semantically content-blinded.
    repeat_sheet = repeat_sheet.drop(columns=[
        "mutation", "pmid", "article_title", "section", "stratum", "rank",
        "corpus_layer", "reranker_score", "panel",
    ])

    diversity_rows = _diversity_controlled_panel(candidate_rows, max_per_pmid=10)
    diversity_sheet = pd.DataFrame(diversity_rows)

    output_dir = paths["output"] / "rag" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / f"{gene.lower()}_rag_benchmark_sheet.csv"
    sheet.to_csv(sheet_path, index=False)
    diversity_path = output_dir / f"{gene.lower()}_rag_benchmark_diversity_controlled.csv"
    diversity_sheet.to_csv(diversity_path, index=False)
    repeat_sheet.to_csv(output_dir / f"{gene.lower()}_rag_benchmark_blind_repeats.csv", index=False)
    repeats.to_csv(output_dir / f"{gene.lower()}_rag_benchmark_repeat_key.csv", index=False)
    (output_dir / f"{gene.lower()}_rag_benchmark_retrieval_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
    )

    sizes = [item["document_size_bias"] for item in diagnostics
             if item.get("document_size_bias") is not None]
    concentration = pd.Series([row["pmid"] for row in rows])
    global_counts = concentration.astype(str).value_counts() if len(concentration) else pd.Series(dtype=int)
    diversity_counts = (diversity_sheet["pmid"].astype(str).value_counts()
                        if len(diversity_sheet) else pd.Series(dtype=int))
    index_fingerprint = hashlib.sha256(index_manifest.read_bytes()).hexdigest()[:16]
    stack_manifest = paths["rag"] / f"{gene.lower()}_rag_stack_manifest.json"
    stack_data = json.loads(stack_manifest.read_text(encoding="utf-8")) if stack_manifest.exists() else {}
    stack_fingerprint = stack_data.get("stack_fingerprint")
    diversity_failures = [item for item in diagnostics
                          if item.get("selected_documents", 0) >= 3
                          and item.get("distinct_pmids", 0) < 3]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "gene": gene, "seed": seed,
        "retrieval_corpus_layer": layer or "unlayered_legacy_index",
        "retrieval_version": analyzer.retrieval_version,
        "index_manifest": str(index_manifest.relative_to(ROOT)),
        "index_manifest_sha256": hashlib.sha256(index_manifest.read_bytes()).hexdigest(),
        "index_fingerprint": index_fingerprint,
        "retrieval_stack_version": RETRIEVAL_STACK_VERSION,
        "retrieval_stack_manifest": (str(stack_manifest.relative_to(ROOT))
                                     if stack_manifest.exists() else None),
        "retrieval_stack_manifest_sha256": (
            hashlib.sha256(stack_manifest.read_bytes()).hexdigest()
            if stack_manifest.exists() else None
        ),
        "retrieval_stack_fingerprint": stack_fingerprint,
        "variants_sampled": int(selected["mutation"].nunique()),
        "stratum_counts": {str(k_): int(v) for k_, v in selected["stratum"].value_counts().items()},
        "sampling_design": design,
        "stratification_source": (
            f"lexical variant mentions in the {layer or 'unlayered'} corpus, "
            "plus configured functional motifs"
        ),
        "passages_per_variant": k, "passages_total": int(len(sheet)),
        "distinct_pmids": int(concentration.nunique()),
        "production_top5_max_passages_per_pmid_per_variant": int(max(
            (item.get("max_passages_from_one_document", 0) for item in diagnostics),
            default=0,
        )),
        "production_top5_max_passages_per_pmid_global": int(global_counts.max()) if len(global_counts) else 0,
        "diversity_controlled_max_passages_per_pmid_global": int(diversity_counts.max()) if len(diversity_counts) else 0,
        "diversity_controlled_passages_total": int(len(diversity_sheet)),
        "diversity_controlled_distinct_pmids": int(diversity_sheet["pmid"].nunique()) if len(diversity_sheet) else 0,
        "diversity_controlled_global_pmid_cap": 10,
        "benchmark_panels": {
            "production_top5": {
                "path": str(sheet_path.relative_to(ROOT)), "passages": int(len(sheet)),
                "human_annotation_panel": True,
            },
            "diversity_controlled": {
                "path": str(diversity_path.relative_to(ROOT)), "passages": int(len(diversity_sheet)),
                "human_annotation_panel": False,
                "source": "stored_candidate_top10",
                "global_pmid_cap": 10,
            },
        },
        "minimum_distinct_pmids_when_three_available": 3,
        "diversity_failures": diversity_failures,
        "mean_document_size_bias": round(sum(sizes) / len(sizes), 3) if sizes else None,
        "retrieval_diagnostics_file":
            f"{gene.lower()}_rag_benchmark_retrieval_diagnostics.json",
        "blind_repeats": int(len(repeats)),
        "judgements_total": int(len(sheet) + len(repeats)),
        "annotation_columns": ANNOTATION_COLUMNS,
        "human_annotation_columns": list(HUMAN_ANNOTATION_COLUMNS),
        "required_human_annotation_columns": [
            "relevance", "specificity", "entailment", "support_type", "quality", "conflict",
        ],
        "optional_human_annotation_columns": ["reviewer_note"],
        "system_generated_columns": list(SYSTEM_ANNOTATION_COLUMNS),
        "citation_flag_policy": "cited_in_claim is generated from the active inference cache; reviewer does not edit it",
        "blinding_mode": "mapping_blinded",
        "blinding_limitations": [
            "repeat rows omit mutation, PMID, title, section and rank",
            "the passage text can itself contain a variant or PMID, so semantic content blinding is not guaranteed",
        ],
        "washout_requirement_days": 14,
        "reviewer_count": 1,
        "declared_limitations": [
            "A single reviewer cannot establish the complete set of relevant passages, "
            "so recall is not estimated and no exhaustive-recall claim is made.",
            "Strata come from the previous retrieval run and are a sampling prior, "
            "not ground truth.",
            "Agreement is intra-reviewer across the washout; it measures rubric "
            "stability, not agreement between people.",
        ],
        "sheet": str(sheet_path.relative_to(ROOT)),
        "diversity_controlled_sheet": str(diversity_path.relative_to(ROOT)),
        "human_annotations_status": "unannotated_rebuild",
        "promotion_gate": "blocked_until_human_annotations_and_all_gates_pass",
    }
    (output_dir / f"{gene.lower()}_rag_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return sheet_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--seed", default="braf_rag_benchmark_v1")
    parser.add_argument("--variants", type=int, default=BENCHMARK_VARIANTS)
    parser.add_argument("--passages", type=int, default=PASSAGES_PER_VARIANT)
    args = parser.parse_args()
    print(build(args.gene, args.seed, args.variants, args.passages))
