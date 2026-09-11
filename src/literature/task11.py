"""Auditable state and artefacts for the Task 11 retrieval rebuild.

This module deliberately keeps curation bookkeeping separate from retrieval
and inference.  It never fabricates human judgements: an incomplete blind
sheet produces a blocked gate, while the retrieved passages remain useful as
an auditable development artefact.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.literature.corpus_layers import INDEXED_LAYERS
from src.literature.evidence_grade import ClaimEvidence, grade_claim
from src.rag.ingestor import _documents_from_tei, _pmid

TASK11_VERSION = "task11_curation_v1"
RETRIEVAL_STACK_VERSION = "rag_stack_v6"


def task11_provisional_dir(output_dir: Path) -> Path:
    """Return the write-once workspace for unpromoted Task 11 outputs."""
    return output_dir / "provisional" / "task11"


def provisionalize_task11_outputs(root: Path, gene: str = "BRAF") -> Path:
    """Move the already-generated Task 11 outputs out of canonical paths.

    The move is explicit and hash-recorded.  It does not touch the RAF V2
    bundle or any structural-only outputs.
    """
    gene = gene.upper()
    output_dir = root / "data" / "output" / gene
    source_scores = output_dir / "scores"
    source_figures = output_dir / "figures" / "structural_ranking"
    provisional = task11_provisional_dir(output_dir)
    target_scores = provisional / "scores"
    target_figures = provisional / "figures"
    target_scores.mkdir(parents=True, exist_ok=True)
    target_figures.mkdir(parents=True, exist_ok=True)
    score_names = (
        f"{gene.lower()}_evidence_adjusted_variant_priority.csv",
        f"{gene.lower()}_evidence_adjusted_variant_priority.parquet",
        f"{gene.lower()}_variant_priority_rag_review_table.csv",
        f"{gene.lower()}_variant_priority_rag_review_table.parquet",
        f"{gene.lower()}_variant_priority_presentation_table.csv",
        f"{gene.lower()}_variant_priority_presentation_table.parquet",
        f"{gene.lower()}_discovery_rank_change_audit.csv",
        f"{gene.lower()}_discovery_rank_change_audit.parquet",
        f"{gene.lower()}_evidence_adjusted_variant_priority_summary.json",
        f"{gene.lower()}_post_curation_ranking_audit.csv",
    )
    figure_names = tuple(
        f"{gene.lower()}_{stem}.{extension}"
        for stem in (
            "evidence_adjusted_priority", "discovery_mechanistic_flag_map",
            "discovery_evidence_classes", "v600e_discovery_case",
        )
        for extension in ("pdf", "svg", "png")
    )
    records = []
    for source_dir, target_dir, names in (
        (source_scores, target_scores, score_names),
        (source_figures, target_figures, figure_names),
    ):
        for name in names:
            source = source_dir / name
            if not source.exists():
                continue
            target = target_dir / name
            source_hash = sha256(source)
            if target.exists():
                if sha256(target) != source_hash:
                    raise RuntimeError(f"provisionalization collision with different bytes: {target}")
                source.unlink()
                status = "already_provisional"
            else:
                shutil.move(str(source), str(target))
                status = "moved_to_provisional"
            records.append({
                "source": str(source.relative_to(root)),
                "target": str(target.relative_to(root)),
                "sha256": sha256(target),
                "status": status,
            })
    manifest = {
        "manifest_version": TASK11_VERSION,
        "created_utc": _now(),
        "gene": gene,
        "status": "provisionalized",
        "canonical_write_policy": "Task 11 writes provisional outputs; promotion is explicit",
        "files": records,
    }
    output = provisional / "provisionalization_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_directory_sha256(directory: Path) -> str:
    """Hash file names and file hashes in deterministic order."""
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def close_grobid_manifest(root: Path, gene: str = "BRAF") -> Path:
    """Close the GROBID inventory and account for primary papers with no chunks.

    The 14-paper count is computed against the curated primary layer, not by
    treating a successfully cached TEI as proof that it supplied an indexable
    chunk.  This catches primary records that have neither usable full text nor
    a non-empty abstract.
    """
    gene = gene.upper()
    evidence = root / "data" / "input" / gene / "evidence"
    rag = root / "data" / "output" / gene / "rag"
    pdf_dir = evidence / "pdfs"
    tei_dir = rag / "tei"
    layers_path = evidence / f"{gene.lower()}_corpus_layers.csv"
    layer_frame = pd.read_csv(layers_path).fillna("")
    layer_frame["pmid"] = layer_frame["pmid"].astype(str).str.strip().str.removesuffix(".0")
    layer_column = "corpus_layer" if "corpus_layer" in layer_frame else "layer"
    primary_pmids = set(layer_frame.loc[
        layer_frame[layer_column].eq("primary_evidence"), "pmid"
    ])

    # Account for every TEI independently.  An empty TEI is distinct from a
    # missing TEI and is retained as such in the manifest.
    tei_rows = []
    tei_pmids_with_chunks: set[str] = set()
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        pmid = _pmid(pdf.name)
        tei = tei_dir / f"{pdf.stem}.tei.xml"
        row = {
            "pmid": pmid,
            "pdf": str(pdf.relative_to(root)),
            "pdf_sha256": sha256(pdf),
            "tei": str(tei.relative_to(root)),
            "tei_present": tei.exists(),
            "tei_sha256": sha256(tei) if tei.exists() else None,
            "chunks": 0,
            "status": "tei_missing",
        }
        if tei.exists():
            try:
                chunks = _documents_from_tei(tei, {"pmid": pmid, "gene": gene})
                row["chunks"] = len(chunks)
                row["status"] = "chunks_available" if chunks else "no_chunks"
                if chunks:
                    tei_pmids_with_chunks.add(pmid)
            except Exception as error:  # the manifest must preserve a bad TEI
                row["status"] = f"parse_error:{type(error).__name__}"
                row["error"] = str(error)
        tei_rows.append(row)

    # The current index is the authoritative chunk inventory, because it also
    # contains abstracts.  Reading the two BM25 corpora is cheap and avoids
    # loading either MedCPT or Qdrant merely to count PMIDs.
    indexed_pmids: set[str] = set()
    indexed_chunks = 0
    for path in sorted(rag.glob("bm25_corpus_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            metadata = item.get("metadata", {})
            if metadata.get("corpus_layer") in INDEXED_LAYERS:
                indexed_pmids.add(str(metadata.get("pmid", "")))
                indexed_chunks += 1
    primary_without_chunks = sorted(primary_pmids - indexed_pmids)
    missing_primary_rows = layer_frame[
        layer_frame["pmid"].isin(primary_without_chunks)
    ][[column for column in ["pmid", "title", "doi", layer_column, "rule_id"]
      if column in layer_frame]].to_dict("records")

    manifest = {
        "manifest_version": TASK11_VERSION,
        "status": "closed",
        "closed_utc": _now(),
        "gene": gene,
        "pdf_count": len(tei_rows),
        "tei_count": sum(bool(row["tei_present"]) for row in tei_rows),
        "tei_with_chunks": sum(row["chunks"] > 0 for row in tei_rows),
        "tei_without_chunks": sum(row["chunks"] == 0 for row in tei_rows),
        "tei_parse_errors": sum(str(row["status"]).startswith("parse_error") for row in tei_rows),
        "indexed_chunk_count": indexed_chunks,
        "primary_evidence_articles": len(primary_pmids),
        "primary_evidence_articles_with_chunks": len(primary_pmids & indexed_pmids),
        "primary_evidence_articles_without_chunks": len(primary_without_chunks),
        "primary_evidence_articles_without_chunks_expected": 14,
        "primary_evidence_without_chunks": missing_primary_rows,
        "abstract_fallback_policy": "abstract only when no successfully parsed full text exists",
        "grobid_policy": "cached TEI is closed; no new extraction occurred in Task 11",
        "tei_files": tei_rows,
        "verification": {
            "count_matches_requested_inventory": len(primary_without_chunks) == 14,
            "all_cached_pdfs_have_tei": all(row["tei_present"] for row in tei_rows),
            "all_cached_tei_parse": all(
                not str(row["status"]).startswith("parse_error") for row in tei_rows
            ),
        },
    }
    output = rag / f"{gene.lower()}_grobid_extraction_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def archive_pre_curation_caches(root: Path, gene: str = "BRAF") -> Path:
    """Move old inference/retrieval caches to an immutable pre-curation area."""
    gene = gene.upper()
    rag = root / "data" / "output" / gene / "rag"
    archive = rag / "archive" / "pre_curation"
    archive.mkdir(parents=True, exist_ok=True)
    records = []
    for cache_name in ("inference_cache", "retrieval_cache"):
        source = rag / cache_name
        if not source.exists():
            continue
        destination = archive / cache_name
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            target = destination / path.name
            if target.exists() and sha256(target) != sha256(path):
                raise RuntimeError(f"archive collision with different bytes: {target}")
            if not target.exists():
                shutil.move(str(path), str(target))
            records.append({
                "cache_type": cache_name,
                "path": str(target.relative_to(root)),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "status": "pre_curation",
            })
    manifest = {
        "manifest_version": TASK11_VERSION,
        "created_utc": _now(),
        "gene": gene,
        "status": "archived_pre_curation",
        "cache_count": len(records),
        "cache_counts_by_type": {
            kind: sum(row["cache_type"] == kind for row in records)
            for kind in ("inference_cache", "retrieval_cache")
        },
        "files": records,
        "active_cache_directories_empty": all(
            not any((rag / name).iterdir()) if (rag / name).exists() else True
            for name in ("inference_cache", "retrieval_cache")
        ),
    }
    output = rag / "braf_rag_cache_archive_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def snapshot_pre_curation_rankings(root: Path, gene: str = "BRAF") -> Path:
    """Preserve old ranking outputs before the new inference overwrites them."""
    gene = gene.upper()
    scores = root / "data" / "output" / gene / "scores"
    destination = root / "data" / "output" / gene / "rag" / "archive" / "pre_curation" / "rankings"
    destination.mkdir(parents=True, exist_ok=True)
    names = (
        f"{gene.lower()}_evidence_adjusted_variant_priority.csv",
        f"{gene.lower()}_evidence_adjusted_variant_priority.parquet",
        f"{gene.lower()}_discovery_rank_change_audit.csv",
        f"{gene.lower()}_discovery_rank_change_audit.parquet",
    )
    records = []
    for name in names:
        source = scores / name
        if not source.exists():
            continue
        target = destination / name
        if not target.exists():
            shutil.copy2(source, target)
        records.append({"path": str(target.relative_to(root)), "bytes": target.stat().st_size,
                        "sha256": sha256(target), "status": "pre_curation"})
    output = destination / "braf_pre_curation_ranking_manifest.json"
    manifest = {"manifest_version": TASK11_VERSION, "created_utc": _now(),
                "gene": gene, "status": "archived_pre_curation", "files": records}
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def snapshot_pre_curation_benchmark(root: Path, gene: str = "BRAF") -> Path:
    """Preserve the old human sheet before sampling the new index."""
    gene = gene.upper()
    benchmark = root / "data" / "output" / gene / "rag" / "benchmark"
    destination = root / "data" / "output" / gene / "rag" / "archive" / "pre_curation" / "benchmark"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(benchmark.glob("*")):
        if not source.is_file():
            continue
        target = destination / source.name
        if not target.exists():
            shutil.copy2(source, target)
        records.append({"path": str(target.relative_to(root)), "bytes": target.stat().st_size,
                        "sha256": sha256(target), "status": "pre_curation"})
    output = destination / "braf_pre_curation_benchmark_manifest.json"
    output.write_text(json.dumps({"manifest_version": TASK11_VERSION, "created_utc": _now(),
                                  "gene": gene, "status": "archived_pre_curation",
                                  "files": records}, indent=2), encoding="utf-8")
    return output


def build_stack_manifest(root: Path, gene: str = "BRAF") -> Path:
    """Bind BM25, MedCPT and Qdrant to the final curated index fingerprint."""
    gene = gene.upper()
    rag = root / "data" / "output" / gene / "rag"
    index_manifest = rag / f"{gene.lower()}_rag_index_manifest.json"
    index = json.loads(index_manifest.read_text(encoding="utf-8"))
    layers = {}
    for layer, block in index.get("indexed_layers", {}).items():
        corpus = rag / f"bm25_corpus_{layer}.jsonl"
        layers[layer] = {
            "bm25_path": str(corpus.relative_to(root)),
            "bm25_sha256": sha256(corpus),
            "qdrant_collection": block["collection"],
            "qdrant_points": block["qdrant_points"],
            "papers": block["papers"],
        }
    stack = {
        "manifest_version": TASK11_VERSION,
        "retrieval_stack_version": RETRIEVAL_STACK_VERSION,
        "created_utc": _now(),
        "gene": gene,
        "index_manifest": str(index_manifest.relative_to(root)),
        "index_manifest_sha256": sha256(index_manifest),
        "index_fingerprint": sha256(index_manifest)[:16],
        "layers": layers,
        "embedding": {
            "document_encoder": index["document_encoder"],
            "query_encoder": index["query_encoder"],
            "dimension": index["embedding_dimension"],
        },
        "reranker": index["cross_encoder"],
        "fusion": "document-level RRF followed by MedCPT cross-encoder",
        "ranking_policy": {
            "channel_rank_normalization": "descending_rank_to_0_1_within_channel_and_method",
            "exact_channel_lexical_filter": True,
            "cross_encoder_query": "gene + exact_variant + mechanistic_question",
            "scope_order": ["exact_variant", "same_residue_analogy", "functional_region", "general_context"],
            "mechanistic_evidence_scopes": ["exact_variant", "same_residue_analogy", "functional_region"],
            "context_reference_scope": "general_context",
            "general_context_can_fill_mechanistic_top_k": False,
            "full_text_preferred": True,
        },
        "query_channels": [
            "variant_exact", "conformational_state", "structure_motif",
            "ppi_dimerization", "disease", "pharmacology",
        ],
        "diversity_policy": {
            "max_passages_per_pmid": 2,
            "minimum_distinct_pmids_when_three_available": 3,
            "global_pmid_cap_between_variants": None,
        },
        "zero_layer_leakage": True,
        "excluded_never_indexed": True,
        "grobid_manifest": index.get("grobid_extraction_manifest_sha256"),
    }
    stack["stack_fingerprint"] = hashlib.sha256(
        json.dumps(stack, sort_keys=True).encode()
    ).hexdigest()[:16]
    output = rag / "braf_rag_stack_manifest.json"
    output.write_text(json.dumps(stack, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def validate_task11_artifacts(root: Path, gene: str = "BRAF",
                              allow_canonical: bool = False) -> dict:
    """Validate that the active Task 11 outputs share one immutable stack.

    This is intentionally a read-only contract check.  It does not repair a
    stale cache or recompute a score; a mismatch stops finalization instead of
    silently mixing pre- and post-curation artefacts.
    """
    gene = gene.upper()
    rag = root / "data" / "output" / gene / "rag"
    scores = task11_provisional_dir(root / "data" / "output" / gene) / "scores"
    index_path = rag / f"{gene.lower()}_rag_index_manifest.json"
    stack_path = rag / f"{gene.lower()}_rag_stack_manifest.json"
    benchmark_path = rag / "benchmark" / f"{gene.lower()}_rag_benchmark_manifest.json"
    diversity_path = rag / "benchmark" / f"{gene.lower()}_rag_benchmark_diversity_controlled.csv"
    retrieval_path = rag / f"{gene.lower()}_retrieval_phase_manifest.json"
    grobid_path = rag / f"{gene.lower()}_grobid_extraction_manifest.json"
    archive_path = rag / f"{gene.lower()}_rag_cache_archive_manifest.json"
    provisionalization_path = (task11_provisional_dir(root / "data" / "output" / gene)
                               / "provisionalization_manifest.json")
    priority_path = scores / f"{gene.lower()}_evidence_adjusted_variant_priority.csv"
    required_paths = [index_path, stack_path, benchmark_path, diversity_path,
                      retrieval_path, grobid_path, archive_path,
                      provisionalization_path, priority_path]
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Task 11 artefacts missing: {missing}")

    index_sha = sha256(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    stack = json.loads(stack_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    diversity = pd.read_csv(diversity_path)
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    grobid = json.loads(grobid_path.read_text(encoding="utf-8"))
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    provisionalization = json.loads(provisionalization_path.read_text(encoding="utf-8"))
    expected_fp = index_sha[:16]
    expected_short_fp = index_sha[:12]
    errors = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(stack.get("index_manifest_sha256") == index_sha,
          "stack is not bound to the current index manifest")
    check(stack.get("index_fingerprint") == expected_fp,
          "stack index fingerprint is stale")
    check(stack.get("retrieval_stack_version") == RETRIEVAL_STACK_VERSION,
          "unexpected retrieval stack version")
    stack_fp = stack.get("stack_fingerprint")
    stack_sha = sha256(stack_path)
    check(bool(stack_fp), "retrieval stack fingerprint is absent")
    check(stack.get("zero_layer_leakage") is True and
          stack.get("excluded_never_indexed") is True,
          "layer isolation contract is not enabled")
    check(benchmark.get("index_manifest_sha256") == index_sha,
          "benchmark is not bound to the current index")
    check(benchmark.get("retrieval_stack_fingerprint") == stack_fp,
          "benchmark is not bound to the current retrieval stack")
    check(retrieval.get("index_fingerprint") == expected_short_fp,
          "retrieval phase is not bound to the current index")
    check(retrieval.get("retrieval_stack_fingerprint") == stack_fp,
          "retrieval phase is not bound to the current retrieval stack")
    check(retrieval.get("variants_total") == 171 and
          retrieval.get("retrieval_caches_available") == 171 and
          not retrieval.get("retrieval_failures"),
          "171-variant retrieval run is incomplete")
    check(len(list((rag / "retrieval_cache").glob("*.json"))) == 171,
          "active retrieval cache count is not 171")
    for cache_path in (rag / "retrieval_cache").glob("*.json"):
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        check(payload.get("index_fingerprint") == expected_short_fp,
              f"stale index fingerprint in {cache_path.name}")
        check(payload.get("stack_fingerprint") == stack_fp,
              f"stale stack fingerprint in {cache_path.name}")
        check(payload.get("stack_manifest_sha256") == stack_sha,
              f"stale stack manifest hash in {cache_path.name}")
    check(grobid.get("status") == "closed" and
          grobid.get("primary_evidence_articles_without_chunks") == 14,
          "GROBID manifest is not closed with the expected 14 primary records")
    check(archive.get("status") == "archived_pre_curation" and
          archive.get("cache_count") == 349,
          "pre-curation cache archive is incomplete")
    check(provisionalization.get("status") == "provisionalized",
          "Task 11 outputs are not marked provisional")
    canonical_priority = root / "data" / "output" / gene / "scores" / priority_path.name
    check(allow_canonical or not canonical_priority.exists(),
          "provisional Task 11 output is still present in the canonical namespace")
    check(benchmark.get("variants_sampled") == 24 and
          benchmark.get("passages_total") == 120 and
          benchmark.get("blind_repeats") == 24 and
          not benchmark.get("diversity_failures"),
          "blind benchmark structural inventory is incomplete")
    check(len(diversity) > 0 and
          diversity.get("panel", pd.Series(dtype=str)).astype(str).eq("diversity_controlled").all() and
          (diversity["pmid"].astype(str).value_counts().max() <= 10),
          "global diversity-controlled panel is incomplete")
    priority = pd.read_csv(priority_path)
    check(len(priority) == 171,
          "priority output does not contain the complete 171-variant run")
    for column, expected in {
        "retrieval_stack_version": RETRIEVAL_STACK_VERSION,
        "retrieval_index_fingerprint": expected_short_fp,
        "retrieval_stack_fingerprint": stack_fp,
        "literature_schema_version": "task11_claim_evidence_v1",
    }.items():
        check(column in priority and priority[column].astype(str).eq(str(expected)).all(),
              f"priority output is missing current {column}")
    for table_name in (
        f"{gene.lower()}_variant_priority_rag_review_table.csv",
        f"{gene.lower()}_variant_priority_presentation_table.csv",
    ):
        table = pd.read_csv(scores / table_name)
        check(len(table) == 171 and
              table["rag_promotion_status"].astype(str).eq("blocked").all() and
              table["claim_grades_provisional"].astype(bool).all() and
              (~table["eligible_for_v2"].astype(bool)).all(),
              f"provisional status contract failed in {table_name}")
    inference_cache_paths = sorted((rag / "inference_cache").glob("*.json"))
    check(len(inference_cache_paths) == 171,
          "inference cache count is not 171")
    for cache_path in inference_cache_paths:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        check(payload.get("retrieval_index_fingerprint") == expected_short_fp,
              f"stale index fingerprint in inference cache {cache_path.name}")
        check(payload.get("retrieval_stack_fingerprint") == stack_fp,
              f"stale stack fingerprint in inference cache {cache_path.name}")
        check(payload.get("retrieval_stack_manifest_sha256") == stack_sha,
              f"stale stack manifest hash in inference cache {cache_path.name}")
    if errors:
        raise RuntimeError("Task 11 artefact validation failed: " + "; ".join(errors))
    return {
        "status": "passed",
        "index_manifest_sha256": index_sha,
        "index_fingerprint": expected_fp,
        "retrieval_stack_fingerprint": stack_fp,
        "checks": {
            "index_stack_match": True,
            "benchmark_stack_match": True,
            "retrieval_stack_match": True,
            "retrieval_cache_count": 171,
            "inference_cache_count": 171,
            "grobid_primary_without_chunks": 14,
            "pre_curation_cache_count": 349,
            "benchmark_passages": 120,
            "benchmark_blind_repeats": 24,
            "production_top5_global_max_pmid": int(
                pd.read_csv(rag / "benchmark" / f"{gene.lower()}_rag_benchmark_sheet.csv")
                ["pmid"].astype(str).value_counts().max()
            ),
            "diversity_controlled_passages": len(diversity),
            "diversity_controlled_global_max_pmid": int(
                diversity["pmid"].astype(str).value_counts().max()
            ),
            "priority_rows_checked": 171,
        },
    }


def ranking_change_audit(root: Path, gene: str = "BRAF") -> Path:
    """Compare the preserved pre-curation order with the new order."""
    gene = gene.upper()
    scores = task11_provisional_dir(root / "data" / "output" / gene) / "scores"
    archive = root / "data" / "output" / gene / "rag" / "archive" / "pre_curation" / "rankings"
    old = archive / f"{gene.lower()}_evidence_adjusted_variant_priority.csv"
    new = scores / f"{gene.lower()}_evidence_adjusted_variant_priority.csv"
    output = scores / f"{gene.lower()}_post_curation_ranking_audit.csv"
    if not old.exists() or not new.exists():
        pd.DataFrame(columns=["mutation", "rank_pre_curation", "rank_post_curation",
                              "rank_delta_post_minus_pre"]).to_csv(output, index=False)
        return output
    old_frame = pd.read_csv(old)
    new_frame = pd.read_csv(new)
    old_rank = old_frame[["mutation", "variant_priority_rank"]].rename(
        columns={"variant_priority_rank": "rank_pre_curation"}
    )
    new_rank = new_frame[["mutation", "variant_priority_rank"]].rename(
        columns={"variant_priority_rank": "rank_post_curation"}
    )
    audit = old_rank.merge(new_rank, on="mutation", how="outer", validate="one_to_one")
    audit["rank_delta_post_minus_pre"] = (
        pd.to_numeric(audit["rank_post_curation"], errors="coerce") -
        pd.to_numeric(audit["rank_pre_curation"], errors="coerce")
    )
    audit["pre_curation_sha256"] = sha256(old)
    audit["post_curation_sha256"] = sha256(new)
    audit.to_csv(output, index=False)
    return output


def _support_type(hypothesis: dict, passages: list[dict]) -> str:
    """Conservatively infer the evidence modality from the cited text."""
    text = " ".join(str(item.get("passage", "")) for item in passages).lower()
    if any(term in text for term in ("mouse", "mice", "xenograft", "in vivo", "animal")):
        return "in_vivo"
    if any(term in text for term in ("cell proliferation", "cell growth", "signaling",
                                     "tumor growth", "transformation", "viability")):
        return "cell_functional"
    if any(term in text for term in ("kinase activity", "phosphorylation", "biochemical",
                                     "enzyme activity")):
        return "biochemical"
    if any(term in text for term in ("structure", "conformation", "dimer", "interface",
                                     "molecular dynamics")):
        return "structural_biophysical"
    return "computational"


def _claim_dimensions(hypothesis: dict, passages: list[dict]) -> dict:
    primary = [item for item in passages if item.get("corpus_layer") == "primary_evidence"]
    pmids = list(dict.fromkeys(str(item.get("pmid")) for item in primary))
    joined = " ".join(str(item.get("passage", "")) for item in primary).lower()
    controls = "yes" if any(term in joined for term in (
        "control", "vehicle", "wild type", "wild-type", "untreated",
    )) else "unknown"
    orthogonal_terms = (
        ("kinase", "phosphorylation"), ("proliferation", "growth"),
        ("binding", "activity"), ("structure", "functional"),
    )
    orthogonal = sum(all(term in joined for term in pair) for pair in orthogonal_terms)
    full_text = bool(primary) and all(bool(item.get("full_text_available")) for item in primary)
    dimensions = []
    if full_text:
        dimensions = ["variant_identity", "assay_modality", "controls", "direction", "source_context"]
    return {
        "primary_evidence_pmids": pmids,
        "context_reference_pmids": list(dict.fromkeys(
            str(item.get("pmid")) for item in passages
            if item.get("corpus_layer") == "context_reference"
        )),
        "full_text_verified": full_text,
        "controls_reported": controls,
        "orthogonal_assays": int(orthogonal),
        "independent_papers": len(pmids),
        "verifiable_full_text_dimensions": dimensions,
    }


def build_claim_evidence_artifact(root: Path, gene: str = "BRAF") -> Path:
    """Convert every current hypothesis into a provisional ClaimEvidence grade."""
    gene = gene.upper()
    rag = root / "data" / "output" / gene / "rag"
    scores = task11_provisional_dir(root / "data" / "output" / gene) / "scores"
    table = pd.read_parquet(scores / f"{gene.lower()}_evidence_adjusted_variant_priority.parquet")
    records = []
    for row in table.to_dict("records"):
        variant = str(row["mutation"])
        cache_files = sorted((rag / "inference_cache").glob(f"{variant}__*.json"))
        if not cache_files:
            continue
        result = json.loads(cache_files[-1].read_text(encoding="utf-8"))
        evidence = result.get("evidence", [])
        for number, hypothesis in enumerate(result.get("inference", {}).get("hypotheses", []), 1):
            cited_ids = [str(value) for value in hypothesis.get("cited_evidence_ids", [])]
            cited = [item for item in evidence if str(item.get("evidence_id")) in cited_ids]
            directness = str(row.get("literature_mechanism_specificity", "none"))
            if directness not in {"exact_variant", "same_residue", "functional_region", "general"}:
                directness = "general"
            dimensions = _claim_dimensions(hypothesis, cited)
            claim = ClaimEvidence(
                claim_id=f"{variant}:H{number:02d}", variant=variant,
                claim_text=(f"{hypothesis.get('mechanism', 'unknown mechanism')}: "
                            f"{hypothesis.get('rationale', '')}"),
                variant_directness=directness,
                support_type=_support_type(hypothesis, cited),
                evidence_ids=cited_ids,
                generation_mode=str(hypothesis.get("generation_mode", "llm_validated")),
                **dimensions,
            )
            candidate_grade = grade_claim(claim)
            records.append({
                "claim": claim.model_dump(),
                "candidate_grade": candidate_grade.model_dump(),
                "candidate_grade_name": candidate_grade.grade,
                "candidate_label": candidate_grade.label,
                "effective_grade": "unreviewed",
                "effective_label": "provisional_candidate",
                "grade_provisional": True,
                "full_text_dimensions_verifiable": bool(
                    claim.full_text_verified and claim.verifiable_full_text_dimensions
                ),
                "source_cache": str(cache_files[-1].relative_to(root)),
            })
        disease = result.get("inference", {}).get("disease_association", {})
        disease_ids = [str(value) for value in disease.get("cited_evidence_ids", [])]
        if disease.get("status", "none") != "none" and disease_ids:
            cited = [item for item in evidence
                     if str(item.get("evidence_id")) in set(disease_ids)]
            dimensions = _claim_dimensions(disease, cited)
            disease_claim = ClaimEvidence(
                claim_id=f"{variant}:D01", variant=variant,
                claim_text=(f"disease association ({disease.get('status')}): "
                            f"{disease.get('rationale', '')}"),
                variant_directness="exact_variant",
                support_type=("human_association" if disease.get("status") == "associated"
                              else "context"),
                evidence_ids=disease_ids,
                generation_mode=str(disease.get("generation_mode", "llm_validated")),
                **dimensions,
            )
            candidate_grade = grade_claim(disease_claim)
            records.append({
                "claim": disease_claim.model_dump(),
                "candidate_grade": candidate_grade.model_dump(),
                "candidate_grade_name": candidate_grade.grade,
                "candidate_label": candidate_grade.label,
                "effective_grade": "unreviewed",
                "effective_label": "provisional_candidate",
                "grade_provisional": True,
                "full_text_dimensions_verifiable": bool(
                    disease_claim.full_text_verified and disease_claim.verifiable_full_text_dimensions
                ),
                "source_cache": str(cache_files[-1].relative_to(root)),
            })
    output = rag / "braf_claim_evidence.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "manifest_version": TASK11_VERSION,
        "created_utc": _now(),
        "gene": gene,
        "claims": len(records),
        "automatic_grade_counts": pd.Series(
            [record["candidate_grade_name"] for record in records]
        ).value_counts().to_dict() if records else {},
        "candidate_grade_counts": pd.Series(
            [record["candidate_grade_name"] for record in records]
        ).value_counts().to_dict() if records else {},
        "effective_grade": "unreviewed",
        "effective_label": "provisional_candidate",
        "benchmark_does_not_validate_individual_claims": True,
        "generation_mode_counts": pd.Series(
            [record["claim"]["generation_mode"] for record in records]
        ).value_counts().to_dict() if records else {},
        "all_grades_provisional": True,
        "a_b_require_full_text_dimensions": True,
        "primary_layer_only_for_direct_claims": True,
        "artifact": str(output.relative_to(root)),
        "sha256": sha256(output),
    }
    (rag / "braf_claim_evidence_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output
