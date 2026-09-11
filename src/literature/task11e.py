"""Bookkeeping and development artefacts for the Task 11E retrieval revision."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.literature.task11 import sha256


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def retire_confirmatory_benchmark(root: Path, gene: str = "BRAF") -> Path:
    """Archive the unannotated V2 confirmatory set without deleting bytes."""
    gene = gene.upper()
    source_dir = root / "data" / "output" / gene / "rag" / "benchmark_v2_confirmatory"
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    archive_dir = root / "data" / "output" / gene / "rag" / "archive" / \
        "task11_confirmatory_v2_superseded_before_annotation"
    archive_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(source_dir.iterdir()):
        if not source.is_file() or source.name == "retirement_manifest.json":
            continue
        digest = sha256(source)
        target = archive_dir / source.name
        if target.exists() and sha256(target) != digest:
            raise RuntimeError(f"confirmatory archive collision: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        records.append({
            "source": str(source.relative_to(root)),
            "archive": str(target.relative_to(root)),
            "sha256": digest, "bytes": source.stat().st_size,
        })
    original_manifest = source_dir / f"{gene.lower()}_rag_benchmark_v2_confirmatory_manifest.json"
    original_manifest_sha256 = sha256(original_manifest) if original_manifest.exists() else None
    if original_manifest.exists():
        payload = json.loads(original_manifest.read_text(encoding="utf-8"))
        payload.update({
            "status": "superseded_before_annotation",
            "reason": "generated from unchanged retrieval caches",
            "superseded_manifest_sha256": original_manifest_sha256,
            "replacement": "rag_stack_v5_confirmatory_pending",
        })
        original_manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    retirement = {
        "manifest_version": "task11e1_confirmatory_retirement_v1",
        "created_utc": _now(), "gene": gene,
        "status": "superseded_before_annotation",
        "reason": "generated from unchanged retrieval caches",
        "original_manifest_sha256_before_update": original_manifest_sha256,
        "files": records,
        "human_annotations_read": False,
        "replacement_required": "rag_stack_v5_confirmatory_pending",
    }
    output = source_dir / "retirement_manifest.json"
    output.write_text(json.dumps(retirement, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def retire_v5_confirmatory_benchmark(root: Path, gene: str = "BRAF") -> Path:
    """Preserve the unannotated V5 set before context separation."""
    gene = gene.upper()
    source_dir = root / "data" / "output" / gene / "rag" / "benchmark_v5_confirmatory"
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    archive_dir = root / "data" / "output" / gene / "rag" / "archive" / \
        "task11_confirmatory_v5_superseded_before_annotation"
    archive_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(source_dir.iterdir()):
        if not source.is_file() or source.name == "retirement_manifest.json":
            continue
        digest = sha256(source)
        target = archive_dir / source.name
        if target.exists() and sha256(target) != digest:
            raise RuntimeError(f"V5 confirmatory archive collision: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        records.append({"source": str(source.relative_to(root)),
                        "archive": str(target.relative_to(root)),
                        "sha256": digest, "bytes": source.stat().st_size})
    manifest_path = source_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_manifest.json"
    original_hash = sha256(manifest_path) if manifest_path.exists() else None
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload.update({
            "status": "superseded_before_annotation",
            "reason": "context passages included in mechanistic Precision@5 denominator",
            "superseded_manifest_sha256": original_hash,
            "replacement": "rag_stack_v6_confirmatory_pending",
        })
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    retirement = {
        "manifest_version": "task11f2_confirmatory_retirement_v1", "created_utc": _now(),
        "gene": gene, "status": "superseded_before_annotation",
        "reason": "context passages included in mechanistic Precision@5 denominator",
        "original_manifest_sha256_before_update": original_hash, "files": records,
        "human_annotations_read": False,
        "replacement_required": "rag_stack_v6_confirmatory_pending",
    }
    output = source_dir / "retirement_manifest.json"
    output.write_text(json.dumps(retirement, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def retire_v6_confirmatory_benchmark(root: Path, gene: str = "BRAF") -> Path:
    """Preserve V6 before correcting coverage and conditional repeat logic."""
    gene = gene.upper()
    source_dir = root / "data" / "output" / gene / "rag" / "benchmark_v6_confirmatory"
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    archive_dir = root / "data" / "output" / gene / "rag" / "archive" / \
        "task11_confirmatory_v6_superseded_before_annotation"
    archive_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(source_dir.iterdir()):
        if not source.is_file() or source.name == "retirement_manifest.json":
            continue
        digest = sha256(source)
        target = archive_dir / source.name
        if target.exists() and sha256(target) != digest:
            raise RuntimeError(f"V6 confirmatory archive collision: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        records.append({"source": str(source.relative_to(root)),
                        "archive": str(target.relative_to(root)),
                        "sha256": digest, "bytes": source.stat().st_size})
    manifest_path = source_dir / f"{gene.lower()}_rag_benchmark_v6_confirmatory_manifest.json"
    original_hash = sha256(manifest_path) if manifest_path.exists() else None
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload.update({
            "status": "superseded_before_annotation",
            "reason": "coverage counted adjudicated abstentions and repeats were not conditionally validated",
            "superseded_manifest_sha256": original_hash,
            "replacement": "rag_stack_v6_confirmatory_v7_pending",
        })
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    retirement = {
        "manifest_version": "task11f3_confirmatory_retirement_v1", "created_utc": _now(),
        "gene": gene, "status": "superseded_before_annotation",
        "reason": "coverage counted adjudicated abstentions and repeats were not conditionally validated",
        "original_manifest_sha256_before_update": original_hash, "files": records,
        "human_annotations_read": False,
        "replacement_required": "rag_stack_v6_confirmatory_v7_pending",
    }
    output = source_dir / "retirement_manifest.json"
    output.write_text(json.dumps(retirement, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def build_development_training_table(root: Path, gene: str = "BRAF") -> Path:
    """Join the first 120 labels to old retrieval scores for development only."""
    gene = gene.upper()
    rag = root / "data" / "output" / gene / "rag"
    benchmark = rag / "benchmark"
    source = benchmark / "diagnostic" / f"{gene.lower()}_rag_benchmark_filled_normalized.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    annotations = pd.read_csv(source)
    rows = []
    for retrieval_path in sorted((rag / "retrieval_cache").glob("*.json")):
        payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
        variant = str(payload.get("variant", ""))
        subset = annotations.loc[annotations["mutation"].astype(str).eq(variant)]
        if subset.empty:
            continue
        channels_by_pmid = payload.get("retrieval_diagnostics", {}).get("channels_per_selected_document", {})
        for item in payload.get("evidence", []):
            match = subset.loc[
                subset["pmid"].astype(str).eq(str(item.get("pmid", "")))
                & subset["passage"].astype(str).eq(str(item.get("passage", "")))
            ]
            if match.empty:
                # The benchmark passage can be a serialized equivalent of the
                # cache passage; PMID and rank remain a safe fallback key.
                match = subset.loc[subset["pmid"].astype(str).eq(str(item.get("pmid", "")))]
            if match.empty:
                continue
            label = match.sort_values("rank").iloc[0]
            rows.append({
                "variant": variant, "passage_id": str(label["item_id"]),
                "pmid": str(item.get("pmid", "")),
                "channel": ";".join(channels_by_pmid.get(str(item.get("pmid", "")), [])),
                "bm25_score": item.get("bm25_score"),
                "dense_score": item.get("retrieval_score"),
                "cross_encoder_score": item.get("reranker_score"),
                "retrieval_rank": int(label.get("rank", 0)),
                "human_relevance": label.get("relevance"),
                "human_specificity": label.get("specificity"),
                "human_entailment": label.get("entailment"),
                "cited_in_claim": label.get("cited_in_claim"),
                "generation_mode": "llm_validated" if payload.get("llm_contract_status") == "passed" else "deterministic_fallback",
                "retrieval_stack_version": "rag_stack_v4",
            })
    output_dir = rag / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "braf_task11e_development_training_table.csv"
    pd.DataFrame(rows).sort_values(["variant", "retrieval_rank", "pmid"]).to_csv(output, index=False)
    manifest = {
        "manifest_version": "task11e_development_table_v1", "created_utc": _now(),
        "gene": gene, "classification": "exploratory_development",
        "source_annotations": str(source.relative_to(root)), "source_annotations_sha256": sha256(source),
        "retrieval_caches": "data/output/BRAF/rag/retrieval_cache",
        "rows": len(rows), "variants": int(annotations["mutation"].nunique()),
        "confirmatory_eligible": False, "kappa_eligible": False,
        "purpose": "diagnostic retrieval tuning only; never a production result",
        "columns": list(pd.DataFrame(rows).columns) if rows else [],
    }
    manifest_path = output.with_name("braf_task11e_development_training_table_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def archive_active_stack_v4(root: Path, gene: str = "BRAF") -> Path:
    """Move active v4 caches/manifest aside before a v5 run can start."""
    gene = gene.upper()
    rag = root / "data" / "output" / gene / "rag"
    archive = rag / "archive" / "task11_stack_v4"
    archive.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("retrieval_cache", "inference_cache"):
        source_dir = rag / name
        target_dir = archive / name
        target_dir.mkdir(parents=True, exist_ok=True)
        if source_dir.exists():
            for source in sorted(source_dir.glob("*.json")):
                target = target_dir / source.name
                digest = sha256(source)
                if target.exists() and sha256(target) != digest:
                    raise RuntimeError(f"v4 cache archive collision: {target}")
                if not target.exists():
                    shutil.move(str(source), str(target))
                records.append({"type": name, "path": str(target.relative_to(root)),
                                "sha256": digest, "bytes": target.stat().st_size})
        source_dir.mkdir(parents=True, exist_ok=True)
    stack = rag / f"{gene.lower()}_rag_stack_manifest.json"
    stack_archive = archive / stack.name
    stack_hash = sha256(stack) if stack.exists() else None
    if stack.exists():
        if stack_archive.exists() and sha256(stack_archive) != stack_hash:
            raise RuntimeError(f"v4 stack archive collision: {stack_archive}")
        if not stack_archive.exists():
            shutil.copy2(stack, stack_archive)
    manifest = {
        "manifest_version": "task11e_v4_archive_v1", "created_utc": _now(),
        "gene": gene, "status": "superseded_before_v5", "stack_version": "rag_stack_v4",
        "stack_manifest_sha256": stack_hash, "files": records,
    }
    output = archive / "archive_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def archive_active_stack_v5(root: Path, gene: str = "BRAF") -> Path:
    """Preserve V5 retrieval/inference caches before V6 cache emission."""
    gene = gene.upper()
    rag = root / "data" / "output" / gene / "rag"
    archive = rag / "archive" / "task11_stack_v5_context_separation"
    archive.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("retrieval_cache", "inference_cache"):
        source_dir = rag / name
        target_dir = archive / name
        target_dir.mkdir(parents=True, exist_ok=True)
        if source_dir.exists():
            for source in sorted(source_dir.glob("*.json")):
                target = target_dir / source.name
                digest = sha256(source)
                if target.exists() and sha256(target) != digest:
                    raise RuntimeError(f"v5 cache archive collision: {target}")
                if not target.exists():
                    shutil.move(str(source), str(target))
                records.append({"type": name, "path": str(target.relative_to(root)),
                                "sha256": digest, "bytes": target.stat().st_size})
        source_dir.mkdir(parents=True, exist_ok=True)
    stack = rag / f"{gene.lower()}_rag_stack_manifest.json"
    stack_archive = archive / stack.name
    stack_hash = sha256(stack) if stack.exists() else None
    if stack.exists():
        if stack_archive.exists() and sha256(stack_archive) != stack_hash:
            raise RuntimeError(f"v5 stack archive collision: {stack_archive}")
        if not stack_archive.exists():
            shutil.copy2(stack, stack_archive)
    manifest = {
        "manifest_version": "task11f2_v5_archive_v1", "created_utc": _now(),
        "gene": gene, "status": "superseded_before_v6",
        "stack_version": "rag_stack_v5", "stack_manifest_sha256": stack_hash,
        "files": records,
    }
    output = archive / "archive_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
