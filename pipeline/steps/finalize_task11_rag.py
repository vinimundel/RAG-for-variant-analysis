"""Finalize Task 11 bookkeeping after the new retrieval/inference run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.config import ROOT, paths_for
from src.literature.benchmark import benchmark_gate
from src.literature.task11 import (
    TASK11_VERSION, build_claim_evidence_artifact, ranking_change_audit, sha256,
    validate_task11_artifacts,
)


def run(gene: str = "BRAF") -> Path:
    gene = gene.upper()
    paths = paths_for(gene)
    rag = paths["rag"]
    benchmark = rag / "benchmark"
    validation = validate_task11_artifacts(ROOT, gene)
    sheet = pd.read_csv(benchmark / f"{gene.lower()}_rag_benchmark_sheet.csv")
    repeats = pd.read_csv(benchmark / f"{gene.lower()}_rag_benchmark_blind_repeats.csv")
    key_path = benchmark / f"{gene.lower()}_rag_benchmark_repeat_key.csv"
    key = pd.read_csv(key_path) if key_path.exists() else None
    gate = benchmark_gate(sheet, repeats, key)
    claims = build_claim_evidence_artifact(ROOT, gene)
    ranking = ranking_change_audit(ROOT, gene)
    index = rag / f"{gene.lower()}_rag_index_manifest.json"
    stack = rag / f"{gene.lower()}_rag_stack_manifest.json"
    exploratory = rag / f"{gene.lower()}_task11_exploratory_annotations_manifest.json"
    confirmatory = rag / "benchmark_v2_confirmatory" / f"{gene.lower()}_rag_benchmark_v2_confirmatory_manifest.json"
    gate_manifest = {
        "manifest_version": TASK11_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gene": gene,
        "status": "passed" if gate["promotion_allowed"] else "blocked",
        "rag_fields_promotable_to_v2": bool(gate["promotion_allowed"]),
        "benchmark": gate,
        "index_manifest": str(index.relative_to(ROOT)),
        "index_manifest_sha256": sha256(index),
        "stack_manifest": str(stack.relative_to(ROOT)),
        "stack_manifest_sha256": sha256(stack) if stack.exists() else None,
        "claims_artifact": str(claims.relative_to(ROOT)),
        "claims_artifact_sha256": sha256(claims),
        "ranking_change_audit": str(ranking.relative_to(ROOT)),
        "ranking_change_audit_sha256": sha256(ranking),
        "legacy_literature_columns_replaced": True,
        "legacy_merge_policy": "replace_entire_literature_output_from_current_claims",
        "failure_policy": "benchmark_failure_blocks_RAG_fields_in_V2_outputs",
        "exploratory_filled_annotations": {
            "manifest": str(exploratory.relative_to(ROOT)) if exploratory.exists() else None,
            "manifest_sha256": sha256(exploratory) if exploratory.exists() else None,
            "classification": "exploratory_development",
            "used_by_official_gate": False,
        },
        "confirmatory_benchmark_v2": {
            "manifest": str(confirmatory.relative_to(ROOT)) if confirmatory.exists() else None,
            "manifest_sha256": sha256(confirmatory) if confirmatory.exists() else None,
            "status": "separate_and_blocked",
        },
        "active_artifact_validation": validation,
    }
    output = rag / f"{gene.lower()}_task11_gate_manifest.json"
    output.write_text(json.dumps(gate_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(run(args.gene))
