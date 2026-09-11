"""Explicitly promote a passed Task 11 result into canonical outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.config import ROOT, paths_for
from src.literature.task11 import sha256, task11_provisional_dir, validate_task11_artifacts


TABULAR_NAMES = (
    "braf_evidence_adjusted_variant_priority.csv",
    "braf_evidence_adjusted_variant_priority.parquet",
    "braf_variant_priority_rag_review_table.csv",
    "braf_variant_priority_rag_review_table.parquet",
    "braf_variant_priority_presentation_table.csv",
    "braf_variant_priority_presentation_table.parquet",
    "braf_discovery_rank_change_audit.csv",
    "braf_discovery_rank_change_audit.parquet",
)
OTHER_NAMES = (
    "braf_evidence_adjusted_variant_priority_summary.json",
    "braf_post_curation_ranking_audit.csv",
)
FIGURE_NAMES = tuple(
    f"braf_{stem}.{extension}"
    for stem in (
        "evidence_adjusted_priority", "discovery_mechanistic_flag_map",
        "discovery_evidence_classes", "v600e_discovery_case",
    )
    for extension in ("pdf", "svg", "png")
)


def _copy_idempotent(source: Path, target: Path) -> dict:
    if not source.exists():
        raise FileNotFoundError(source)
    source_hash = sha256(source)
    if target.exists() and sha256(target) != source_hash:
        raise RuntimeError(f"promotion would overwrite different bytes: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        status = "copied"
    else:
        status = "already_promoted_same_hash"
    return {"source": str(source.relative_to(ROOT)), "target": str(target.relative_to(ROOT)),
            "sha256": sha256(target), "status": status}


def _promote_table(source: Path, target: Path) -> dict:
    """Copy a table and make its production eligibility explicit."""
    frame = pd.read_csv(source) if source.suffix == ".csv" else pd.read_parquet(source)
    frame["rag_promotion_status"] = "promoted"
    frame["claim_grades_provisional"] = True
    frame["eligible_for_v2"] = True
    if target.exists():
        existing = pd.read_csv(target) if target.suffix == ".csv" else pd.read_parquet(target)
        if not existing.equals(frame):
            raise RuntimeError(f"promotion would overwrite different table: {target}")
        status = "already_promoted_same_content"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".csv":
            frame.to_csv(target, index=False)
        else:
            frame.to_parquet(target, index=False)
        status = "promoted_with_status_columns"
    return {"source": str(source.relative_to(ROOT)), "target": str(target.relative_to(ROOT)),
            "sha256": sha256(target), "status": status}


def _promote_summary(source: Path, target: Path) -> dict:
    summary = json.loads(source.read_text(encoding="utf-8"))
    summary.update({
        "rag_promotion_status": "promoted",
        "claim_grades_provisional": True,
        "eligible_for_v2": True,
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"source": str(source.relative_to(ROOT)), "target": str(target.relative_to(ROOT)),
            "sha256": sha256(target), "status": "promoted_with_status_fields"}


def run(gene: str = "BRAF") -> Path:
    gene = gene.upper()
    if gene != "BRAF":
        raise ValueError("Task 11 promotion is currently restricted to BRAF")
    paths = paths_for(gene)
    rag = paths["rag"]
    provisional = task11_provisional_dir(paths["output"])
    gate_path = rag / f"{gene.lower()}_task11_gate_manifest.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "passed" or gate.get("rag_fields_promotable_to_v2") is not True:
        raise RuntimeError("Task 11 promotion denied: benchmark gate has not passed")
    validation = validate_task11_artifacts(ROOT, gene, allow_canonical=True)

    records = []
    for name in TABULAR_NAMES:
        records.append(_promote_table(provisional / "scores" / name, paths["scores"] / name))
    records.append(_promote_summary(
        provisional / "scores" / OTHER_NAMES[0], paths["scores"] / OTHER_NAMES[0]
    ))
    records.append(_copy_idempotent(
        provisional / "scores" / OTHER_NAMES[1], paths["scores"] / OTHER_NAMES[1]
    ))
    for name in FIGURE_NAMES:
        records.append(_copy_idempotent(provisional / "figures" / name, paths["figures"] / name))

    output = rag / "braf_task11_promotion_manifest.json"
    manifest = {
        "manifest_version": "task11_curation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gene": gene,
        "status": "promoted",
        "promotion_is_explicit": True,
        "gate_manifest": str(gate_path.relative_to(ROOT)),
        "gate_manifest_sha256": sha256(gate_path),
        "pre_promotion_validation": validation,
        "claim_grades_provisional": True,
        "eligible_for_v2": True,
        "files": records,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(run(args.gene))
