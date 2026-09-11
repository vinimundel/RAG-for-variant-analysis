"""Single-use, hash-locked finalization for the V7 mechanistic benchmark."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.literature.benchmark import CONFIRMATORY_VOCABULARIES, HUMAN_REQUIRED_COLUMNS, cohen_kappa
from src.literature.task11 import sha256


BENCHMARK_TAG = "v7"
RESULT_VERSION = "task11f3_confirmatory_result_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_marker(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"single-use marker already exists: {path.name}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _validate_vocab(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    errors = {}
    for column in columns:
        values = frame[column].fillna("").astype(str).str.strip()
        allowed = CONFIRMATORY_VOCABULARIES[column]
        bad = sorted(set(values) - set(allowed))
        if values.eq("").any() or bad:
            errors[column] = {"blank": int(values.eq("").sum()), "unknown": bad}
    if errors:
        raise ValueError(json.dumps({"source": source, "vocabulary_errors": errors}, ensure_ascii=False))


def _unchanged(template: pd.DataFrame, filled: pd.DataFrame, human: set[str], source: str) -> None:
    if list(template.columns) != list(filled.columns):
        raise ValueError(f"{source}: columns differ from frozen template")
    for column in template.columns:
        if column in human:
            continue
        if not template[column].fillna("<NA>").astype(str).equals(
            filled[column].fillna("<NA>").astype(str)
        ):
            raise ValueError(f"{source}: non-annotatable column changed: {column}")


def _automatic_scope_for_human(value: str) -> str:
    return {"same_residue": "same_residue_analogy"}.get(str(value), str(value))


def _validate_conditional_repeats(repeats: pd.DataFrame) -> dict[str, int]:
    """Validate only the rubric fields applicable to each repeat domain."""
    allowed_statuses = {"mechanistic_evidence_retrieved", "no_mechanistic_evidence_retrieved"}
    statuses = repeats["mechanistic_evidence_status"].fillna("").astype(str)
    if not statuses.isin(allowed_statuses).all():
        raise ValueError("blind repeats contain an invalid evidence status")
    mechanistic = repeats.loc[statuses.eq("mechanistic_evidence_retrieved")]
    abstention = repeats.loc[statuses.eq("no_mechanistic_evidence_retrieved")]
    if len(mechanistic):
        _validate_vocab(mechanistic, list(HUMAN_REQUIRED_COLUMNS), "mechanistic blind repeats")
        if mechanistic["abstention_judgment"].fillna("").astype(str).str.strip().ne("").any():
            raise ValueError("mechanistic repeat contains abstention labels")
    if len(abstention):
        values = abstention["abstention_judgment"].fillna("").astype(str).str.strip()
        if values.eq("").any() or not values.isin({"correct_abstention", "missed_evidence"}).all():
            raise ValueError("abstention repeat has an invalid or blank abstention_judgment")
        for column in HUMAN_REQUIRED_COLUMNS:
            if abstention[column].fillna("").astype(str).str.strip().ne("").any():
                raise ValueError("abstention repeat contains mechanistic rubric labels")
    return {"mechanistic_repeats": len(mechanistic), "abstention_repeats": len(abstention)}


def finalize_v7(root: Path, gene: str = "BRAF") -> Path:
    gene = gene.upper()
    out_dir = root / "data" / "output" / gene / "rag" / f"benchmark_{BENCHMARK_TAG}_confirmatory"
    manifest_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "sealed":
        raise RuntimeError("V7 benchmark is not sealed or was already completed")
    frozen = manifest.get("frozen_hashes", {})
    controls_path = root / "data" / "output" / gene / "rag" / \
        f"{gene.lower()}_task11f2_v6_control_manifest.json"
    if not controls_path.exists():
        raise RuntimeError("V6 controls are missing")
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if controls.get("status") != "passed":
        raise RuntimeError("V6 controls are not passed")
    template = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_sheet_template.csv"
    abstention_template = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_abstention_sheet_template.csv"
    repeat_template = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_blind_repeats_template.csv"
    key_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_repeat_key.json"
    inference_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_inference.json"
    pool_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_abstention_adjudication_pool.csv"
    stack_path = root / "data" / "output" / gene / "rag" / f"{gene.lower()}_rag_stack_manifest.json"
    index_path = root / "data" / "output" / gene / "rag" / f"{gene.lower()}_rag_index_manifest.json"
    current_hashes = {
        "full_sheet_template": sha256(template),
        "abstention_sheet_template": sha256(abstention_template),
        "blind_repeat_template": sha256(repeat_template),
        "blind_repeat_key": sha256(key_path),
        "inference_artifact": sha256(inference_path),
        "control_manifest": sha256(controls_path),
        "stack_manifest": sha256(stack_path),
        "index_manifest": sha256(index_path),
    }
    if "abstention_pool" in frozen:
        current_hashes["abstention_pool"] = sha256(pool_path)
    mismatches = {
        name: {"frozen": frozen.get(name), "current": value}
        for name, value in current_hashes.items() if frozen.get(name) != value
    }
    if mismatches:
        raise RuntimeError(json.dumps({"frozen_artifact_hash_mismatch": mismatches}, indent=2))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if key.get("status") != "sealed":
        raise RuntimeError("V7 repeat key is not sealed")

    marker = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_evaluation_opened.json"
    opening = {
        "status": "opened", "created_utc": _now(), "gene": gene,
        "benchmark_manifest_sha256": sha256(manifest_path),
        "frozen_hashes_verified": current_hashes,
        "labels_read_after_marker": True,
        "retrieval_recalculated": False, "inference_recalculated": False,
    }
    # The marker is created before either filled label file is read.
    _atomic_marker(marker, opening)

    sheet_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_sheet_filled.csv"
    abstention_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_abstention_sheet_filled.csv"
    repeat_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_blind_repeats_filled.csv"
    if not sheet_path.exists() or not abstention_path.exists() or not repeat_path.exists():
        raise FileNotFoundError("V7 filled annotation files are missing; opening remains recorded")

    sheet = pd.read_csv(sheet_path)
    abstention = pd.read_csv(abstention_path)
    repeats = pd.read_csv(repeat_path)
    sheet_template = pd.read_csv(template)
    abstention_template_frame = pd.read_csv(abstention_template)
    repeat_template_frame = pd.read_csv(repeat_template)
    human = set(HUMAN_REQUIRED_COLUMNS)
    _unchanged(sheet_template, sheet, human, "mechanistic sheet")
    _unchanged(abstention_template_frame, abstention, {"abstention_judgment"}, "abstention sheet")
    _unchanged(repeat_template_frame, repeats, human | {"abstention_judgment"}, "blind repeats")
    _validate_vocab(sheet, list(HUMAN_REQUIRED_COLUMNS), "mechanistic sheet")
    repeat_counts = _validate_conditional_repeats(repeats)
    if set(abstention["abstention_judgment"].astype(str)) - {"correct_abstention", "missed_evidence"} \
            or abstention["abstention_judgment"].astype(str).eq("").any():
        raise ValueError("abstention_judgment has an invalid or blank value")
    if not sheet["automatic_evidence_scope"].isin(
        ["exact_variant", "same_residue_analogy", "functional_region"]
    ).all():
        raise ValueError("mechanistic sheet contains non-mechanistic scope")
    if sheet["system_citation_key"].astype(str).eq("").any() or not sheet["system_citation_key"].is_unique:
        raise ValueError("system citation keys are incomplete or duplicated")
    if not sheet["cited_in_claim"].astype(str).isin({"yes", "no"}).all():
        raise ValueError("invalid system citation flag")

    total_variants = int(manifest["variants"])
    variants = set(sheet["mutation"].astype(str)) | set(abstention["mutation"].astype(str))
    if len(variants) != total_variants:
        raise ValueError("V7 does not cover all sampled variants")
    if not sheet.groupby("mutation").size().le(5).all():
        raise ValueError("more than five mechanistic passages for a variant")
    if len(repeats) != total_variants:
        raise ValueError("blind repeat count differs from sampled variants")

    relevant = sheet["relevance"].astype(str).isin({"relevant", "directly_on_point"})
    per_variant_rows = []
    for mutation, group in sheet.groupby("mutation", sort=True):
        group_relevant = group["relevance"].astype(str).isin({"relevant", "directly_on_point"})
        per_variant_rows.append({
            "mutation": mutation, "stratum": str(group["stratum"].iloc[0]),
            "mechanistic_passages_returned": len(group),
            "mechanistic_precision_at_5": float(group_relevant.mean()) if len(group) else None,
            "has_human_mechanistic_evidence": bool(group_relevant.any()),
            "system_mechanistic_evidence_returned": True,
        })
    for mutation, group in abstention.groupby("mutation", sort=True):
        per_variant_rows.append({
            "mutation": mutation, "stratum": str(group["stratum"].iloc[0]),
            "mechanistic_passages_returned": 0,
            "mechanistic_precision_at_5": None,
            "has_human_mechanistic_evidence": str(group["abstention_judgment"].iloc[0]) == "missed_evidence",
            "system_mechanistic_evidence_returned": False,
        })
    per_variant = pd.DataFrame(per_variant_rows).sort_values("mutation")
    per_variant_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_per_variant_metrics.csv"
    per_variant.to_csv(per_variant_path, index=False)
    per_stratum = per_variant.groupby("stratum", as_index=False).agg(
        variants=("mutation", "count"),
        mechanistic_precision_at_5=("mechanistic_precision_at_5", "mean"),
        variant_evidence_coverage=("system_mechanistic_evidence_returned", "mean"),
        human_confirmed_mechanistic_evidence=("has_human_mechanistic_evidence", "mean"),
    )
    per_stratum_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_per_stratum_metrics.csv"
    per_stratum.to_csv(per_stratum_path, index=False)

    cited = sheet["cited_in_claim"].astype(str).eq("yes")
    citation_precision = float(sheet.loc[cited, "entailment"].astype(str).eq("supports").mean()) if cited.any() else None
    exact = sheet.loc[cited & sheet["specificity"].astype(str).eq("exact_variant")]
    exact_precision = float(exact["entailment"].astype(str).eq("supports").mean()) if len(exact) else None
    human_scope = sheet["specificity"].astype(str).map(_automatic_scope_for_human)
    scope_precision = float(human_scope.eq(sheet["automatic_evidence_scope"].astype(str)).mean()) if len(sheet) else None
    mechanistic_precision = float(relevant.mean()) if len(sheet) else None
    # Coverage is a retrieval property: a missed passage discovered during
    # adjudication does not retroactively count as retrieved coverage.
    coverage = float(per_variant["system_mechanistic_evidence_returned"].mean()) if len(per_variant) else None
    correct_abstention = float(
        abstention["abstention_judgment"].astype(str).eq("correct_abstention").mean()
    ) if len(abstention) else None
    mapping = pd.DataFrame(key["mapping"])
    original = pd.concat([
        sheet[["item_id", "relevance"]].rename(columns={"item_id": "original_item_id", "relevance": "relevance_original"}),
        abstention[["item_id", "abstention_judgment"]].rename(columns={"item_id": "original_item_id", "abstention_judgment": "abstention_original"}),
    ], ignore_index=True)
    repeat_labels = repeats.rename(columns={
        "relevance": "relevance_repeat",
        "abstention_judgment": "abstention_repeat",
    })[["repeat_item_id", "relevance_repeat", "abstention_repeat"]]
    joined = mapping.merge(original, on="original_item_id", validate="one_to_one").merge(
        repeat_labels, on="repeat_item_id", validate="one_to_one", suffixes=("_original", "_repeat")
    )
    relevance_rows = joined.loc[joined["annotation_domain"].eq("mechanistic")].dropna(
        subset=["relevance_original", "relevance_repeat"]
    )
    abstention_rows = joined.loc[joined["annotation_domain"].eq("abstention")].dropna(
        subset=["abstention_original", "abstention_repeat"]
    )
    relevance_kappa = cohen_kappa(
        relevance_rows["relevance_original"].tolist(), relevance_rows["relevance_repeat"].tolist()
    )
    abstention_kappa = cohen_kappa(
        abstention_rows["abstention_original"].tolist(), abstention_rows["abstention_repeat"].tolist()
    )
    thresholds = manifest["thresholds_frozen_before_annotation"]
    metrics = {
        "mechanistic_precision_at_5": mechanistic_precision,
        "variant_evidence_coverage": coverage,
        "correct_abstention_rate": correct_abstention,
        "citation_precision": citation_precision,
        "exact_claim_precision": exact_precision,
        "scope_precision": scope_precision,
        "relevance_kappa": float(relevance_kappa),
        "abstention_kappa": float(abstention_kappa),
    }
    passed = all([
        metrics["mechanistic_precision_at_5"] is not None and metrics["mechanistic_precision_at_5"] >= thresholds["mechanistic_precision_at_5"],
        metrics["variant_evidence_coverage"] is not None and metrics["variant_evidence_coverage"] >= thresholds["variant_evidence_coverage"],
        metrics["correct_abstention_rate"] is not None and metrics["correct_abstention_rate"] >= thresholds["correct_abstention_rate"],
        metrics["citation_precision"] == thresholds["citation_precision"],
        metrics["exact_claim_precision"] == thresholds["exact_claim_precision"],
        metrics["scope_precision"] == thresholds["scope_precision"],
        metrics["relevance_kappa"] >= thresholds["relevance_kappa"],
        metrics["abstention_kappa"] >= thresholds["abstention_kappa"],
    ])
    result = {
        "manifest_version": RESULT_VERSION, "created_utc": _now(),
        "gene": gene, "status": "passed" if passed else "negative",
        "promotion_allowed": bool(passed), "metrics": metrics, "thresholds": thresholds,
        "per_variant_metrics": str(per_variant_path.relative_to(root)),
        "per_stratum_metrics": str(per_stratum_path.relative_to(root)),
        "repeat_validation_counts": repeat_counts,
        "input_sha256": {"filled_sheet": sha256(sheet_path), "filled_abstention": sha256(abstention_path),
                         "filled_repeats": sha256(repeat_path), **current_hashes},
        "recalculated_retrieval": False, "recalculated_inference": False,
    }
    result_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    completed_marker = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_evaluation_completed.json"
    _atomic_marker(completed_marker, {
        "status": "completed", "created_utc": _now(), "gene": gene,
        "result_sha256": sha256(result_path), "promotion_allowed": bool(passed),
    })
    manifest["status"] = "completed"
    manifest["completed_result"] = str(result_path.relative_to(root))
    manifest["completed_result_sha256"] = sha256(result_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    gate_path = root / "data" / "output" / gene / "rag" / f"{gene.lower()}_task11_gate_manifest.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    gate["confirmatory_v7"] = {**result, "result_sha256": sha256(result_path)}
    gate["status"] = "passed" if passed else "blocked"
    gate["rag_fields_promotable_to_v2"] = bool(passed)
    gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_path


def preflight_v7(root: Path, gene: str = "BRAF") -> Path:
    """Read-only validation of the sealed V7 finalizer inputs."""
    gene = gene.upper()
    out_dir = root / "data" / "output" / gene / "rag" / f"benchmark_{BENCHMARK_TAG}_confirmatory"
    manifest_path = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "sealed":
        raise RuntimeError("V7 benchmark is not sealed")
    controls_path = root / "data" / "output" / gene / "rag" / \
        f"{gene.lower()}_task11f2_v6_control_manifest.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if controls.get("status") != "passed":
        raise RuntimeError("V6 controls are not passed")
    stack_path = root / "data" / "output" / gene / "rag" / f"{gene.lower()}_rag_stack_manifest.json"
    index_path = root / "data" / "output" / gene / "rag" / f"{gene.lower()}_rag_index_manifest.json"
    paths = {
        "full_sheet_template": out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_sheet_template.csv",
        "abstention_sheet_template": out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_abstention_sheet_template.csv",
        "blind_repeat_template": out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_blind_repeats_template.csv",
        "blind_repeat_key": out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_repeat_key.json",
        "inference_artifact": out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_confirmatory_inference.json",
        "control_manifest": controls_path, "stack_manifest": stack_path, "index_manifest": index_path,
    }
    if "abstention_pool" in manifest.get("frozen_hashes", {}):
        paths["abstention_pool"] = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_abstention_adjudication_pool.csv"
    current = {name: sha256(path) for name, path in paths.items()}
    if current != manifest.get("frozen_hashes", {}):
        raise RuntimeError(json.dumps({"frozen_artifact_hash_mismatch": {
            name: {"frozen": manifest.get("frozen_hashes", {}).get(name), "current": value}
            for name, value in current.items()
            if manifest.get("frozen_hashes", {}).get(name) != value
        }}, indent=2))
    key = json.loads(paths["blind_repeat_key"].read_text(encoding="utf-8"))
    if key.get("status") != "sealed":
        raise RuntimeError("V7 repeat key is not sealed")
    markers = [out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_evaluation_opened.json",
               out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_evaluation_completed.json"]
    if any(path.exists() for path in markers):
        raise RuntimeError("V6 finalizer marker already exists")
    output = out_dir / f"{gene.lower()}_rag_benchmark_{BENCHMARK_TAG}_preflight.json"
    output.write_text(json.dumps({
        "status": "sealed_preflight_passed", "created_utc": _now(), "gene": gene,
        "hashes_verified": current, "labels_read": False,
        "retrieval_recalculated": False, "inference_recalculated": False,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
