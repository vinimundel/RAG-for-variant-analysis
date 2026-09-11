"""Single-use finalization of the sealed Task 11F v5 benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.literature.benchmark import (
    CONFIRMATORY_VOCABULARIES, HUMAN_REQUIRED_COLUMNS, citation_accuracy,
    cohen_kappa, ndcg_at_k, precision_at_k, reciprocal_rank, retrieval_metrics,
)
from src.literature.task11 import sha256


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_open_marker(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, ensure_ascii=False).encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("Task 11F v5 evaluation opening already exists; single-use consumed") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _assert_vocab(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    errors = {}
    for column in columns:
        values = frame[column].fillna("").astype(str).str.strip()
        allowed = CONFIRMATORY_VOCABULARIES[column]
        bad = sorted(set(values) - set(allowed))
        if values.eq("").any() or bad:
            errors[column] = {"blank": int(values.eq("").sum()), "unknown": bad}
    if errors:
        raise ValueError(json.dumps({"source": source, "vocabulary_errors": errors}, ensure_ascii=False))


def _assert_nonannotatable_unchanged(template: pd.DataFrame, filled: pd.DataFrame,
                                     source: str, human_columns: list[str]) -> None:
    if list(template.columns) != list(filled.columns):
        raise ValueError(f"{source}: columns differ from frozen template")
    for column in template.columns:
        if column in human_columns:
            continue
        left = template[column].fillna("<NA>").astype(str)
        right = filled[column].fillna("<NA>").astype(str)
        if not left.equals(right):
            raise ValueError(f"{source}: non-annotatable column changed: {column}")


def finalize_v5(root: Path, gene: str = "BRAF") -> Path:
    gene = gene.upper()
    out_dir = root / "data" / "output" / gene / "rag" / "benchmark_v5_confirmatory"
    manifest_path = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_annotation":
        raise RuntimeError("v5 benchmark is not in frozen_before_annotation state")
    control_path = root / "data" / "output" / gene / "rag" / f"{gene.lower()}_task11e_v5_control_manifest.json"
    if not control_path.exists():
        raise RuntimeError("v5 controls have not passed")
    controls = json.loads(control_path.read_text(encoding="utf-8"))
    if controls.get("status") != "passed":
        raise RuntimeError("v5 controls are blocked")

    filled_sheet = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_sheet_filled.csv"
    filled_repeats = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_blind_repeats_filled.csv"
    template_sheet = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_sheet_template.csv"
    template_repeats = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_blind_repeats_template.csv"
    key_path = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_repeat_key.json"
    marker = out_dir / f"{gene.lower()}_rag_benchmark_v5_evaluation_opened.json"
    opening = {
        "status": "opened", "created_utc": _now(), "gene": gene,
        "benchmark_manifest_sha256": sha256(manifest_path),
        "control_manifest_sha256": sha256(control_path),
        "template_sheet_sha256": sha256(template_sheet),
        "template_repeats_sha256": sha256(template_repeats),
        "repeat_key_sha256": sha256(key_path),
        "labels_read_after_marker": True,
    }
    # This is intentionally before pandas reads either label file.
    _atomic_open_marker(marker, opening)
    if not filled_sheet.exists() or not filled_repeats.exists():
        raise FileNotFoundError("v5 filled annotation files are missing; single-use opening is retained")

    sheet = pd.read_csv(filled_sheet)
    repeats = pd.read_csv(filled_repeats)
    template = pd.read_csv(template_sheet)
    repeat_template = pd.read_csv(template_repeats)
    _assert_nonannotatable_unchanged(template, sheet, "main sheet", list(HUMAN_REQUIRED_COLUMNS))
    _assert_nonannotatable_unchanged(repeat_template, repeats, "blind repeats", list(HUMAN_REQUIRED_COLUMNS))
    _assert_vocab(sheet, list(HUMAN_REQUIRED_COLUMNS), "main sheet")
    _assert_vocab(repeats, list(HUMAN_REQUIRED_COLUMNS), "blind repeats")
    if len(sheet) != 120 or len(repeats) != 24:
        raise ValueError("v5 benchmark dimensions are not 120 + 24")
    if not sheet.groupby("mutation").size().eq(5).all():
        raise ValueError("v5 main sheet does not contain five passages per variant")
    if not sheet.groupby("mutation")["pmid"].apply(lambda values: values.astype(str).value_counts().max() <= 2).all():
        raise ValueError("more than two passages from one PMID for a variant")
    if not sheet["cited_in_claim"].astype(str).isin({"yes", "no"}).all():
        raise ValueError("system citation flag has an invalid value")
    if sheet["system_citation_key"].astype(str).eq("").any() or not sheet["system_citation_key"].is_unique:
        raise ValueError("system citation keys are incomplete or duplicated")

    per_variant = retrieval_metrics(sheet)
    per_variant["mrr_at_5"] = sheet.groupby("mutation", sort=True).apply(
        lambda group: reciprocal_rank([
            {"irrelevant": 0, "background": 1, "relevant": 2, "directly_on_point": 3}[value]
            for value in group.sort_values("rank")["relevance"]
        ]), include_groups=False
    ).to_numpy()
    citation = citation_accuracy(sheet)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    mapping = pd.DataFrame(key["mapping"])
    joined = mapping.merge(sheet[["item_id", *HUMAN_REQUIRED_COLUMNS]],
                           left_on="original_item_id", right_on="item_id", validate="one_to_one") \
        .merge(repeats[["repeat_item_id", *HUMAN_REQUIRED_COLUMNS]],
               on="repeat_item_id", validate="one_to_one", suffixes=("_original", "_repeat"))
    kappa = cohen_kappa(joined["relevance_repeat"].tolist(), joined["relevance_original"].tolist())
    thresholds = manifest["thresholds_frozen_before_annotation"]
    metrics = {
        "precision_at_5": float(per_variant["precision_at_5"].mean()),
        "ndcg_at_5": float(per_variant["ndcg_at_5"].mean()),
        "mrr_at_5": float(per_variant["mrr_at_5"].mean()),
        "citation_precision": citation["citation_accuracy"],
        "exact_claim_precision": citation["exact_claim_accuracy"],
        "intra_rater_kappa": float(kappa),
    }
    passed = (
        metrics["precision_at_5"] >= thresholds["precision_at_5"]
        and metrics["citation_precision"] == thresholds["citation_precision"]
        and metrics["exact_claim_precision"] == thresholds["exact_claim_precision"]
        and metrics["intra_rater_kappa"] >= thresholds["intra_rater_kappa"]
    )
    result = {
        "manifest_version": "task11f1_confirmatory_result_v1", "created_utc": _now(),
        "gene": gene, "status": "passed" if passed else "negative",
        "promotion_allowed": bool(passed), "single_use_opening": str(marker.relative_to(root)),
        "metrics": metrics, "thresholds": thresholds,
        "inputs": {key: str(path.relative_to(root)) for key, path in {
            "filled_sheet": filled_sheet, "filled_repeats": filled_repeats,
            "template_sheet": template_sheet, "template_repeats": template_repeats,
            "repeat_key": key_path, "control_manifest": control_path,
        }.items()},
        "input_sha256": {key: sha256(path) for key, path in {
            "filled_sheet": filled_sheet, "filled_repeats": filled_repeats,
            "template_sheet": template_sheet, "template_repeats": template_repeats,
            "repeat_key": key_path, "control_manifest": control_path,
        }.items()},
        "recalculated_retrieval": False, "refit_model": False,
    }
    output = out_dir / f"{gene.lower()}_rag_benchmark_v5_confirmatory_result.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    gate_path = root / "data" / "output" / gene / "rag" / f"{gene.lower()}_task11_gate_manifest.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    gate["confirmatory_v5"] = {**result, "result_sha256": sha256(output)}
    if passed:
        gate["status"] = "passed"
        gate["rag_fields_promotable_to_v2"] = True
    else:
        gate["status"] = "blocked"
        gate["rag_fields_promotable_to_v2"] = False
    gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
