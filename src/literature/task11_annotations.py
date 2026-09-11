"""Freeze and audit the historical Task 11 annotation sheets.

The filled sheets are useful development evidence, but they were produced
under an older thematic rubric.  This module therefore treats their bytes as
immutable inputs, validates them against their own closed historical
vocabularies, and writes only explicitly diagnostic derivatives.  Nothing in
this module is a source for the confirmatory benchmark gate.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.literature.benchmark import (
    HUMAN_REQUIRED_COLUMNS, SYSTEM_CITATION_COLUMNS, CONFIRMATORY_VOCABULARIES,
)
from src.literature.task11 import sha256


DEVELOPMENT_CLASSIFICATION = "exploratory_development"
ARCHIVE_DIRNAME = "archive/task11_exploratory_development"

# These are closed vocabularies observed in the two pre-11C filled sheets.
# They are not accepted by benchmark_gate.  Their only purpose is to make the
# historical record auditable without silently treating it as a new rubric.
LEGACY_MAIN_VOCABULARIES = {
    "relevance": frozenset({0, 1, 2, 3}),
    "specificity": frozenset({"exact_variant", "same_residue", "same_functional_motif",
                               "same_domain", "gene_level", "none"}),
    "entailment": frozenset({"direct", "indirect", "mention_only", "none"}),
    "support_type": frozenset({
        "preclinical_mechanistic", "resistance_mechanism_background",
        "clinical_resistance_case", "biochemical_functional_assay",
        "therapeutic_mechanism_preclinical", "functional_mechanism_background",
        "clinical_therapeutic_background", "cellular_pathway_assay", "screening_method",
        "therapeutic_context", "noninformative", "clinical_response_cohort",
        "disease_epigenetic_background", "clinical_response_case",
        "disease_therapeutic_background", "cellular_drug_response_model",
        "lexical_collision", "therapeutic_background",
    }),
    "quality": frozenset({"high", "medium", "low", "none"}),
    "conflict": frozenset({"no", "substitution_specificity_risk"}),
}

LEGACY_BLIND_VOCABULARIES = {
    "relevance": frozenset({"high", "medium"}),
    "specificity": frozenset({"high", "medium", "low"}),
    "entailment": frozenset({"full", "partial", "insufficient"}),
    "support_type": frozenset({"direct_mechanistic", "direct_clinical", "background", "metadata_only"}),
    "quality": frozenset({"high", "medium", "low"}),
    "conflict": frozenset({"no"}),
}

LEGACY_TO_DIAGNOSTIC = {
    "specificity": {
        "exact_variant": "exact_variant", "same_residue": "same_residue",
        "same_functional_motif": "functional_region", "same_domain": "general",
        "gene_level": "general", "none": "general",
        "high": "exact_variant", "medium": "functional_region", "low": "general",
    },
    "entailment": {
        "direct": "supports", "indirect": "neutral", "mention_only": "neutral", "none": "neutral",
        "full": "supports", "partial": "neutral", "insufficient": "neutral",
    },
    "support_type": {
        "preclinical_mechanistic": "structural_biophysical", "biochemical_functional_assay": "biochemical",
        "cellular_pathway_assay": "cell_functional", "cellular_drug_response_model": "cell_functional",
        "clinical_resistance_case": "human_association", "clinical_response_case": "human_association",
        "clinical_response_cohort": "human_association", "direct_mechanistic": "structural_biophysical",
        "direct_clinical": "human_association", "background": "context", "metadata_only": "context",
        "resistance_mechanism_background": "context", "therapeutic_mechanism_preclinical": "context",
        "functional_mechanism_background": "context", "clinical_therapeutic_background": "context",
        "screening_method": "computational", "therapeutic_context": "context",
        "noninformative": "context", "disease_epigenetic_background": "context",
        "disease_therapeutic_background": "context", "lexical_collision": "context",
        "therapeutic_background": "context",
    },
    "quality": {"high": "controlled", "medium": "uncontrolled", "low": "unclear", "none": "unclear"},
    "conflict": {"no": "no", "substitution_specificity_risk": "yes"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _values(frame: pd.DataFrame, column: str) -> set:
    if column not in frame:
        return set()
    values = frame[column].dropna().tolist()
    return {value.strip() if isinstance(value, str) else value for value in values}


def validate_annotation_vocabulary(frame: pd.DataFrame, *, profile: str,
                                    source_name: str = "annotations") -> dict:
    """Strictly validate one historical or confirmatory annotation schema."""
    if profile == "legacy_main":
        vocabularies = {**LEGACY_MAIN_VOCABULARIES,
                        "cited_in_claim": frozenset({"yes", "no"})}
    elif profile == "legacy_blind":
        vocabularies = {**LEGACY_BLIND_VOCABULARIES,
                        "cited_in_claim": frozenset({"yes", "no"})}
    elif profile == "confirmatory":
        vocabularies = CONFIRMATORY_VOCABULARIES
    else:
        raise ValueError(f"unknown annotation vocabulary profile: {profile}")

    required = [*HUMAN_REQUIRED_COLUMNS, *SYSTEM_CITATION_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    unknown = {}
    blank = []
    for column in required:
        if column not in frame:
            continue
        raw = frame[column]
        if raw.isna().any() or raw.astype(str).str.strip().eq("").any():
            blank.append(column)
        values = _values(frame, column)
        allowed = vocabularies.get(column)
        if allowed is not None:
            bad = sorted((value for value in values if value not in allowed), key=str)
            if bad:
                unknown[column] = bad
    if missing or blank or unknown:
        raise ValueError(json.dumps({
            "source": source_name, "profile": profile, "missing": missing,
            "blank": blank, "unknown_values": unknown,
        }, ensure_ascii=False, sort_keys=True))
    return {
        "source": source_name, "profile": profile, "status": "passed",
        "rows": int(len(frame)), "validated_columns": required,
        "vocabulary_hash": _vocabulary_hash(vocabularies),
    }


def _vocabulary_hash(vocabularies: dict) -> str:
    import hashlib
    payload = json.dumps({key: sorted(map(str, values)) for key, values in sorted(vocabularies.items())},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _diagnostic_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a new, canonicalized diagnostic frame; never mutate the source."""
    out = frame.copy()
    for column in [*HUMAN_REQUIRED_COLUMNS, "cited_in_claim"]:
        if column not in out:
            continue
        if column == "relevance":
            out["relevance"] = out[column].map({0: "irrelevant", 1: "background", 2: "relevant", 3: "directly_on_point"})
            out["relevance"] = out["relevance"].fillna(out[column].astype(str).str.strip().str.lower().map({
                "high": "relevant", "medium": "background", "low": "irrelevant",
            }))
        elif column == "cited_in_claim":
            out[column] = out[column].astype(str).str.strip().str.lower()
        elif column in LEGACY_TO_DIAGNOSTIC:
            out[f"{column}_normalized"] = out[column].map(LEGACY_TO_DIAGNOSTIC[column])
            out[column] = out[f"{column}_normalized"]
        else:
            out[column] = out[column].astype(str).str.strip().str.lower()
    out["diagnostic_only"] = True
    out["confirmatory_eligible"] = False
    out["kappa_eligible"] = False
    out["annotation_classification"] = DEVELOPMENT_CLASSIFICATION
    return out


def freeze_exploratory_annotations(root: Path, gene: str = "BRAF") -> Path:
    """Hash and archive both filled sheets without changing either source."""
    gene = gene.upper()
    benchmark = root / "data" / "output" / gene / "rag" / "benchmark"
    sources = {
        "main": benchmark / f"{gene.lower()}_rag_benchmark_filled.csv",
        "blind": benchmark / f"{gene.lower()}_rag_benchmark_blind_filled.csv",
    }
    archive = root / "data" / "output" / gene / "rag" / ARCHIVE_DIRNAME
    diagnostic = benchmark / "diagnostic"
    archive.mkdir(parents=True, exist_ok=True)
    diagnostic.mkdir(parents=True, exist_ok=True)
    records = []
    for label, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source)
        profile = "legacy_main" if label == "main" else "legacy_blind"
        validation = validate_annotation_vocabulary(frame, profile=profile,
                                                    source_name=str(source.relative_to(root)))
        digest = sha256(source)
        target = archive / source.name
        if target.exists() and sha256(target) != digest:
            raise RuntimeError(f"archive collision with different bytes: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        normalized = _diagnostic_normalize(frame)
        normalized_path = diagnostic / f"{source.stem}_normalized.csv"
        normalized.to_csv(normalized_path, index=False)
        records.append({
            "label": label, "source": str(source.relative_to(root)),
            "archive": str(target.relative_to(root)), "sha256": digest,
            "bytes": source.stat().st_size, "rows": int(len(frame)),
            "classification": DEVELOPMENT_CLASSIFICATION,
            "confirmatory_benchmark_eligible": False, "kappa_eligible": False,
            "vocabulary_validation": validation,
            "diagnostic_normalized": str(normalized_path.relative_to(root)),
        })
    manifest = {
        "manifest_version": "task11_annotations_v1",
        "created_utc": _now(), "gene": gene,
        "classification": DEVELOPMENT_CLASSIFICATION,
        "status": "frozen_as_exploratory_development",
        "source_policy": "the two filled CSVs are byte-preserved and never used by benchmark_gate",
        "confirmatory_benchmark": str((benchmark / f"{gene.lower()}_rag_benchmark_sheet.csv").relative_to(root)),
        "confirmatory_status": "blocked_until_new_v2_benchmark_is_annotated_and_passes",
        "blind_evaluation_status": "thematic_only;not_kappa_eligible",
        "files": records,
    }
    output = root / "data" / "output" / gene / "rag" / f"{gene.lower()}_task11_exploratory_annotations_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
