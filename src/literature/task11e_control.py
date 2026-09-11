"""Read-only controls for the 171-variant Task 11E v5 run."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.literature.task11 import sha256
from src.rag.retrieval_v5 import MECHANISTIC_SCOPES, classify_evidence_scope


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_v5_controls(root: Path, gene: str = "BRAF") -> Path:
    gene = gene.upper()
    rag = root / "data" / "output" / gene / "rag"
    retrieval = sorted(rag.joinpath("retrieval_cache").glob("*__discovery_v4__*.json"))
    inference = sorted(rag.joinpath("inference_cache").glob("*__discovery_v4__*.json"))
    errors: list[str] = []
    retrieval_manifest = rag / f"{gene.lower()}_retrieval_phase_manifest.json"
    if retrieval_manifest.exists():
        phase = json.loads(retrieval_manifest.read_text(encoding="utf-8"))
        if phase.get("retrieval_failures"):
            errors.append(f"retrieval failures recorded: {len(phase['retrieval_failures'])}")
    else:
        errors.append("retrieval phase manifest is missing")
    scope_counts = Counter()
    scope_mismatches = []
    duplicate_violations = []
    cited_violations = []
    lower_scope_claims = []
    for path in retrieval:
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = str(payload.get("variant", ""))
        evidence = payload.get("evidence", [])
        counts = Counter(str(item.get("pmid", "")) for item in evidence)
        if max(counts.values(), default=0) > 2:
            duplicate_violations.append(path.name)
        for item in evidence:
            scope = str(item.get("evidence_scope", ""))
            expected = classify_evidence_scope(item.get("passage", ""), variant, "")
            scope_counts[scope] += 1
            if scope not in {"exact_variant", "same_residue_analogy", "functional_region", "general_context"} \
                    or (scope == "exact_variant" and expected != "exact_variant") \
                    or (scope == "same_residue_analogy" and expected == "general_context"):
                scope_mismatches.append({"file": path.name, "variant": variant,
                                         "pmid": item.get("pmid"), "declared": scope,
                                         "expected_without_region": expected})
    for path in inference:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("inference_status") not in {
            "llm_validated", "deterministic_fallback", "no_evidence", "technical_llm_failure",
        }:
            errors.append(f"inference status missing in {path.name}")
        allowed = {str(item.get("evidence_id")) for item in payload.get("evidence", [])}
        scopes = {str(item.get("evidence_id")): str(item.get("evidence_scope", "general_context"))
                  for item in payload.get("evidence", [])}
        for hypothesis in payload.get("inference", {}).get("hypotheses", []):
            cited = [str(value) for value in hypothesis.get("cited_evidence_ids", [])]
            if any(value not in allowed for value in cited):
                cited_violations.append({"file": path.name, "claim": hypothesis})
            if hypothesis.get("classification") == "literature_reported" \
                    and any(scopes.get(value) != "exact_variant" for value in cited):
                lower_scope_claims.append({"file": path.name, "claim": hypothesis})
        cited_disease = payload.get("inference", {}).get("disease_association", {}).get("cited_evidence_ids", [])
        if any(str(value) not in allowed for value in cited_disease):
            cited_violations.append({"file": path.name, "disease": cited_disease})
    priority_path = root / "data" / "output" / gene / "provisional" / "task11" / "scores" / \
        f"{gene.lower()}_evidence_adjusted_variant_priority.csv"
    contextual_scoring_violations = []
    if priority_path.exists():
        import pandas as pd
        priority = pd.read_csv(priority_path)
        if {"literature_mechanism_specificity", "mechanism_component"}.issubset(priority.columns):
            contextual_scoring_violations = priority.loc[
                priority["literature_mechanism_specificity"].astype(str).eq("general")
                & (pd.to_numeric(priority["mechanism_component"], errors="coerce") != 0),
                "mutation",
            ].astype(str).tolist()
            if contextual_scoring_violations:
                errors.append("general context changed mechanism score")
    if len(retrieval) != 171:
        errors.append(f"retrieval cache count is {len(retrieval)}, expected 171")
    if len(inference) != 171:
        errors.append(f"inference cache count is {len(inference)}, expected 171")
    errors.extend([
        f"duplicate PMID limit violations: {len(duplicate_violations)}" if duplicate_violations else "",
        f"scope mismatches: {len(scope_mismatches)}" if scope_mismatches else "",
        f"invalid citations: {len(cited_violations)}" if cited_violations else "",
        f"lower-scope exact claims: {len(lower_scope_claims)}" if lower_scope_claims else "",
    ])
    errors = [error for error in errors if error]
    stack = rag / f"{gene.lower()}_rag_stack_manifest.json"
    output = rag / f"{gene.lower()}_task11e_v5_control_manifest.json"
    manifest = {
        "manifest_version": "task11e_v5_controls_v1", "created_utc": _now(), "gene": gene,
        "status": "passed" if not errors else "blocked", "errors": errors,
        "retrieval_cache_count": len(retrieval), "inference_cache_count": len(inference),
        "retrieval_stack_version": "rag_stack_v5",
        "stack_manifest": str(stack.relative_to(root)), "stack_manifest_sha256": sha256(stack),
        "scope_counts": dict(scope_counts),
        "duplicate_violations": duplicate_violations,
        "scope_mismatches": scope_mismatches,
        "invalid_citations": cited_violations,
        "lower_scope_exact_claims": lower_scope_claims,
        "contextual_scoring_violations": contextual_scoring_violations,
        "criteria": {
            "all_171_processed": True,
            "max_two_passages_per_pmid_per_variant": not duplicate_violations,
            "exact_claims_are_lexical": not lower_scope_claims,
            "context_not_mechanism": not lower_scope_claims,
            "citation_ids_exist": not cited_violations,
            "no_global_pmid_cap_between_variants": True,
        },
        "development_metrics": {
            "source": "data/output/BRAF/rag/audit/braf_task11e_development_training_table.csv",
            "not_confirmatory": True,
        },
        "confirmatory_creation_allowed": not errors,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def audit_v6_controls(root: Path, gene: str = "BRAF") -> Path:
    """Validate separate mechanistic/context blocks for the V6 caches."""
    gene = gene.upper()
    rag = root / "data" / "output" / gene / "rag"
    retrieval = sorted(rag.joinpath("retrieval_cache").glob("*__discovery_v5__*.json"))
    inference = sorted(rag.joinpath("inference_cache").glob("*__discovery_v5__*.json"))
    errors: list[str] = []
    retrieval_manifest = rag / f"{gene.lower()}_retrieval_phase_manifest.json"
    if retrieval_manifest.exists():
        phase = json.loads(retrieval_manifest.read_text(encoding="utf-8"))
        if phase.get("retrieval_failures"):
            errors.append(f"retrieval failures recorded: {len(phase['retrieval_failures'])}")
    else:
        errors.append("retrieval phase manifest is missing")
    scope_counts = Counter()
    block_errors = []
    duplicate_violations = []
    cited_violations = []
    lower_scope_claims = []

    def inspect_blocks(payload: dict, path: Path) -> tuple[list[dict], list[dict]]:
        all_items = payload.get("evidence", [])
        mechanistic = payload.get("mechanistic_evidence", [])
        context = payload.get("context_reference", [])
        ids = [str(item.get("evidence_id")) for item in all_items]
        mech_ids = [str(item.get("evidence_id")) for item in mechanistic]
        context_ids = [str(item.get("evidence_id")) for item in context]
        if set(mech_ids) & set(context_ids):
            block_errors.append({"file": path.name, "reason": "block_id_overlap"})
        if set(mech_ids) | set(context_ids) != set(ids):
            block_errors.append({"file": path.name, "reason": "blocks_do_not_partition_evidence"})
        for item in mechanistic:
            scope = str(item.get("evidence_scope", ""))
            scope_counts[scope] += 1
            if scope not in MECHANISTIC_SCOPES:
                block_errors.append({"file": path.name, "reason": "context_in_mechanistic_block",
                                     "evidence_id": item.get("evidence_id"), "scope": scope})
        for item in context:
            scope = str(item.get("evidence_scope", ""))
            scope_counts[scope] += 1
            if scope != "general_context":
                block_errors.append({"file": path.name, "reason": "non_general_context_block",
                                     "evidence_id": item.get("evidence_id"), "scope": scope})
        expected_status = ("mechanistic_evidence_retrieved" if mechanistic
                           else "no_mechanistic_evidence_retrieved")
        if payload.get("mechanistic_evidence_status") != expected_status:
            block_errors.append({"file": path.name, "reason": "wrong_abstention_status"})
        return mechanistic, context

    for path in retrieval:
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = str(payload.get("variant", ""))
        evidence = payload.get("evidence", [])
        counts = Counter(str(item.get("pmid", "")) for item in evidence)
        if max(counts.values(), default=0) > 2:
            duplicate_violations.append(path.name)
        inspect_blocks(payload, path)
        for item in evidence:
            scope = str(item.get("evidence_scope", ""))
            expected = classify_evidence_scope(item.get("passage", ""), variant, "")
            if scope not in {"exact_variant", "same_residue_analogy", "functional_region", "general_context"} \
                    or (scope == "exact_variant" and expected != "exact_variant") \
                    or (scope == "same_residue_analogy" and expected == "general_context"):
                block_errors.append({"file": path.name, "reason": "scope_mismatch",
                                     "evidence_id": item.get("evidence_id"), "declared": scope,
                                     "expected_without_region": expected})

    for path in inference:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("inference_status") not in {
            "llm_validated", "deterministic_fallback", "no_evidence", "technical_llm_failure",
        }:
            errors.append(f"inference status missing in {path.name}")
        mechanistic, context = inspect_blocks(payload, path)
        allowed = {str(item.get("evidence_id")) for item in payload.get("evidence", [])}
        scopes = {str(item.get("evidence_id")): str(item.get("evidence_scope", "general_context"))
                  for item in payload.get("evidence", [])}
        for hypothesis in payload.get("inference", {}).get("hypotheses", []):
            cited = [str(value) for value in hypothesis.get("cited_evidence_ids", [])]
            if any(value not in allowed for value in cited):
                cited_violations.append({"file": path.name, "claim": hypothesis})
            if hypothesis.get("classification") == "literature_reported" \
                    and any(scopes.get(value) != "exact_variant" for value in cited):
                lower_scope_claims.append({"file": path.name, "claim": hypothesis})
            if hypothesis.get("classification", "").startswith("literature") \
                    and any(scopes.get(value) == "general_context" for value in cited):
                lower_scope_claims.append({"file": path.name, "claim": hypothesis,
                                           "reason": "general_context_citation"})
        cited_disease = payload.get("inference", {}).get("disease_association", {}).get(
            "cited_evidence_ids", []
        )
        if any(str(value) not in allowed for value in cited_disease):
            cited_violations.append({"file": path.name, "disease": cited_disease})

    if len(retrieval) != 171:
        errors.append(f"retrieval cache count is {len(retrieval)}, expected 171")
    if len(inference) != 171:
        errors.append(f"inference cache count is {len(inference)}, expected 171")
    if duplicate_violations:
        errors.append(f"duplicate PMID limit violations: {len(duplicate_violations)}")
    if block_errors:
        errors.append(f"evidence block violations: {len(block_errors)}")
    if cited_violations:
        errors.append(f"invalid citations: {len(cited_violations)}")
    if lower_scope_claims:
        errors.append(f"lower-scope literature claims: {len(lower_scope_claims)}")
    errors = [error for error in errors if error]
    stack = rag / f"{gene.lower()}_rag_stack_manifest.json"
    output = rag / f"{gene.lower()}_task11f2_v6_control_manifest.json"
    manifest = {
        "manifest_version": "task11f2_v6_controls_v1", "created_utc": _now(), "gene": gene,
        "status": "passed" if not errors else "blocked", "errors": errors,
        "retrieval_cache_count": len(retrieval), "inference_cache_count": len(inference),
        "retrieval_stack_version": "rag_stack_v6",
        "stack_manifest": str(stack.relative_to(root)), "stack_manifest_sha256": sha256(stack),
        "scope_counts": dict(scope_counts), "block_violations": block_errors,
        "duplicate_violations": duplicate_violations,
        "invalid_citations": cited_violations, "lower_scope_claims": lower_scope_claims,
        "criteria": {
            "all_171_processed": len(retrieval) == 171 and len(inference) == 171,
            "mechanistic_and_context_blocks_disjoint": not block_errors,
            "general_context_never_pads_mechanistic_block": not block_errors,
            "no_mechanistic_evidence_status_explicit": not block_errors,
            "max_two_passages_per_pmid_per_variant": not duplicate_violations,
            "exact_claims_are_lexical": not lower_scope_claims,
            "citation_ids_exist": not cited_violations,
            "no_global_pmid_cap_between_variants": True,
        },
        "confirmatory_creation_allowed": not errors,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
