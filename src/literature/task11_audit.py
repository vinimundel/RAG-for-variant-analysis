"""Development-only failure audit for the historical Task 11 RAG run."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.literature.task11 import sha256
from src.literature.task11_annotations import DEVELOPMENT_CLASSIFICATION


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mode_map(rag: Path) -> dict[str, dict]:
    output = {}
    for path in sorted((rag / "inference_cache").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            inference = payload.get("inference", {})
            variant = str(inference.get("variant", ""))
            if variant:
                output[variant] = {
                    "llm_contract_status": payload.get("llm_contract_status", "unknown"),
                    "generation_mode": (
                        "llm_validated" if payload.get("llm_contract_status") == "passed"
                        else "deterministic_fallback"
                    ),
                    "evidence_status": payload.get("evidence_status", "unknown"),
                }
        except (OSError, json.JSONDecodeError):
            continue
    return output


def _channel_map(rag: Path) -> dict[tuple[str, str], list[str]]:
    output = {}
    for path in sorted((rag / "retrieval_cache").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            variant = str(payload.get("variant", ""))
            channels = payload.get("retrieval_diagnostics", {}).get("channels_per_selected_document", {})
            for item in payload.get("evidence", []):
                output[(variant, str(item.get("pmid", "")))] = list(channels.get(str(item.get("pmid", "")), []))
        except (OSError, json.JSONDecodeError):
            continue
    return output


def _citation_class(row: pd.Series) -> str:
    if str(row.get("cited_in_claim", "no")) != "yes":
        return "not_cited"
    if str(row.get("relevance")) == "irrelevant":
        return "irrelevant"
    if str(row.get("specificity")) == "exact_variant" and str(row.get("entailment")) == "supports" \
            and str(row.get("relevance")) == "directly_on_point":
        return "exact_supported"
    return "indirect"


def audit_task11_failures(root: Path, gene: str = "BRAF") -> Path:
    gene = gene.upper()
    rag = root / "data" / "output" / gene / "rag"
    benchmark = rag / "benchmark"
    diagnostic = benchmark / "diagnostic" / f"{gene.lower()}_rag_benchmark_filled_normalized.csv"
    source = benchmark / f"{gene.lower()}_rag_benchmark_filled.csv"
    if not diagnostic.exists() or not source.exists():
        raise FileNotFoundError("Task 11 exploratory annotation files must be frozen first")
    annotations = pd.read_csv(diagnostic)
    mode_map = _mode_map(rag)
    channels = _channel_map(rag)
    annotations["citation_class"] = annotations.apply(_citation_class, axis=1)
    annotations["general_as_specific"] = (
        annotations["cited_in_claim"].astype(str).eq("yes")
        & annotations["specificity"].astype(str).eq("general")
    )
    annotations["same_residue_analogy"] = annotations["specificity"].astype(str).eq("same_residue")
    annotations["generation_mode"] = annotations["mutation"].map(
        lambda value: mode_map.get(str(value), {}).get("generation_mode", "unknown")
    )
    annotations["llm_contract_status"] = annotations["mutation"].map(
        lambda value: mode_map.get(str(value), {}).get("llm_contract_status", "unknown")
    )
    annotations["retrieval_channels"] = annotations.apply(
        lambda row: ";".join(channels.get((str(row["mutation"]), str(row["pmid"])), [])), axis=1
    )

    out_dir = rag / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_rows = []
    for (mutation, stratum), group in annotations.groupby(["mutation", "stratum"], sort=True):
        relevant = group["relevance"].isin({"relevant", "directly_on_point"})
        variant_rows.append({
            "mutation": mutation, "stratum": stratum, "passages": int(len(group)),
            "precision_at_5": float(relevant.mean()),
            "cited_passages": int(group["cited_in_claim"].eq("yes").sum()),
            "exact_supported_citations": int(group["citation_class"].eq("exact_supported").sum()),
            "indirect_citations": int(group["citation_class"].eq("indirect").sum()),
            "irrelevant_citations": int(group["citation_class"].eq("irrelevant").sum()),
            "general_as_specific": int(group["general_as_specific"].sum()),
            "same_residue_analogies": int(group["same_residue_analogy"].sum()),
            "generation_mode": group["generation_mode"].iloc[0],
            "llm_contract_status": group["llm_contract_status"].iloc[0],
        })
    variant = pd.DataFrame(variant_rows)
    variant_path = out_dir / "task11d_precision_at5_by_variant.csv"
    variant.to_csv(variant_path, index=False)

    channel_rows = []
    exploded = annotations.assign(_channel=annotations["retrieval_channels"].str.split(";")) \
        .explode("_channel")
    exploded = exploded.loc[exploded["_channel"].fillna("").ne("")]
    for channel, group in exploded.groupby("_channel", sort=True):
        relevant = group["relevance"].isin({"relevant", "directly_on_point"})
        cited = group["cited_in_claim"].eq("yes")
        channel_rows.append({
            "query_channel": channel, "passages_attributed": int(len(group)),
            "precision_attributed": float(relevant.mean()),
            "cited_passages": int(cited.sum()),
            "exact_supported_citations": int(group["citation_class"].eq("exact_supported").sum()),
            "indirect_citations": int(group["citation_class"].eq("indirect").sum()),
            "general_as_specific": int(group["general_as_specific"].sum()),
        })
    channel_path = out_dir / "task11d_performance_by_query_channel.csv"
    pd.DataFrame(channel_rows).to_csv(channel_path, index=False)

    pmid_counts = annotations["pmid"].astype(str).value_counts().rename_axis("pmid").reset_index(name="passages")
    pmid_counts["share"] = pmid_counts["passages"] / len(annotations)
    pmid_counts["excessive_repetition"] = pmid_counts["passages"] > 2
    pmid_path = out_dir / "task11d_repeated_pmids.csv"
    pmid_counts.to_csv(pmid_path, index=False)

    failure_columns = [
        "item_id", "mutation", "stratum", "rank", "pmid", "article_title", "section",
        "specificity", "entailment", "relevance", "support_type", "cited_in_claim",
        "system_citation_key", "citation_class", "general_as_specific", "same_residue_analogy",
        "generation_mode", "llm_contract_status", "retrieval_channels", "reviewer_note",
    ]
    failures = annotations.loc[
        annotations["citation_class"].isin({"indirect", "irrelevant"})
        | annotations["general_as_specific"] | annotations["same_residue_analogy"],
        [column for column in failure_columns if column in annotations],
    ]
    failure_path = out_dir / "task11d_failure_cases.csv"
    failures.to_csv(failure_path, index=False)

    mode_rows = []
    for mode, group in annotations.groupby("generation_mode", sort=True):
        cited = group["cited_in_claim"].eq("yes")
        mode_rows.append({
            "generation_mode": mode, "passages": int(len(group)),
            "precision_attributed": float(group["relevance"].isin({"relevant", "directly_on_point"}).mean()),
            "cited_passages": int(cited.sum()),
            "exact_supported_citations": int(group["citation_class"].eq("exact_supported").sum()),
            "indirect_citations": int(group["citation_class"].eq("indirect").sum()),
            "irrelevant_citations": int(group["citation_class"].eq("irrelevant").sum()),
            "general_as_specific": int(group["general_as_specific"].sum()),
            "same_residue_analogies": int(group["same_residue_analogy"].sum()),
        })
    mode_path = out_dir / "task11d_generation_mode_comparison.csv"
    pd.DataFrame(mode_rows).to_csv(mode_path, index=False)

    manifest = {
        "manifest_version": "task11d_failure_audit_v1", "created_utc": _now(), "gene": gene,
        "classification": DEVELOPMENT_CLASSIFICATION, "confirmatory": False,
        "source_annotations": str(source.relative_to(root)), "source_annotations_sha256": sha256(source),
        "normalized_annotations": str(diagnostic.relative_to(root)), "normalized_sha256": sha256(diagnostic),
        "official_benchmark_gate_uses_filled_annotations": False,
        "definitions": {
            "precision_at_5": "reviewer relevance in {relevant,directly_on_point} over five rows",
            "exact_supported": "system cited + exact_variant + supports + directly_on_point",
            "indirect": "system cited but exact support contract is not met",
            "general_as_specific": "cited passage labeled general",
            "same_residue_analogy": "passage concerns another substitution at the same residue",
        },
        "outputs": {key: str(path.relative_to(root)) for key, path in {
            "variant": variant_path, "channel": channel_path, "pmid": pmid_path,
            "failures": failure_path, "generation_mode": mode_path,
        }.items()},
        "counts": {
            "annotated_passages": int(len(annotations)),
            "variants": int(annotations["mutation"].nunique()),
            "cited_passages": int(annotations["cited_in_claim"].eq("yes").sum()),
            "general_as_specific": int(annotations["general_as_specific"].sum()),
            "same_residue_analogies": int(annotations["same_residue_analogy"].sum()),
            "pmids": int(annotations["pmid"].nunique()),
            "max_passages_one_pmid": int(pmid_counts["passages"].max()),
        },
    }
    output = out_dir / "task11d_failure_audit_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
