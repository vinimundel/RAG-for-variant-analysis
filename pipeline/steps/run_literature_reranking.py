"""Run the discovery-v3 literature, biophysical-hypothesis and priority workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pipeline.config import get_gene_config, paths_for
from src.features.discovery_hypotheses import annotate_discovery_hypotheses
from src.model.evidence_priority import add_discovery_priority
from src.model.mechanistic_hypothesis import render_mechanistic_hypothesis
from src.rag.chain import EVIDENCE_LAYER, DiscoveryLiteratureAnalyzer
from src.rag.llm import llm_model_name
from src.rag.prompts import DISCOVERY_PROMPT_VERSION
from src.rag.schemas import DiscoveryQuery, DiscoveryRAGResult, LiteratureEvidence
from src.rag.query_profiles import QUERY_VERSION, get_condition_profile
from src.rag.variant_mentions import (
    contains_exact_variant,
    same_residue_variant_mentioned,
)
from src.literature.task11 import RETRIEVAL_STACK_VERSION, task11_provisional_dir


def _cache_path(cache_dir: Path, variant: str, condition_profile: str = "braf_oncology",
                condition_terms: tuple[str, ...] = (), query_version: str = "discovery_v3",
                index_fingerprint: str = "unversioned") -> Path:
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", llm_model_name())
    context = json.dumps({"profile": condition_profile, "terms": condition_terms,
                          "query_version": query_version, "index": index_fingerprint}, sort_keys=True)
    context_hash = hashlib.sha256(context.encode()).hexdigest()[:10]
    return cache_dir / (
        f"{variant}__{query_version}__{condition_profile}__{context_hash}__"
        f"{DISCOVERY_PROMPT_VERSION}__{model}.json"
    )


def _retrieval_cache_path(cache_dir: Path, variant: str, condition_profile: str,
                          condition_terms: tuple[str, ...], query_version: str,
                          index_fingerprint: str = "unversioned") -> Path:
    context = json.dumps({"profile": condition_profile, "terms": condition_terms,
                          "query_version": query_version, "index": index_fingerprint}, sort_keys=True)
    context_hash = hashlib.sha256(context.encode()).hexdigest()[:10]
    return cache_dir / f"{variant}__{query_version}__{condition_profile}__{context_hash}__retrieval.json"


def _discovery_request(row: dict, gene: str, condition_profile: str,
                       condition_terms: tuple[str, ...], query_version: str) -> DiscoveryQuery:
    return DiscoveryQuery(
        gene=gene, variant=str(row["mutation"]), position=int(row["pos"]),
        functional_region=str(row["functional_region"]),
        ppi_partner=(str(row["ppi_partner"]) if pd.notna(row.get("ppi_partner")) else None),
        structural_observation=(
            f"delta_charge={row.get('delta_charge')}; delta_volume={row.get('delta_volume')} A3; "
            f"delta_hydrophobicity={row.get('delta_hydrophobicity')}; "
            f"secondary_structure={row.get('secondary_structure', 'unknown')}; "
            f"flags={row.get('mechanistic_flags', '')}"
        ),
        biophysical_hypotheses_json=str(row["biophysical_hypotheses"]), k=10,
        condition_profile=condition_profile, condition_terms=list(condition_terms),
        query_version=query_version, pharmacology_requested=True,
    )


def _specificity(variant: str, evidence, cited_ids: list[str], functional_region: str = "") -> str:
    cited = [item for item in evidence if item.evidence_id in set(cited_ids)]
    joined = " ".join(item.passage for item in cited).lower()
    if contains_exact_variant(joined, variant):
        return "exact_variant"
    if same_residue_variant_mentioned(joined, variant):
        return "same_residue"
    aliases = {functional_region.lower().replace("_", " ")}
    if functional_region == "activation_segment": aliases.add("activation loop")
    if functional_region == "p_loop": aliases.update({"p-loop", "phosphate-binding loop"})
    if functional_region == "alpha_c_helix": aliases.update({"alpha-c helix", "αc helix"})
    if functional_region == "interdomain_linker": aliases.update({"interdomain linker", "cleavage site"})
    if functional_region == "membrane_insertion_hairpins": aliases.update({"beta hairpin", "β-hairpin", "pore"})
    if functional_region == "pore_forming_n_terminal_domain": aliases.update({"n-terminal domain", "pore-forming domain"})
    if any(alias and alias in joined for alias in aliases):
        return "functional_region"
    return "general" if cited else "none"


def _source_type(evidence, cited_ids: list[str]) -> str:
    kinds = {item.source_type for item in evidence if item.evidence_id in set(cited_ids)}
    return "grobid_full_text" if "grobid_full_text" in kinds else (
        "pubmed_abstract" if "pubmed_abstract" in kinds else "none"
    )


def _condition_tier(profile_name: str, evidence, cited_ids: list[str]) -> str:
    passage = " ".join(item.passage.lower() for item in evidence if item.evidence_id in set(cited_ids))
    if not passage:
        return "none"
    profile = get_condition_profile(profile_name)
    for tier_index, terms in enumerate(profile.tiers, 1):
        if any(term.lower() in passage for term in terms):
            return f"tier_{tier_index}"
    return "profile_context_unmatched"


def _normalize_generation_modes(result: DiscoveryRAGResult) -> DiscoveryRAGResult:
    """Make provenance explicit even for caches written before Task 11B."""
    hypotheses = []
    for hypothesis in result.inference.hypotheses:
        if hypothesis.rule_ids:
            mode = "biophysical_rule"
        elif result.llm_contract_status == "deterministic_fallback":
            mode = "deterministic"
        else:
            mode = "llm_validated"
        hypotheses.append(hypothesis.model_copy(update={"generation_mode": mode}))
    disease_mode = ("deterministic" if result.llm_contract_status == "deterministic_fallback"
                    else "llm_validated")
    inference = result.inference.model_copy(update={
        "hypotheses": hypotheses,
        "disease_association": result.inference.disease_association.model_copy(
            update={"generation_mode": disease_mode}
        ),
    })
    evidence_status = result.evidence_status
    if evidence_status == "none":
        if any(h.classification == "literature_reported" for h in inference.hypotheses):
            evidence_status = "exact"
        elif any(h.classification == "literature_supported_hypothesis" for h in inference.hypotheses):
            evidence_status = "indirect"
    if result.llm_contract_status == "deterministic_fallback":
        inference_status = "no_evidence" if evidence_status == "none" else "technical_llm_failure"
    else:
        inference_status = "llm_validated"
    failure_reason = result.llm_failure_reason
    if result.llm_contract_status == "deterministic_fallback" and failure_reason is None:
        failure_reason = "legacy_cache_fallback_without_error_detail"
    return result.model_copy(update={
        "inference": inference, "evidence_status": evidence_status,
        "inference_status": inference_status, "llm_failure_reason": failure_reason,
    })


_THERAPY_AGENTS = (
    "dabrafenib", "vemurafenib", "encorafenib", "trametinib", "cobimetinib",
    "binimetinib", "sorafenib", "LY3009120",
)


def _exact_variant_therapy_context(variant: str, evidence) -> dict[str, str]:
    """Extract an auditable therapy observation, never a treatment recommendation."""
    hits = []
    for item in evidence:
        passage = item.passage.replace("\n", " ")
        if not contains_exact_variant(passage, variant):
            continue
        agents = [agent for agent in _THERAPY_AGENTS if re.search(rf"\b{re.escape(agent)}\b", passage, re.I)]
        if agents:
            hits.append((item, agents, passage.lower()))
    if not hits:
        return {"therapy_evidence_status": "none", "therapy_hypothesis": "",
                "therapy_cited_evidence_ids": "", "therapy_source_type": "none"}
    item, agents, lowered = hits[0]
    names = ", ".join(agents)
    has_resistance = any(word in lowered for word in ("resistant", "resistance", "refractory", "insensitive"))
    has_sensitivity = any(word in lowered for word in ("sensitive", "sensitivity", "response", "responded", "efficacy"))
    if has_resistance and has_sensitivity:
        status = "mixed_or_context_dependent"
        statement = (
            f"An exact-variant therapeutic passage discusses {variant} with {names}, but reports "
            "context-dependent sensitivity/resistance; this warrants model-specific testing."
        )
    elif has_resistance:
        status = "resistance_or_refractoriness_reported"
        statement = (
            f"A retrieved exact-variant passage reports {variant} in a {names} resistance/refractoriness "
            "context; this is a literature observation, not a treatment recommendation."
        )
    elif has_sensitivity:
        status = "sensitivity_or_response_reported"
        statement = (
            f"A retrieved exact-variant passage reports {variant} in a {names} sensitivity/response context; "
            "the finding requires disease- and model-specific validation."
        )
    else:
        status = "agent_discussed_without_direction"
        statement = (
            f"A retrieved exact-variant passage discusses {variant} together with {names}; no directional "
            "response claim was extracted."
        )
    return {
        "therapy_evidence_status": status,
        "therapy_hypothesis": statement,
        "therapy_cited_evidence_ids": item.evidence_id,
        "therapy_source_type": item.source_type,
    }


def _save_all_formats(fig, path: Path) -> None:
    for extension in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=600 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_figures(frame: pd.DataFrame, figure_dir: Path, gene: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    top = frame.nsmallest(30, "variant_priority_rank").sort_values("variant_priority_score")
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F2CF5B"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    left = np.zeros(len(top))
    for column, label, color in zip(
        ["structural_component", "mechanism_component", "disease_component", "biophysical_component"],
        ["Structural", "Literature mechanism", "Variant–disease", "Biophysical/context"], colors,
    ):
        values = top[column].to_numpy(float)
        ax.barh(top["mutation"], values, left=left, label=label, color=color)
        left += values
    ax.set(xlabel="Discovery variant-priority score (0–100)", ylabel="Variant",
           title=f"{gene} exploratory priority: auditable component contributions")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    sns.despine(ax=ax)
    _save_all_formats(fig, figure_dir / f"{gene.lower()}_evidence_adjusted_priority")

    flag_columns = [c for c in ["flag_charge_shift", "flag_volume_shift", "flag_hydrophobicity_shift",
                                "flag_buried_residue", "flag_ppi_interface_context",
                                "flag_ppi_hotspot_candidate", "flag_active_site_context",
                                "flag_phosphosite_context", "flag_allosteric_high_coupling"] if c in top]
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    sns.heatmap(top.set_index("mutation")[flag_columns].astype(int), cmap=["#F2F2F2", "#D55E00"],
                cbar=False, linewidths=.25, linecolor="white", ax=ax)
    ax.set(title="Mechanistic possibility flags among top discovery variants", xlabel="Rule flag", ylabel="Variant")
    ax.set_xticklabels([c.removeprefix("flag_").replace("_", " ") for c in flag_columns], rotation=40, ha="right")
    _save_all_formats(fig, figure_dir / f"{gene.lower()}_discovery_mechanistic_flag_map")

    classifications = []
    for value in frame["discovery_hypotheses"]:
        classifications.extend(item["classification"] for item in json.loads(value))
    counts = pd.Series(classifications).value_counts().reindex(
        ["literature_reported", "literature_supported_hypothesis", "biophysical_hypothesis", "context_only"],
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.bar(counts.index.str.replace("_", " "), counts.values, color="#0072B2")
    ax.set(ylabel="Hypotheses", title="Discovery hypothesis evidence classes")
    ax.tick_params(axis="x", rotation=25)
    sns.despine(ax=ax)
    _save_all_formats(fig, figure_dir / f"{gene.lower()}_discovery_evidence_classes")

    if gene.upper() == "BRAF":
        case_rows = frame.loc[frame["mutation"].eq("V600E")]
        case_name = "v600e_discovery_case"
    else:
        eligible = frame.loc[
            ~frame["structural_component_missing"].fillna(True).astype(bool)
        ].copy()
        exact = eligible.loc[eligible["literature_mechanism_specificity"].eq("exact_variant")]
        case_rows = (exact if len(exact) else eligible).nsmallest(1, "variant_priority_rank")
        case_name = "discovery_case"
    if case_rows.empty:
        return
    v = case_rows.iloc[0]
    case_variant = str(v["mutation"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.8]})
    components = [v[c] for c in ["structural_component", "mechanism_component",
                                  "disease_component", "biophysical_component"]]
    axes[0].barh(["Structure", "Mechanism", "Disease", "Biophysics"], components, color=colors)
    axes[0].set(xlim=(0, 35), xlabel="Priority points",
                title=f"{case_variant} total = {v['variant_priority_score']:.1f}")
    axes[1].axis("off")
    axes[1].text(0, 1, f"{case_variant} discovery case", weight="bold", va="top", fontsize=11)
    ddg_text = (f"RaSP ΔΔG: {v['rasp_ddg']:.2f} kcal/mol (global score {v['stability_score']:.1f})"
                if pd.notna(v.get("rasp_ddg")) else "RaSP ΔΔG: unavailable")
    axes[1].text(0, .84, ddg_text, va="top")
    axes[1].text(0, .68, f"Region: {v['functional_region']} · Δcharge {v['delta_charge']:+.1f} · Δhydropathy {v['delta_hydrophobicity']:+.1f}", va="top")
    literature = str(v.get("literature_mechanism", "none")).replace("_", " ")
    specificity = str(v.get("literature_mechanism_specificity", "none")).replace("_", " ")
    axes[1].text(0, .50, f"Literature: {literature} · {specificity}", va="top", color="#B2182B")
    axes[1].text(0, .34, f"Condition association: {v['disease_association_status']}", va="top")
    axes[1].text(0, .16, "Hypothesis-generating result; experimental validation required.", va="top", style="italic")
    _save_all_formats(fig, figure_dir / f"{gene.lower()}_{case_name}")


def resolve_retrieval_layer(index_manifest: Path) -> str | None:
    """Pick the layer mechanistic retrieval must use, from the built index itself.

    A curated index exposes ``primary_evidence`` and retrieval is bound to it.
    Only an index built before curation exposes the pre-layer collection, and
    that choice is recorded in the retrieval manifest rather than assumed.
    """
    manifest = json.loads(index_manifest.read_text(encoding="utf-8"))
    layers = manifest.get("indexed_layers", {})
    if EVIDENCE_LAYER in layers:
        return EVIDENCE_LAYER
    if "None" in layers or not layers:
        return None
    raise RuntimeError(
        f"Index at {index_manifest} has no {EVIDENCE_LAYER!r} collection; "
        f"available layers: {sorted(layers)}"
    )


def run(gene: str, limit: int | None = None, condition_profile: str | None = None,
        condition_terms: tuple[str, ...] = (), phase: str = "full") -> Path:
    gene = gene.upper()
    cfg, paths = get_gene_config(gene), paths_for(gene)
    condition_profile = condition_profile or str(cfg["discovery_profile"])
    if phase not in {"full", "retrieval", "inference"}:
        raise ValueError(f"Unknown phase: {phase}")
    get_condition_profile(condition_profile)
    if gene != "GSDMD" and condition_profile != "braf_oncology":
        raise ValueError("Neurological condition profiles are currently validated only for GSDMD")
    query_version = QUERY_VERSION if gene == "GSDMD" else "discovery_v5"
    index_manifest = paths["rag"] / f"{gene.lower()}_rag_index_manifest.json"
    if not index_manifest.exists():
        raise FileNotFoundError(f"RAG index manifest not found: {index_manifest}")
    index_fingerprint = hashlib.sha256(index_manifest.read_bytes()).hexdigest()[:12]
    stack_manifest = paths["rag"] / f"{gene.lower()}_rag_stack_manifest.json"
    stack_fingerprint = None
    stack_manifest_sha256 = None
    if stack_manifest.exists():
        stack_data = json.loads(stack_manifest.read_text(encoding="utf-8"))
        stack_fingerprint = stack_data.get("stack_fingerprint")
        stack_manifest_sha256 = hashlib.sha256(stack_manifest.read_bytes()).hexdigest()
    retrieval_layer = resolve_retrieval_layer(index_manifest)
    provisional_dir = task11_provisional_dir(paths["output"])
    provisional_scores = provisional_dir / "scores"
    provisional_figures = provisional_dir / "figures"
    provisional_scores.mkdir(parents=True, exist_ok=True)
    provisional_figures.mkdir(parents=True, exist_ok=True)
    structural_path = paths["scores"] / f"{gene.lower()}_structural_variant_ranking.parquet"
    structural = pd.read_parquet(structural_path)
    frame = annotate_discovery_hypotheses(
        structural, cfg["regions"], cfg.get("functional_motifs", {}), gene=gene
    )
    if limit:
        frame = frame.head(limit).copy()
    cache_dir = paths["rag"] / "inference_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    retrieval_cache_dir = paths["rag"] / "retrieval_cache"
    retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
    analyzer = DiscoveryLiteratureAnalyzer(gene, candidate_k=30, rag_dir=paths["rag"],
                                           layer=retrieval_layer)
    records = frame.to_dict("records")
    requests = {
        str(row["mutation"]): _discovery_request(
            row, gene, condition_profile, condition_terms, query_version
        ) for row in records
    }

    # Retrieval and LLM inference are deliberately separate phases.  Persisting
    # passages first lets MedCPT/Qdrant be released before Ollama loads an 8B
    # model, preventing avoidable WSL memory pressure and making both phases resumable.
    retrieved: dict[str, tuple[list[LiteratureEvidence], list[str]]] = {}
    retrieval_errors: dict[str, str] = {}
    pending = [row for row in records if not _cache_path(
        cache_dir, str(row["mutation"]), condition_profile, condition_terms, query_version,
        index_fingerprint,
    ).exists()]
    retrieval_rows = pending if phase in {"full", "retrieval"} else []
    for index, row in enumerate(retrieval_rows, 1):
        variant = str(row["mutation"])
        retrieval_cache = _retrieval_cache_path(
            retrieval_cache_dir, variant, condition_profile, condition_terms, query_version,
            index_fingerprint,
        )
        try:
            if retrieval_cache.exists():
                payload = json.loads(retrieval_cache.read_text(encoding="utf-8"))
                queries = [str(value) for value in payload["retrieval_queries"]]
                evidence = [LiteratureEvidence.model_validate(item) for item in payload["evidence"]]
            else:
                channel_queries = analyzer._channel_queries(requests[variant])
                queries = [q for items in channel_queries.values() for q in items]
                evidence = analyzer.retrieve_discovery(
                    channel_queries, requests[variant].k, variant=variant,
                    functional_region=requests[variant].functional_region,
                    reranker_query=(f"{gene} {variant} mechanistic question: "
                                    "Does the evidence directly support a mechanistic effect?"),
                )
                retrieval_cache.write_text(json.dumps({
                    "variant": variant, "gene": gene,
                    "retrieval_version": analyzer.retrieval_version,
                    "index_fingerprint": index_fingerprint,
                    "stack_fingerprint": stack_fingerprint,
                    "stack_manifest_sha256": stack_manifest_sha256,
                    "query_version": query_version, "condition_profile": condition_profile,
                    "condition_terms": list(condition_terms), "retrieval_queries": queries,
                    "retrieval_channels": {channel: list(items)
                                           for channel, items in channel_queries.items()},
                    # PMID distribution, document size and lexical/dense split,
                    # persisted per variant so retrieval bias stays auditable.
                    "retrieval_diagnostics": analyzer.last_retrieval_diagnostics,
                    "evidence": [item.model_dump() for item in evidence],
                    "mechanistic_evidence": [
                        item.model_dump()
                        for item in analyzer.last_retrieval_blocks.get("mechanistic_evidence", [])
                    ],
                    "context_reference": [
                        item.model_dump()
                        for item in analyzer.last_retrieval_blocks.get("context_reference", [])
                    ],
                    "mechanistic_evidence_status": (
                        "mechanistic_evidence_retrieved"
                        if analyzer.last_retrieval_blocks.get("mechanistic_evidence") else
                        "no_mechanistic_evidence_retrieved"
                    ),
                }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            retrieved[variant] = (evidence, queries)
            print(f"Discovery retrieval {index}/{len(retrieval_rows)}: {variant} -> {len(evidence)} passages")
        except Exception as error:
            retrieval_errors[variant] = f"{type(error).__name__}: {error}"
            print(f"Discovery retrieval {index}/{len(retrieval_rows)}: {variant} -> failed")
    analyzer.release_retrieval_models()

    if phase == "retrieval":
        retrieval_manifest = paths["rag"] / f"{gene.lower()}_retrieval_phase_manifest.json"
        retrieval_manifest.write_text(json.dumps({
            "gene": gene, "condition_profile": condition_profile,
            "query_version": query_version, "index_fingerprint": index_fingerprint,
            "retrieval_stack_version": RETRIEVAL_STACK_VERSION,
            "retrieval_stack_manifest": (
                str(stack_manifest.relative_to(paths["rag"])) if stack_manifest.exists() else None
            ),
            "retrieval_stack_manifest_sha256": (
                stack_manifest_sha256
            ),
            "retrieval_stack_fingerprint": stack_fingerprint,
            "variants_total": len(records),
            "retrieval_caches_available": sum(
                _retrieval_cache_path(
                    retrieval_cache_dir, str(row["mutation"]), condition_profile,
                    condition_terms, query_version, index_fingerprint,
                ).exists() for row in records
            ),
            "retrieval_failures": retrieval_errors,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return retrieval_manifest

    if phase == "inference":
        for row in pending:
            variant = str(row["mutation"])
            retrieval_cache = _retrieval_cache_path(
                retrieval_cache_dir, variant, condition_profile, condition_terms,
                query_version, index_fingerprint,
            )
            try:
                payload = json.loads(retrieval_cache.read_text(encoding="utf-8"))
                retrieved[variant] = (
                    [LiteratureEvidence.model_validate(item) for item in payload["evidence"]],
                    [str(value) for value in payload["retrieval_queries"]],
                )
            except Exception as error:
                retrieval_errors[variant] = f"{type(error).__name__}: {error}"

    literature_rows = []
    for row in records:
        variant = row["mutation"]
        cache = _cache_path(
            cache_dir, variant, condition_profile, condition_terms, query_version,
            index_fingerprint,
        )
        request = requests[str(variant)]
        try:
            if cache.exists():
                result = DiscoveryRAGResult.model_validate_json(cache.read_text(encoding="utf-8"))
            else:
                if str(variant) in retrieval_errors:
                    raise RuntimeError(retrieval_errors[str(variant)])
                evidence, queries = retrieved[str(variant)]
                result = analyzer.infer_discovery(
                    request, evidence=evidence, retrieval_queries=queries
                )
            # Migrate retrieval-channel provenance into old inference caches
            # without rerunning retrieval or the LLM.
            retrieval_cache = _retrieval_cache_path(
                retrieval_cache_dir, str(variant), condition_profile, condition_terms,
                query_version, index_fingerprint,
            )
            if retrieval_cache.exists() and result.evidence and not any(
                item.retrieval_channels for item in result.evidence
            ):
                retrieval_payload = json.loads(retrieval_cache.read_text(encoding="utf-8"))
                selected_channels = retrieval_payload.get("retrieval_diagnostics", {}).get(
                    "channels_per_selected_document", {}
                )
                result = result.model_copy(update={
                    "evidence": [item.model_copy(update={
                        "retrieval_channels": list(selected_channels.get(str(item.pmid), []))
                    }) for item in result.evidence]
                })
            result = _normalize_generation_modes(result).model_copy(update={
                "retrieval_index_fingerprint": index_fingerprint,
                "retrieval_stack_fingerprint": stack_fingerprint,
                "retrieval_stack_manifest_sha256": stack_manifest_sha256,
            })
            # Re-serializing an existing cache only adds provenance fields and
            # generation modes; it never reruns retrieval or inference.
            cache.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            inference, evidence = result.inference, result.evidence
            literature_hypotheses = [h for h in inference.hypotheses if h.classification.startswith("literature")]
            best = literature_hypotheses[0] if literature_hypotheses else None
            best_ids = best.cited_evidence_ids if best else []
            disease_ids = inference.disease_association.cited_evidence_ids
            therapy = (_exact_variant_therapy_context(variant, evidence) if gene == "BRAF" else
                       {"therapy_evidence_status": "not_evaluated", "therapy_hypothesis": "",
                        "therapy_cited_evidence_ids": "", "therapy_source_type": "none"})
            literature_rows.append({
                "mutation": variant, "discovery_pipeline_status": "complete",
                "llm_contract_status": result.llm_contract_status,
                "inference_status": result.inference_status,
                "evidence_status": result.evidence_status,
                "mechanistic_evidence_status": result.mechanistic_evidence_status,
                "mechanistic_evidence_count": len(result.mechanistic_evidence),
                "context_reference_count": len(result.context_reference),
                "llm_retry_count": result.llm_retry_count,
                "llm_failure_reason": result.llm_failure_reason,
                "generation_mode": ";".join(sorted({
                    hypothesis.generation_mode for hypothesis in inference.hypotheses
                })) or "none",
                "disease_generation_mode": (
                    inference.disease_association.generation_mode
                    if inference.disease_association.status != "none" else "none"
                ),
                "disease_association_status": inference.disease_association.status,
                "disease_context": inference.disease_association.disease_context,
                "disease_association_rationale": inference.disease_association.rationale,
                "disease_cited_evidence_ids": ";".join(disease_ids),
                "disease_association_source_type": _source_type(evidence, disease_ids),
                "condition_evidence_tier": _condition_tier(condition_profile, evidence, disease_ids),
                "literature_mechanism_specificity": _specificity(
                    variant, evidence, best_ids, str(row["functional_region"])) if best else "none",
                "literature_mechanism_source_type": _source_type(evidence, best_ids),
                "literature_mechanism": best.mechanism if best else "none",
                "literature_predicted_direction": best.predicted_direction if best else "unclear",
                "literature_conformational_state": best.conformational_state if best else "not_specified",
                "literature_cited_evidence_ids": ";".join(best_ids),
                "evidence_conflict_flag": inference.evidence_conflict_flag or any(h.evidence_conflict for h in inference.hypotheses),
                "discovery_hypotheses": json.dumps([h.model_dump() for h in inference.hypotheses], ensure_ascii=False),
                "discovery_hypothesis_count": len(inference.hypotheses),
                "discovery_uncertainty": inference.uncertainty,
                "retrieved_pmids": ";".join(dict.fromkeys(e.pmid for e in evidence)),
                "retrieved_source_types": ";".join(dict.fromkeys(e.source_type for e in evidence)),
                "retrieved_source_collections": ";".join(dict.fromkeys(e.source_collection for e in evidence)),
                "retrieved_article_titles": " || ".join(dict.fromkeys(e.article_title or "" for e in evidence)),
                "retrieved_passages": " || ".join(f"{e.evidence_id}:{e.passage[:650].replace(chr(10), ' ')}" for e in evidence),
                "retrieval_crossencoder_scores": ";".join(f"{e.reranker_score:.5g}" for e in evidence),
                "retrieval_queries": " || ".join(result.retrieval_queries),
                "retrieval_version": result.retrieval_version,
                "retrieval_stack_version": RETRIEVAL_STACK_VERSION,
                "retrieval_index_fingerprint": index_fingerprint,
                "retrieval_stack_fingerprint": stack_fingerprint,
                "literature_schema_version": "task11_claim_evidence_v1",
                "condition_profile": result.condition_profile,
                "condition_terms": ";".join(result.condition_terms),
                "query_version": result.query_version,
                "llm_model": llm_model_name(), "llm_prompt_version": DISCOVERY_PROMPT_VERSION,
                **therapy,
            })
        except Exception as error:
            literature_rows.append({"mutation": variant, "discovery_pipeline_status": "failed",
                                    "discovery_uncertainty": f"{type(error).__name__}: {error}"})
        print(f"Discovery inference {len(literature_rows)}/{len(frame)}: {variant} -> "
              f"{literature_rows[-1]['discovery_pipeline_status']}")
    detailed = frame.merge(pd.DataFrame(literature_rows), on="mutation", how="left", validate="one_to_one")
    detailed["retrieval_stack_version"] = RETRIEVAL_STACK_VERSION
    detailed["retrieval_index_fingerprint"] = index_fingerprint
    detailed["retrieval_stack_fingerprint"] = stack_fingerprint
    # This table is deliberately provisional.  Only the explicit promotion
    # command may copy it into the canonical production namespace.
    detailed["rag_promotion_status"] = "blocked"
    detailed["claim_grades_provisional"] = True
    detailed["eligible_for_v2"] = False
    detailed = add_discovery_priority(detailed).sort_values(["variant_priority_rank", "stability_rank", "mutation"])
    rendered = detailed.apply(render_mechanistic_hypothesis, axis=1)
    detailed["mechanistic_hypothesis"] = rendered.map(lambda item: item[0])
    detailed["mechanistic_hypothesis_evidence_ids"] = rendered.map(lambda item: item[1])
    output = provisional_scores / f"{gene.lower()}_evidence_adjusted_variant_priority.csv"
    detailed.to_csv(output, index=False)
    detailed.to_parquet(output.with_suffix(".parquet"), index=False)
    review_columns = [
        "mutation", "variant_priority_rank", "variant_priority_score", "structural_component",
        "mechanism_component", "disease_component", "biophysical_component", "stability_rank",
        "stability_score", "rasp_ddg", "structural_component_missing", "functional_region",
        "secondary_structure", "mechanistic_flags", "biophysical_rule_ids", "discovery_hypotheses",
        "mechanistic_hypothesis", "mechanistic_hypothesis_evidence_ids",
        "therapy_evidence_status", "therapy_hypothesis", "therapy_cited_evidence_ids", "therapy_source_type",
        "disease_association_status", "disease_context", "literature_mechanism",
        "condition_evidence_tier",
        "literature_predicted_direction", "literature_conformational_state", "evidence_conflict_flag",
        "literature_cited_evidence_ids", "disease_cited_evidence_ids", "retrieved_pmids",
        "retrieved_source_types", "retrieved_source_collections", "retrieved_passages",
        "llm_contract_status", "discovery_uncertainty",
        "inference_status", "evidence_status", "llm_retry_count", "llm_failure_reason",
        "generation_mode", "disease_generation_mode", "rag_promotion_status",
        "claim_grades_provisional", "eligible_for_v2",
        "condition_profile", "condition_terms", "query_version", "retrieval_queries",
    ]
    review = detailed[[c for c in review_columns if c in detailed]]
    review_output = output.with_name(f"{gene.lower()}_variant_priority_rag_review_table.csv")
    review.to_csv(review_output, index=False)
    review.to_parquet(review_output.with_suffix(".parquet"), index=False)
    presentation_columns = [
        "mutation", "variant_priority_rank", "variant_priority_score", "structural_component",
        "mechanism_component", "disease_component", "biophysical_component",
        "mechanistic_hypothesis", "mechanistic_hypothesis_evidence_ids",
        "rasp_ddg", "apbs_delta_phi_local_p95_abs_kbt_e", "encom_relative_entropy_change",
        "encom_delta_vibrational_entropy", "backbone_ca_displacement_a",
        "functional_region", "secondary_structure_segment", "ppi_partner", "ppi_source_complex",
        "disease_association_status", "literature_mechanism",
        "condition_evidence_tier",
        "therapy_evidence_status", "therapy_hypothesis", "therapy_cited_evidence_ids", "therapy_source_type",
        "literature_cited_evidence_ids", "disease_cited_evidence_ids", "evidence_conflict_flag",
        "discovery_uncertainty",
        "inference_status", "evidence_status", "llm_retry_count", "llm_failure_reason",
        "generation_mode", "disease_generation_mode", "rag_promotion_status",
        "claim_grades_provisional", "eligible_for_v2",
        "condition_profile", "condition_terms", "query_version",
    ]
    presentation = detailed[[c for c in presentation_columns if c in detailed]]
    presentation_output = output.with_name(f"{gene.lower()}_variant_priority_presentation_table.csv")
    presentation.to_csv(presentation_output, index=False)
    presentation.to_parquet(presentation_output.with_suffix(".parquet"), index=False)
    rank_audit = detailed[[
        "mutation", "stability_rank", "stability_score", "variant_priority_rank",
        "variant_priority_score", "structural_component", "mechanism_component",
        "disease_component", "biophysical_component", "structural_component_missing",
    ]].copy()
    rank_audit["rank_change_discovery_minus_structural"] = (
        rank_audit["variant_priority_rank"] - rank_audit["stability_rank"]
    )
    rank_audit_output = output.with_name(f"{gene.lower()}_discovery_rank_change_audit.csv")
    rank_audit.to_csv(rank_audit_output, index=False)
    rank_audit.to_parquet(rank_audit_output.with_suffix(".parquet"), index=False)
    _save_figures(detailed, provisional_figures, gene.upper())
    if gene == "BRAF":
        case = detailed.loc[detailed["mutation"].eq("V600E")]
    else:
        structurally_covered = detailed.loc[
            ~detailed["structural_component_missing"].fillna(True).astype(bool)
        ]
        exact = structurally_covered.loc[
            structurally_covered["literature_mechanism_specificity"].eq("exact_variant")
        ]
        case = (exact if len(exact) else structurally_covered).nsmallest(1, "variant_priority_rank")
    summary = {
        "gene": gene.upper(), "mode": "discovery", "retrieval_version": analyzer.retrieval_version,
        "condition_profile": condition_profile, "condition_terms": list(condition_terms),
        "query_version": query_version,
        # Which curated layer mechanistic retrieval was actually bound to.
        "retrieval_corpus_layer": retrieval_layer or "unlayered_legacy_index",
        "retrieval_index_fingerprint": index_fingerprint,
        "retrieval_stack_fingerprint": stack_fingerprint,
        "retrieval_stack_version": RETRIEVAL_STACK_VERSION,
        "literature_schema_version": "task11_claim_evidence_v1",
        "legacy_literature_columns_replaced": True,
        "rag_promotion_status": "blocked",
        "claim_grades_provisional": True,
        "eligible_for_v2": False,
        "variants": len(detailed), "priority_scores_available": int(detailed["variant_priority_score"].notna().sum()),
        "structural_scores_available": int(detailed["stability_score"].notna().sum()),
        "technical_failures": int(detailed["discovery_pipeline_status"].eq("failed").sum()),
        "llm_contract_status_counts": {str(k): int(v) for k, v in detailed["llm_contract_status"].value_counts(dropna=False).items()},
        "inference_status_counts": {str(k): int(v) for k, v in detailed["inference_status"].value_counts(dropna=False).items()},
        "evidence_status_counts": {str(k): int(v) for k, v in detailed["evidence_status"].value_counts(dropna=False).items()},
        "llm_failure_records": int(detailed["llm_failure_reason"].fillna("").astype(str).str.strip().ne("").sum()),
        "inference_generation_mode_counts": {
            "llm_validated": int(detailed["llm_contract_status"].eq("passed").sum()),
            "deterministic": int(detailed["llm_contract_status"].eq("deterministic_fallback").sum()),
        },
        "generation_mode_counts": {str(k): int(v) for k, v in
                                    detailed["generation_mode"].value_counts(dropna=False).items()},
        "disease_generation_mode_counts": {str(k): int(v) for k, v in
                                            detailed["disease_generation_mode"].value_counts(dropna=False).items()},
        "disease_association_counts": {str(k): int(v) for k, v in detailed["disease_association_status"].value_counts(dropna=False).items()},
        "mechanism_specificity_counts": {str(k): int(v) for k, v in detailed["literature_mechanism_specificity"].value_counts(dropna=False).items()},
        "component_caps": {"structural": 35, "mechanism": 35, "disease": 10, "biophysical": 20},
        "clinvar_used_as_score_input": False, "human_review_table": str(review_output),
        "presentation_table": str(presentation_output),
        "rank_change_audit_table": str(rank_audit_output),
        "case_study": (case[["mutation", "variant_priority_score", "stability_score", "mechanism_component",
                              "disease_component", "biophysical_component", "literature_conformational_state"]]
                       .iloc[0].to_dict() if len(case) else None),
        "scientific_boundary": "High-sensitivity hypothesis generation; not causal or clinical evidence.",
    }
    if gene == "BRAF":
        summary["v600e_acceptance"] = (
            case[["variant_priority_score", "stability_score", "mechanism_component",
                  "disease_component", "biophysical_component", "literature_conformational_state"]]
            .iloc[0].to_dict() if len(case) else None
        )
    output.with_name(f"{gene.lower()}_evidence_adjusted_variant_priority_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--condition-profile")
    parser.add_argument("--condition-term", action="append", default=[])
    parser.add_argument("--phase", choices=("full", "retrieval", "inference"), default="full")
    args = parser.parse_args()
    print(run(args.gene, args.limit, args.condition_profile, tuple(args.condition_term), args.phase))
