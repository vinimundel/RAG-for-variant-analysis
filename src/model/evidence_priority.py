"""Evidence-aware prioritization kept explicitly separate from structural severity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_evidence_priority(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    verdict = out.get("literature_verdict", pd.Series("insufficient_evidence", index=out.index))
    specificity = out.get("literature_specificity", pd.Series("none", index=out.index))
    modifier = pd.Series(0.0, index=out.index)
    supported = verdict.eq("supported")
    modifier.loc[supported & specificity.eq("exact_variant")] = 20.0
    modifier.loc[supported & specificity.eq("same_residue")] = 12.0
    modifier.loc[supported & specificity.eq("regional")] = 7.5
    modifier.loc[verdict.eq("contradicted")] = -10.0
    structural = pd.to_numeric(out.get("stability_score", np.nan), errors="coerce")
    out["literature_priority_modifier"] = modifier
    priority = (structural.fillna(0) + modifier).clip(0, 100)
    # No structural estimate and no evidence modifier is unknown, not a score of 0.
    priority = priority.where(structural.notna() | modifier.ne(0))
    out["evidence_adjusted_priority_score"] = priority
    out["priority_has_structural_score"] = structural.notna()
    out["evidence_adjusted_priority_rank"] = out["evidence_adjusted_priority_score"].rank(
        method="min", ascending=False
    ).astype("Int64")
    out["priority_score_definition"] = (
        "clip(stability_score_if_available_else_0 + literature_modifier,0,100);"
        "NA_when_no_structural_score_and_modifier_zero;"
        "modifier=+20 exact_variant,+12 same_residue,+7.5 regional,-10 contradicted,0 insufficient;"
        "research_priority_not_clinical_probability"
    )
    return out


def add_discovery_priority(frame: pd.DataFrame) -> pd.DataFrame:
    """Compose the declared 35/35/10/20 exploratory variant priority score."""
    out = frame.copy()
    stability = pd.to_numeric(out.get("stability_score", np.nan), errors="coerce")
    existing_missing = out["structural_component_missing"].astype(bool) if "structural_component_missing" in out else pd.Series(False, index=out.index)
    out["structural_component_missing"] = existing_missing | stability.isna()
    if "structural_component" not in out:
        out["structural_component"] = stability.fillna(0).clip(0, 100) * 0.35
    else:
        out["structural_component"] = pd.to_numeric(out["structural_component"], errors="coerce").fillna(0).clip(0, 35)

    specificity = out.get("literature_mechanism_specificity", pd.Series("none", index=out.index))
    source = out.get("literature_mechanism_source_type", pd.Series("none", index=out.index))
    mechanism = pd.Series(0.0, index=out.index)
    full = source.eq("grobid_full_text")
    abstract = source.eq("pubmed_abstract")
    for label, full_score, abstract_score in (
        # Only a lexical exact-variant result can materially affect discovery
        # priority.  Same-residue and regional results remain visible as
        # analogies/context but are deliberately capped at negligible values.
        ("exact_variant", 35.0, 30.0), ("same_residue", 3.0, 2.0),
        # A configured, explicitly annotated functional region remains a
        # contextual mechanism axis; same-domain/gene-level/general context
        # is normalized to ``general`` and contributes zero.
        ("functional_region", 18.0, 15.0), ("general", 0.0, 0.0),
    ):
        mechanism.loc[specificity.eq(label) & full] = full_score
        mechanism.loc[specificity.eq(label) & abstract] = abstract_score
    out["mechanism_component"] = mechanism.clip(0, 35)

    disease_status = out.get("disease_association_status", pd.Series("none", index=out.index))
    disease_source = out.get("disease_association_source_type", pd.Series("none", index=out.index))
    disease = pd.Series(0.0, index=out.index)
    disease.loc[disease_status.eq("associated") & disease_source.eq("grobid_full_text")] = 10.0
    disease.loc[disease_status.eq("associated") & disease_source.eq("pubmed_abstract")] = 8.0
    disease.loc[disease_status.eq("cooccurrence") & disease_source.eq("grobid_full_text")] = 5.0
    disease.loc[disease_status.eq("cooccurrence") & disease_source.eq("pubmed_abstract")] = 4.0
    out["disease_component"] = disease.clip(0, 10)
    out["biophysical_component"] = pd.to_numeric(
        out.get("biophysical_component", 0), errors="coerce"
    ).fillna(0).clip(0, 20)
    out["variant_priority_score"] = out[[
        "structural_component", "mechanism_component", "disease_component", "biophysical_component"
    ]].sum(axis=1).clip(0, 100)
    out["variant_priority_rank"] = out["variant_priority_score"].rank(
        method="min", ascending=False
    ).astype("Int64")
    out["priority_score_definition"] = (
        "discovery_v3:structural<=35+exact_variant_literature<=35+indirect_analogy<=3+"
        "regional_context<=18+general_context=0+exact_variant_disease<=10+"
        "deterministic_biophysical_context<=20;hypothesis_generation_not_clinical_probability"
    )
    return out
