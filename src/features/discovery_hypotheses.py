"""Deterministic, explicitly hypothetical biophysical discovery annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BiophysicalHypothesis:
    mechanism: str
    target: str
    predicted_direction: str
    rationale: str
    rule_ids: tuple[str, ...]
    classification: str = "biophysical_hypothesis"
    requires_experimental_validation: bool = True


def _functional_context(position: int, regions: dict, motifs: dict) -> tuple[str, list[str]]:
    labels = [name for name, (start, end) in motifs.items() if start <= position <= end]
    labels += [name for name, (start, end) in regions.items() if start <= position <= end]
    # Motifs are more informative and occur first; retain stable unique order.
    return (labels[0] if labels else "unannotated_region", list(dict.fromkeys(labels)))


def annotate_discovery_hypotheses(frame: pd.DataFrame, regions: dict, motifs: dict,
                                  gene: str = "BRAF") -> pd.DataFrame:
    """Add capped 0--20 component and up to three rule-derived hypotheses.

    These rules indicate plausible experimental targets. They do not assert that
    the mutation has the predicted effect or direction.
    """
    gene = gene.upper()
    out = frame.copy()
    out["gene"] = gene
    rows = []
    for row in out.to_dict("records"):
        pos, wt, mut = int(row["pos"]), str(row["wt_aa"]), str(row["mut_aa"])
        dc = abs(float(row.get("delta_charge", 0) or 0))
        dv = abs(float(row.get("delta_volume", 0) or 0))
        dh = abs(float(row.get("delta_hydrophobicity", 0) or 0))
        chemical_rules, chemical_points = [], 0.0
        if dc >= 1:
            chemical_rules.append("charge_shift_ge_1")
            chemical_points += 3
        if dv >= 50:
            chemical_rules.append("volume_shift_ge_50A3")
            chemical_points += 3
        if dh >= 3:
            chemical_rules.append("hydrophobicity_shift_ge_3")
            chemical_points += 3
        if mut in {"P", "G", "C"} or wt in {"P", "G", "C"}:
            chemical_rules.append("special_residue_pro_gly_cys_change")
            chemical_points += 3
        chemical_points = min(10.0, chemical_points)

        primary_region, contexts = _functional_context(pos, regions, motifs)
        context_rules, context_points = [], 0.0
        if primary_region != "unannotated_region":
            context_rules.append(f"functional_region:{primary_region}")
            context_points += 3
        if bool(row.get("flag_ppi_interface_context", False)):
            context_rules.append("audited_ppi_interface")
            context_points += 4
        if bool(row.get("flag_ppi_hotspot_candidate", False)):
            context_rules.append("ppi_hotspot_proxy")
            context_points += 2
        if bool(row.get("flag_active_site_context", False)) or bool(row.get("flag_binding_site_context", False)):
            context_rules.append("active_or_binding_site_context")
            context_points += 3
        if bool(row.get("flag_phosphosite_context", False)):
            context_rules.append("ptm_or_near_phosphosite")
            context_points += 3
        if bool(row.get("flag_allosteric_high_coupling", False)):
            context_rules.append("high_allosteric_coupling")
            context_points += 2
        secondary = str(row.get("secondary_structure", "unknown"))
        if secondary not in {"unknown", "nan", "None", ""}:
            context_rules.append(f"secondary_structure:{secondary}")
            context_points += 2
        context_points = min(10.0, context_points)

        hypotheses: list[BiophysicalHypothesis] = []
        perturbation = chemical_rules or ["amino_acid_substitution"]
        if gene == "BRAF" and primary_region in {
            "activation_segment", "dfg_motif", "alpha_c_helix", "p_loop", "hrd_motif"
        }:
            hypotheses.append(BiophysicalHypothesis(
                mechanism="conformational_equilibrium",
                target=f"BRAF kinase {primary_region}", predicted_direction="alteration_possible",
                rationale=(f"A physicochemical substitution in the {primary_region} may shift the "
                           "relative stability of inactive and active kinase conformations."),
                rule_ids=tuple(perturbation + [f"functional_region:{primary_region}"]),
            ))
        if gene == "GSDMD" and primary_region in {
            "caspase_3_7_inactivating_site", "inflammatory_caspase_cleavage_region", "interdomain_linker"
        }:
            hypotheses.append(BiophysicalHypothesis(
                mechanism="proteolytic_activation_or_inactivation",
                target=f"GSDMD {primary_region}", predicted_direction="cleavage_regulation_alteration_possible",
                rationale=(f"A physicochemical substitution in the {primary_region} may alter local "
                           "recognition, accessibility or processing by a regulatory protease."),
                rule_ids=tuple(perturbation + [f"functional_region:{primary_region}"]),
            ))
        if gene == "GSDMD" and primary_region in {
            "membrane_insertion_hairpins", "pore_forming_n_terminal_domain", "cys191_regulatory_site"
        }:
            hypotheses.append(BiophysicalHypothesis(
                mechanism="membrane_pore_formation",
                target=f"GSDMD {primary_region}", predicted_direction="membrane_or_pore_activity_alteration_possible",
                rationale=(f"A physicochemical substitution in the {primary_region} may alter membrane engagement, "
                           "the prepore-to-pore rearrangement or oligomeric pore stability."),
                rule_ids=tuple(perturbation + [f"functional_region:{primary_region}"]),
            ))
        if gene == "GSDMD" and primary_region == "autoinhibitory_c_terminal_domain":
            hypotheses.append(BiophysicalHypothesis(
                mechanism="autoinhibitory_domain_coupling", target="GSDMD N-terminal/C-terminal coupling",
                predicted_direction="autoinhibition_alteration_possible",
                rationale="A physicochemical substitution in the C-terminal domain may alter packing that restrains the pore-forming domain.",
                rule_ids=tuple(perturbation + ["functional_region:autoinhibitory_c_terminal_domain"]),
            ))
        if bool(row.get("flag_ppi_interface_context", False)):
            hypotheses.append(BiophysicalHypothesis(
                mechanism="protein_protein_interaction", target=str(row.get("ppi_partner") or f"{gene} partner"),
                predicted_direction="affinity_or_geometry_alteration_possible",
                rationale="A physicochemical change at an audited interface may alter contact geometry or affinity.",
                rule_ids=tuple(perturbation + ["audited_ppi_interface"]),
            ))
        if bool(row.get("flag_buried_residue", False)) or bool(row.get("flag_packing_disruption_candidate", False)):
            hypotheses.append(BiophysicalHypothesis(
                mechanism="local_packing_or_stability", target=primary_region,
                predicted_direction="local_destabilization_possible",
                rationale="A substitution at a buried position may perturb packing even when global scores are modest.",
                rule_ids=tuple(perturbation + ["buried_residue"]),
            ))
        if bool(row.get("flag_active_site_context", False)) or bool(row.get("flag_binding_site_context", False)):
            hypotheses.append(BiophysicalHypothesis(
                mechanism="catalysis_or_ligand_binding", target=primary_region,
                predicted_direction="activity_or_binding_alteration_possible",
                rationale="Proximity to an active or binding site makes altered catalysis or ligand recognition plausible.",
                rule_ids=tuple(perturbation + ["active_or_binding_site_context"]),
            ))
        if bool(row.get("flag_phosphosite_context", False)):
            hypotheses.append(BiophysicalHypothesis(
                mechanism="post_translational_regulation", target=primary_region,
                predicted_direction="regulatory_effect_possible",
                rationale="The substitution is at or near an annotated modification site and may alter regulatory recognition.",
                rule_ids=tuple(perturbation + ["ptm_or_near_phosphosite"]),
            ))
        if bool(row.get("flag_allosteric_high_coupling", False)):
            hypotheses.append(BiophysicalHypothesis(
                mechanism="allosteric_coupling", target=primary_region,
                predicted_direction="signal_propagation_alteration_possible",
                rationale="High modeled coupling makes altered propagation of a local perturbation a testable possibility.",
                rule_ids=tuple(perturbation + ["high_allosteric_coupling"]),
            ))
        if not hypotheses and chemical_rules:
            hypotheses.append(BiophysicalHypothesis(
                mechanism="local_physicochemical_environment", target=primary_region,
                predicted_direction="local_effect_possible",
                rationale="The substitution produces a large physicochemical change worth experimental follow-up.",
                rule_ids=tuple(chemical_rules),
            ))
        hypotheses = hypotheses[:3]
        rows.append({
            "functional_region": primary_region,
            "functional_contexts": ";".join(contexts),
            "biophysical_perturbation_component": chemical_points,
            "biophysical_context_component": context_points,
            "biophysical_component": chemical_points + context_points,
            "biophysical_rule_ids": ";".join(dict.fromkeys(chemical_rules + context_rules)),
            "biophysical_hypotheses": json.dumps([asdict(item) for item in hypotheses], ensure_ascii=False),
            "biophysical_hypothesis_count": len(hypotheses),
        })
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
