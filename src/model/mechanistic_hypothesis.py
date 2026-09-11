"""Deterministic, evidence-aware rendering of one presentation hypothesis."""

from __future__ import annotations

import json
import re

import pandas as pd

from src.features.physicochemical import PhysicochemicalEngine


_AA_NAME = {
    "A": "alanine", "R": "arginine", "N": "asparagine", "D": "aspartate",
    "C": "cysteine", "Q": "glutamine", "E": "glutamate", "G": "glycine",
    "H": "histidine", "I": "isoleucine", "L": "leucine", "K": "lysine",
    "M": "methionine", "F": "phenylalanine", "P": "proline", "S": "serine",
    "T": "threonine", "W": "tryptophan", "Y": "tyrosine", "V": "valine",
}


_REGION_CONTEXT = {
    "p_loop": (
        "the glycine-rich P-loop (residues 464–471)",
        "shapes the ATP-phosphate pocket and couples nucleotide positioning to kinase conformation",
    ),
    "alpha_c_helix": (
        "the αC helix (residues 491–505)",
        "controls the αC-in/αC-out switch and the catalytic Lys483–Glu501 ion pair",
    ),
    "hrd_motif": (
        "the HRD catalytic-loop motif (residues 574–576)",
        "organizes catalytic-loop geometry and activation-segment coupling",
    ),
    "dfg_motif": (
        "the DFG motif at the start of the activation segment (residues 594–596)",
        "coordinates the Mg/ATP environment and distinguishes DFG-in from DFG-out kinase states",
    ),
    "activation_segment": (
        "the activation segment (residues 594–623)",
        "controls substrate access and the equilibrium between inactive and active kinase states",
    ),
    "ras_binding_domain": (
        "the RAS-binding domain (residues 155–227)",
        "mediates recruitment of BRAF by activated RAS",
    ),
    "cysteine_rich_domain": (
        "the cysteine-rich domain (residues 234–280)",
        "contributes to RAS/membrane engagement and BRAF regulatory architecture",
    ),
    "serine_rich_hinge": (
        "the serine-rich regulatory linker (residues 281–456)",
        "connects the N-terminal regulatory modules to the kinase domain",
    ),
    "c_terminal_tail": (
        "the C-terminal regulatory tail (residues 718–766)",
        "contains the phospho-dependent 14-3-3 regulatory region",
    ),
}

_GSDMD_REGION_CONTEXT = {
    "pore_forming_n_terminal_domain": (
        "the pore-forming N-terminal domain (residues 1–275)",
        "binds acidic membrane lipids and oligomerizes after proteolytic release",
    ),
    "interdomain_linker": (
        "the interdomain linker (approximately residues 250–285)",
        "couples proteolytic processing to release of the pore-forming fragment",
    ),
    "autoinhibitory_c_terminal_domain": (
        "the autoinhibitory C-terminal domain (residues 276–484)",
        "restrains the N-terminal domain in full-length GSDMD",
    ),
    "caspase_3_7_inactivating_site": (
        "the Asp87-centered inactivating cleavage region",
        "can redirect processing toward loss of pyroptotic pore-forming activity",
    ),
    "inflammatory_caspase_cleavage_region": (
        "the inflammatory-caspase cleavage region near Asp275",
        "controls liberation of the pore-forming N-terminal fragment",
    ),
    "membrane_insertion_hairpins": (
        "the N-terminal membrane-insertion hairpin region",
        "undergoes a large rearrangement during the prepore-to-pore transition",
    ),
    "cys191_regulatory_site": (
        "the Cys191 regulatory site",
        "links local cysteine chemistry to GSDMD pore-forming regulation",
    ),
}


def _present(value) -> bool:
    try:
        return value is not None and not pd.isna(value)
    except (TypeError, ValueError):
        return value is not None


def _json_list(value) -> list[dict]:
    if not _present(value):
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _secondary_element(row: pd.Series) -> str:
    kind = str(row.get("secondary_structure", "unknown"))
    start, end = row.get("secondary_structure_segment_start"), row.get("secondary_structure_segment_end")
    labels = {"helix": "α-helix", "sheet": "β-strand", "loop": "loop"}
    if kind in labels and _present(start) and _present(end):
        return f"the experimentally assigned {labels[kind]} spanning residues {int(start)}–{int(end)}"
    return ""


def _site_context(row: pd.Series) -> tuple[str, str]:
    region = str(row.get("functional_region", "unannotated_region"))
    gene = str(row.get("gene", "BRAF")).upper()
    if gene == "GSDMD" and region in _GSDMD_REGION_CONTEXT:
        return _GSDMD_REGION_CONTEXT[region]
    if region in _REGION_CONTEXT:
        return _REGION_CONTEXT[region]
    element = _secondary_element(row)
    if element:
        return element, f"forms part of the experimentally observed {gene} structural scaffold"
    return f"the annotated {gene} region", "provides local structural context for the substitution"


def _signed_change(value: float, increase: str, decrease: str, unchanged: str,
                   unit: str = "") -> str:
    direction = increase if value > 0 else decrease if value < 0 else "does not change"
    magnitude = f"{abs(value):.2f}{unit}"
    return f"{direction} by {magnitude}" if value else unchanged


def _substitution(row: pd.Series) -> str:
    wt, mut = str(row.get("wt_aa", "")).upper(), str(row.get("mut_aa", "")).upper()
    dc = float(row.get("delta_charge", 0) or 0)
    dv = float(row.get("delta_volume", 0) or 0)
    dh = float(row.get("delta_hydrophobicity", 0) or 0)
    effects = [
        f"converts {_AA_NAME.get(wt, wt)} to {_AA_NAME.get(mut, mut)}",
        _signed_change(dh, "raises Kyte–Doolittle hydropathy", "lowers Kyte–Doolittle hydropathy",
                       "preserves Kyte–Doolittle hydropathy"),
        _signed_change(dv, "increases side-chain volume", "decreases side-chain volume",
                       "preserves side-chain volume", " Å³"),
        _signed_change(dc, "raises formal side-chain charge", "lowers formal side-chain charge",
                       "preserves formal side-chain charge", " e"),
    ]
    if mut == "P":
        effects.append("introducing proline geometry that can restrict the backbone φ angle and disrupt regular secondary structure")
    elif wt == "G" and mut != "G":
        effects.append("removing glycine's unusually broad backbone conformational freedom")
    elif mut == "G" and wt != "G":
        effects.append("introducing glycine and increasing backbone conformational freedom")
    if wt == "C" or mut == "C":
        effects.append("altering cysteine-dependent local chemistry")
    return ", ".join(effects)


def _computed_signals(row: pd.Series) -> list[str]:
    signals = []
    ddg = row.get("rasp_ddg")
    if _present(ddg):
        tier = str(row.get("stability_tier", "predicted"))
        signals.append(f"RaSP predicts {tier} global destabilization (ΔΔG {float(ddg):.2f} kcal/mol)")
    apbs = row.get("apbs_delta_phi_local_p95_abs_kbt_e")
    if _present(apbs):
        signals.append(
            f"APBS detects local electrostatic redistribution within 8 Å "
            f"(95th percentile |Δφ| {float(apbs):.2f} kBT/e)"
        )
    encom = row.get("encom_relative_entropy_change")
    delta_entropy = row.get("encom_delta_vibrational_entropy")
    if _present(encom) and _present(delta_entropy):
        direction = "increased" if float(delta_entropy) > 0 else "reduced"
        percent = float(encom) * 100
        precision = 4 if percent < 0.01 else 2
        signals.append(
            f"ENCoM normal-mode analysis predicts {direction} vibrational entropy "
            f"({percent:.{precision}f}% relative change)"
        )
    displacement = row.get("backbone_ca_displacement_a")
    if _present(displacement):
        signals.append(
            f"the globally aligned minimized models differ by {float(displacement):.3f} Å at the residue Cα"
        )
    return signals


def _prs_clause(row: pd.Series) -> str:
    values = []
    for column, value in row.items():
        if column.startswith("prs_to_") and _present(value):
            values.append((float(value), column.removeprefix("prs_to_").replace("_", " ")))
    if not values or not bool(row.get("flag_allosteric_high_coupling", False)):
        return ""
    value, target = max(values)
    return (
        f"PRS assigns high modeled perturbation coupling from this residue to the {target} "
        f"(mean response {value:.4f}, unitless)"
    )


def _ppi_mechanism(row: pd.Series) -> str:
    if not bool(row.get("flag_ppi_interface_context", False)):
        return ""
    gene = str(row.get("gene", "BRAF")).upper()
    partner = str(row.get("ppi_partner", f"a {gene} partner"))
    source = str(row.get("ppi_source_complex", "an experimental complex"))
    bsa = row.get("ppi_interface_bsa")
    amount = f", burying {float(bsa):.1f} Å²" if _present(bsa) else ""
    if gene == "GSDMD" and partner == "GSDMD":
        consequence = "may alter protomer packing, oligomerization or stability of the membrane pore"
    elif partner == "14-3-3":
        consequence = "may alter 14-3-3-dependent BRAF assembly and conformational regulation"
    elif partner == "BRAF":
        consequence = "may alter RAF dimer geometry or affinity and thereby MAPK signaling"
    elif partner == "MEK1":
        consequence = "may alter BRAF–MEK recognition and substrate phosphorylation"
    else:
        consequence = f"may alter the geometry or affinity of the BRAF–{partner} interaction"
    return f"the residue lies at the {gene}–{partner} interface in {source}{amount}, so the substitution {consequence}"


def _literature_clause(row: pd.Series) -> str:
    hypotheses = _json_list(row.get("discovery_hypotheses"))
    reported = next(
        (item for item in hypotheses if str(item.get("classification", "")).startswith("literature")),
        None,
    )
    if not reported:
        return "retrieval found no variant-specific mechanistic report, so the proposed direction is computational"
    state = str(reported.get("conformational_state", "not_specified"))
    direction = str(reported.get("predicted_direction", "unclear"))
    gene = str(row.get("gene", "BRAF")).upper()
    if state == "inactive_to_active" or direction == "shift_toward_active_state":
        claim = "exact-variant literature reports a shift from inactive toward active kinase conformation"
    elif gene == "GSDMD" and state == "cleavage_control":
        claim = "exact-variant literature reports altered proteolytic control of GSDMD"
    elif gene == "GSDMD" and state == "pore_pathway":
        claim = "exact-variant literature reports an effect in the GSDMD membrane-pore pathway"
    elif direction not in {"", "unclear", "alteration_reported"}:
        claim = f"retrieved literature supports {direction.replace('_', ' ')}"
    else:
        target = str(reported.get("target", f"{gene} function")).replace("_", " ")
        claim = f"retrieved literature reports alteration of {target}"
    disease = str(row.get("disease_association_status", "none"))
    if disease == "associated":
        claim += f" in an exact variant–{str(row.get('disease_context', 'disease'))} context"
    elif disease == "cooccurrence":
        claim += " with variant–tumor co-occurrence"
    return claim


def _therapy_clause(row: pd.Series) -> str:
    text = str(row.get("therapy_hypothesis", "") or "")
    if not text or text.lower() == "nan":
        return ""
    agents = re.findall(
        r"\b(?:dabrafenib|vemurafenib|encorafenib|trametinib|cobimetinib|binimetinib|sorafenib|LY3009120)\b",
        text,
        flags=re.I,
    )
    if not agents:
        return ""
    names = "/".join(dict.fromkeys(agent.lower() for agent in agents))
    status = str(row.get("therapy_evidence_status", ""))
    if "resistance" in status or "refractoriness" in status:
        return f"an exact-variant passage reports a {names} resistance context"
    if "mixed" in status:
        return f"exact-variant passages describe context-dependent {names} sensitivity and resistance"
    if "sensitivity" in status or "response" in status:
        return f"an exact-variant passage reports a {names} sensitivity/response context"
    return f"an exact-variant passage discusses {names} without a directional response claim"


def _evidence_ids(row: pd.Series) -> str:
    values = []
    for column in ("literature_cited_evidence_ids", "disease_cited_evidence_ids",
                   "therapy_cited_evidence_ids"):
        raw = str(row.get(column, "") or "")
        values.extend(item for item in raw.split(";") if re.fullmatch(r"E\d+", item))
    return ";".join(dict.fromkeys(values))


def render_mechanistic_hypothesis(row: pd.Series) -> tuple[str, str]:
    """Return one detailed modal sentence and its deduplicated evidence IDs."""
    variant = str(row["mutation"])
    gene = str(row.get("gene", "BRAF")).upper()
    site, role = _site_context(row)
    opening = f"{gene} {variant} {_substitution(row)} in {site}, which {role}"
    signals = _computed_signals(row)
    clauses = [opening]
    if signals:
        clauses.append(", while ".join(signals))
    ppi = _ppi_mechanism(row)
    prs = _prs_clause(row)
    if prs:
        clauses.append(prs)
    if ppi:
        clauses.append(ppi)
        synthesis = (
            "the combined local and interaction-level changes could modify GSDMD oligomeric pore assembly"
            if gene == "GSDMD" else
            "the combined local and interaction-level changes could modify complex-dependent MAPK signaling"
        )
    elif gene == "GSDMD" and str(row.get("functional_region", "")) in {
        "pore_forming_n_terminal_domain", "membrane_insertion_hairpins", "cys191_regulatory_site"
    }:
        synthesis = "these changes could alter membrane engagement, oligomerization or the prepore-to-pore transition"
    elif gene == "GSDMD" and str(row.get("functional_region", "")) in {
        "interdomain_linker", "inflammatory_caspase_cleavage_region", "caspase_3_7_inactivating_site"
    }:
        synthesis = "these changes could alter protease recognition, cleavage accessibility or fragment release"
    elif str(row.get("functional_region", "")) in {
        "p_loop", "alpha_c_helix", "hrd_motif", "dfg_motif", "activation_segment"
    }:
        synthesis = "these changes could shift catalytic geometry or the active/inactive kinase equilibrium"
    elif bool(row.get("flag_buried_residue", False)):
        synthesis = f"these changes could disrupt local packing and propagate into {gene} domain dynamics"
    else:
        synthesis = "these changes could modify local structure or dynamics in the annotated regulatory context"
    clauses.append(f"{_literature_clause(row)}, and {synthesis}")
    therapy = _therapy_clause(row)
    if therapy:
        clauses.append(therapy)
    if bool(row.get("evidence_conflict_flag", False)):
        clauses.append("retrieved evidence contains conflicting mechanistic directions")
    sentence = "; ".join(clauses).rstrip(".; ") + "."
    return sentence, _evidence_ids(row)
