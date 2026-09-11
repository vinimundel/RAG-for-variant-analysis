import json

import pandas as pd

from src.features.discovery_hypotheses import annotate_discovery_hypotheses
from src.model.mechanistic_hypothesis import render_mechanistic_hypothesis


def test_v600e_gets_activation_segment_conformational_hypothesis():
    frame = pd.DataFrame([{
        "mutation": "V600E", "wt_aa": "V", "mut_aa": "E", "pos": 600,
        "delta_charge": -1.0, "delta_volume": -1.6, "delta_hydrophobicity": -7.7,
        "flag_ppi_interface_context": False, "flag_ppi_hotspot_candidate": False,
        "flag_active_site_context": True, "flag_binding_site_context": False,
        "flag_phosphosite_context": False, "flag_allosteric_high_coupling": True,
        "flag_buried_residue": False, "flag_packing_disruption_candidate": False,
        "secondary_structure": "loop",
    }])
    result = annotate_discovery_hypotheses(
        frame, {"kinase_domain": (457, 717), "activation_segment": (594, 623)},
        {"activation_segment": (594, 623)},
    )
    hypotheses = json.loads(result.loc[0, "biophysical_hypotheses"])
    assert result.loc[0, "functional_region"] == "activation_segment"
    assert hypotheses[0]["mechanism"] == "conformational_equilibrium"
    assert 0 < result.loc[0, "biophysical_component"] <= 20


def test_presentation_hypothesis_reports_quantitative_multimodal_signals():
    row = pd.Series({
        "mutation": "V600E", "wt_aa": "V", "mut_aa": "E",
        "delta_charge": -1.0, "delta_volume": -1.6, "delta_hydrophobicity": -7.7,
        "functional_region": "activation_segment", "rasp_ddg": 0.624,
        "stability_tier": "mild", "apbs_delta_phi_local_p95_abs_kbt_e": 1.336,
        "encom_relative_entropy_change": 0.0019,
        "encom_delta_vibrational_entropy": -0.082,
        "backbone_ca_displacement_a": 0.043,
        "flag_ppi_interface_context": False, "flag_buried_residue": False,
        "flag_allosteric_high_coupling": False,
        "discovery_hypotheses": json.dumps([{
            "classification": "literature_reported",
            "conformational_state": "inactive_to_active",
            "predicted_direction": "shift_toward_active_state",
        }]),
        "disease_association_status": "associated",
        "literature_cited_evidence_ids": "E03", "disease_cited_evidence_ids": "E03",
        "therapy_cited_evidence_ids": "",
    })
    sentence, evidence = render_mechanistic_hypothesis(row)
    assert "valine to glutamate" in sentence
    assert "7.70" in sentence and "1.60 Å³" in sentence and "1.00 e" in sentence
    assert "1.34 kBT/e" in sentence and "0.19%" in sentence and "0.043 Å" in sentence
    assert "inactive toward active" in sentence
    assert "testable hypothesis" not in sentence
    assert evidence == "E03"
