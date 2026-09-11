"""Versioned, score-neutral condition profiles for modular literature retrieval."""

from __future__ import annotations

from dataclasses import dataclass


QUERY_VERSION = "gsdmd_modular_v1"


@dataclass(frozen=True)
class ConditionProfile:
    name: str
    tiers: tuple[tuple[str, ...], ...]
    cell_context: tuple[str, ...] = ()

    @property
    def terms(self) -> tuple[str, ...]:
        return tuple(term for tier in self.tiers for term in tier)


CONDITION_PROFILES: dict[str, ConditionProfile] = {
    "general_gsdmd": ConditionProfile("general_gsdmd", (("inflammatory disease", "pyroptosis"),)),
    "neuro_aging": ConditionProfile(
        "neuro_aging",
        (
            ("aging", "Alzheimer disease", "dementia", "tauopathy", "frontotemporal dementia"),
            ("Parkinson disease", "amyotrophic lateral sclerosis", "multiple sclerosis", "neurodegeneration"),
            ("stroke", "cerebral ischemia", "intracerebral hemorrhage", "brain injury", "neuroinflammation"),
        ),
        ("microglia", "astrocyte", "neuron", "blood-brain barrier"),
    ),
    "alzheimer_dementia": ConditionProfile(
        "alzheimer_dementia", (("Alzheimer disease", "dementia", "tauopathy", "cognitive impairment"),),
        ("microglia", "astrocyte", "neuron"),
    ),
    "parkinson": ConditionProfile("parkinson", (("Parkinson disease", "alpha-synuclein", "dopaminergic neuron"),)),
    "als_ftd": ConditionProfile(
        "als_ftd", (("amyotrophic lateral sclerosis", "frontotemporal dementia", "TDP-43", "motor neuron"),)
    ),
    "multiple_sclerosis": ConditionProfile(
        "multiple_sclerosis", (("multiple sclerosis", "demyelination", "experimental autoimmune encephalomyelitis"),)
    ),
    "stroke_ischemia": ConditionProfile(
        "stroke_ischemia", (("stroke", "cerebral ischemia", "ischemia reperfusion", "intracerebral hemorrhage"),)
    ),
    "brain_injury": ConditionProfile(
        "brain_injury", (("traumatic brain injury", "brain injury", "blood-brain barrier", "neuroinflammation"),)
    ),
    # Kept solely for existing BRAF runs. It is never selected by GSDMD.
    "braf_oncology": ConditionProfile(
        "braf_oncology", (("cancer", "tumor", "melanoma", "oncogenic disease"),)
    ),
}


def get_condition_profile(name: str) -> ConditionProfile:
    try:
        return CONDITION_PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown condition profile {name!r}; available: {', '.join(sorted(CONDITION_PROFILES))}"
        ) from error


def normalized_condition_terms(profile: ConditionProfile, extra_terms: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return stable unique terms; free terms deepen retrieval and never encode weights."""
    cleaned = [str(term).strip() for term in (*profile.terms, *extra_terms) if str(term).strip()]
    return tuple(dict.fromkeys(cleaned))
