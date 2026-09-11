"""Pure scope and rank helpers for the Task 11E retrieval stack."""

from __future__ import annotations

from collections.abc import Iterable

from src.rag.variant_mentions import contains_exact_variant, same_residue_variant_mentioned


EVIDENCE_SCOPES = (
    "exact_variant", "same_residue_analogy", "functional_region", "general_context",
)
MECHANISTIC_SCOPES = frozenset({
    "exact_variant", "same_residue_analogy", "functional_region",
})
SCOPE_PRIORITY = {scope: len(EVIDENCE_SCOPES) - index for index, scope in enumerate(EVIDENCE_SCOPES)}

REGION_ALIASES = {
    "activation_segment": ("activation segment", "activation loop"),
    "p_loop": ("p-loop", "phosphate-binding loop", "phosphate binding loop"),
    "alpha_c_helix": ("alpha-c helix", "αc helix"),
    "interdomain_linker": ("interdomain linker", "cleavage region", "cleavage site"),
    "pore_forming_n_terminal_domain": ("n-terminal domain", "pore-forming domain"),
    "autoinhibitory_c_terminal_domain": ("c-terminal domain", "autoinhibitory domain"),
}


def region_aliases(functional_region: str) -> tuple[str, ...]:
    normalized = str(functional_region or "").lower()
    return REGION_ALIASES.get(normalized, (normalized.replace("_", " "),))


def classify_evidence_scope(passage: str, variant: str, functional_region: str = "") -> str:
    """Classify a passage once, from strongest lexical evidence to weakest."""
    text = str(passage).lower()
    if variant and contains_exact_variant(text, variant):
        return "exact_variant"
    if variant and same_residue_variant_mentioned(text, variant):
        return "same_residue_analogy"
    if any(alias and alias in text for alias in region_aliases(functional_region)):
        return "functional_region"
    return "general_context"


def normalized_rank_scores(items: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Convert a descending per-channel ranking into a comparable [0,1] score."""
    values = list(items)
    if not values:
        return {}
    count = len(values)
    return {key: (count - index) / count for index, (key, _score) in enumerate(values)}


def strongest_scope(scopes: Iterable[str]) -> str:
    values = list(scopes)
    return max(values, key=lambda scope: SCOPE_PRIORITY.get(scope, 0), default="general_context")


def split_evidence_blocks(items: Iterable):
    """Return independent mechanistic and contextual evidence blocks.

    General context is intentionally never used to pad the mechanistic block.
    The helper is kept type-light so it can be used by both cached models and
    retrieval-time ``LiteratureEvidence`` instances.
    """
    mechanistic = []
    context = []
    for item in items:
        if getattr(item, "evidence_scope", None) in MECHANISTIC_SCOPES:
            mechanistic.append(item)
        else:
            context.append(item)
    return mechanistic, context
