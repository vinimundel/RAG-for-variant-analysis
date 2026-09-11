"""Sampling design and retrieval metrics for the human evaluation of the RAG.

The evaluation is deliberately small and honest about it: 24 stratified
variants, five retrieved passages each, and 24 of those passages re-presented
blind after a washout so intra-rater agreement can be measured.  Recall is never
estimated, because a single reviewer cannot establish the complete set of
relevant passages in a corpus of this size.

Sampling is seeded and deterministic, so the sheet can be regenerated exactly.
Metrics are pure functions of the annotations.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

import pandas as pd

BENCHMARK_VARIANTS = 24
PASSAGES_PER_VARIANT = 5
BLIND_REPEATS = 24
HUMAN_REQUIRED_COLUMNS = (
    "relevance", "specificity", "entailment", "support_type", "quality", "conflict",
)
SYSTEM_CITATION_COLUMNS = ("cited_in_claim", "system_citation_key")
# The confirmatory rubric is deliberately closed.  The two filled development
# sheets predate this rubric and are validated against separate, historical
# vocabularies by ``task11_annotations``; they must never be passed to the
# publication gate.
CONFIRMATORY_VOCABULARIES = {
    "relevance": frozenset({"irrelevant", "background", "relevant", "directly_on_point"}),
    "specificity": frozenset({"exact_variant", "same_residue", "functional_region", "general"}),
    "entailment": frozenset({"supports", "neutral", "contradicts"}),
    "support_type": frozenset({
        "structural_biophysical", "biochemical", "cell_functional", "in_vivo",
        "human_association", "computational", "context",
    }),
    "quality": frozenset({"controlled", "uncontrolled", "unclear"}),
    "conflict": frozenset({"yes", "no"}),
    "cited_in_claim": frozenset({"yes", "no"}),
}
# Graded relevance used by nDCG. Binary precision treats >= 2 as relevant.
RELEVANCE_SCALE = {"irrelevant": 0, "background": 1, "relevant": 2, "directly_on_point": 3}
RELEVANT_THRESHOLD = 2

# Sampling strata. Order is the allocation order when a stratum is short.
STRATA = (
    "exact_variant_in_corpus",
    "same_residue_in_corpus",
    "functional_region_only",
    "no_corpus_mention",
    "conflicting_evidence",
)
# Conflict cannot be established lexically: it needs adjudication of two sources
# that disagree. The stratum exists so its emptiness is reported rather than
# hidden, and is filled once a reviewed adjudication set exists.
ADJUDICATED_STRATA = ("conflicting_evidence",)


def corpus_mentions(passages: Iterable[str]) -> dict[int, set[str]]:
    """Map each mentioned residue position to the variants seen at it.

    Computed once over the primary-evidence corpus, so stratifying every variant
    costs one pass rather than one scan per variant.
    """
    from src.rag.variant_mentions import mentioned_one_letter_variants

    by_position: dict[int, set[str]] = {}
    for passage in passages:
        for mention in mentioned_one_letter_variants(passage):
            by_position.setdefault(int(mention[1:-1]), set()).add(mention)
    return by_position


def assign_stratum(row: dict, by_position: dict[int, set[str]],
                   motif_positions: set[int] = frozenset()) -> str:
    """Label one variant by what the curated corpus actually mentions.

    This is a sampling prior, not ground truth: it decides which variants get
    reviewed, never what the reviewer is allowed to conclude about them.
    """
    if bool(row.get("adjudicated_conflict", False)):
        return "conflicting_evidence"
    mutation = str(row["mutation"]).upper()
    position = int(row.get("pos") or mutation[1:-1])
    mentions = by_position.get(position, set())
    if mutation in mentions:
        return "exact_variant_in_corpus"
    if mentions:
        return "same_residue_in_corpus"
    if position in motif_positions:
        return "functional_region_only"
    return "no_corpus_mention"


def motif_positions(functional_motifs: dict[str, tuple[int, int]]) -> set[int]:
    """Every residue inside a configured functional motif."""
    return {position for start, end in functional_motifs.values()
            for position in range(int(start), int(end) + 1)}


def _stable_order(values: Iterable[str], seed: str) -> list[str]:
    """Deterministic shuffle that does not depend on the interpreter's hashing."""
    return sorted(values, key=lambda value: hashlib.sha256(
        f"{seed}:{value}".encode()).hexdigest())


def allocate(total: int, strata: Sequence[str], available: dict[str, int]) -> dict[str, int]:
    """Split ``total`` across strata, redistributing whatever a stratum cannot fill."""
    base, remainder = divmod(total, len(strata))
    quota = {stratum: base + (1 if index < remainder else 0)
             for index, stratum in enumerate(strata)}
    allocated = {stratum: min(quota[stratum], available.get(stratum, 0)) for stratum in strata}
    shortfall = total - sum(allocated.values())
    while shortfall > 0:
        # Redistribute to strata that still have unused variants, in fixed order.
        expandable = [stratum for stratum in strata
                      if available.get(stratum, 0) > allocated[stratum]]
        if not expandable:
            break
        for stratum in expandable:
            if shortfall == 0:
                break
            allocated[stratum] += 1
            shortfall -= 1
    return allocated


def sample_variants(frame: pd.DataFrame, by_position: dict[int, set[str]] | None = None,
                    motifs: set[int] = frozenset(), seed: str = "braf_rag_benchmark_v1",
                    total: int = BENCHMARK_VARIANTS) -> tuple[pd.DataFrame, dict]:
    """Pick the stratified review set, plus the allocation actually achieved.

    A stratum the corpus cannot fill is reported with its shortfall instead of
    being quietly absorbed by the others.
    """
    by_position = by_position or {}
    working = frame.copy()
    working["stratum"] = [assign_stratum(row, by_position, motifs)
                          for row in working.to_dict("records")]
    available = {stratum: int((working["stratum"] == stratum).sum()) for stratum in STRATA}
    allocated = allocate(total, STRATA, available)
    chosen = []
    for stratum in STRATA:
        pool = working.loc[working["stratum"] == stratum, "mutation"].astype(str).tolist()
        chosen.extend(_stable_order(pool, f"{seed}:{stratum}")[:allocated[stratum]])
    selected = working[working["mutation"].astype(str).isin(chosen)].copy()
    base, remainder = divmod(total, len(STRATA))
    design = {
        "available_per_stratum": available, "allocated_per_stratum": allocated,
        "target_per_stratum": {stratum: base + (1 if index < remainder else 0)
                               for index, stratum in enumerate(STRATA)},
        "empty_strata": [stratum for stratum in STRATA if not available[stratum]],
        "adjudicated_strata_not_derivable_from_text": list(ADJUDICATED_STRATA),
        "sampled_total": int(len(selected)),
    }
    return selected.sort_values(["stratum", "mutation"]).reset_index(drop=True), design


def blind_repeat_plan(item_ids: Sequence[str], seed: str = "braf_rag_benchmark_v1",
                      repeats: int = BLIND_REPEATS) -> pd.DataFrame:
    """Choose which annotated passages come back blind, and under which new id.

    The reviewer receives only the new ids; the mapping stays in a separate key
    file so a repeat cannot be recognised from the sheet.
    """
    picked = _stable_order([str(item) for item in item_ids], f"{seed}:repeat")[:repeats]
    ordered = _stable_order(picked, f"{seed}:repeat_order")
    return pd.DataFrame([{"repeat_item_id": f"R{index:03d}", "original_item_id": item}
                         for index, item in enumerate(ordered, 1)])


def precision_at_k(relevance: Sequence[int], k: int = PASSAGES_PER_VARIANT) -> float:
    """Fraction of the top ``k`` passages judged relevant."""
    top = list(relevance)[:k]
    if not top:
        return float("nan")
    return sum(1 for value in top if value >= RELEVANT_THRESHOLD) / len(top)


def dcg(relevance: Sequence[int], k: int = PASSAGES_PER_VARIANT) -> float:
    return sum(value / math.log2(rank + 1)
               for rank, value in enumerate(list(relevance)[:k], 1))


def ndcg_at_k(relevance: Sequence[int], k: int = PASSAGES_PER_VARIANT) -> float:
    """Ranking quality against the best ordering of the same judged passages.

    With only the retrieved passages judged, the ideal ordering is their own
    descending order; this measures ranking, not coverage of the corpus.
    """
    ideal = dcg(sorted(relevance, reverse=True), k)
    if ideal == 0:
        return float("nan")
    return dcg(relevance, k) / ideal


def reciprocal_rank(relevance: Sequence[int]) -> float:
    """Reciprocal rank of the first relevant passage; 0 when none is relevant."""
    for rank, value in enumerate(relevance, 1):
        if value >= RELEVANT_THRESHOLD:
            return 1 / rank
    return 0.0


def cohen_kappa(first: Sequence, second: Sequence) -> float:
    """Agreement between two labellings of the same items, chance-corrected.

    Used here for one reviewer against themselves across the washout, which
    measures stability of the rubric rather than agreement between people.
    """
    pairs = [(a, b) for a, b in zip(first, second, strict=True)
             if a is not None and b is not None]
    if not pairs:
        return float("nan")
    labels = sorted({label for pair in pairs for label in pair}, key=str)
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    expected = sum(
        (sum(1 for a, _ in pairs if a == label) / total)
        * (sum(1 for _, b in pairs if b == label) / total)
        for label in labels
    )
    if expected == 1:
        # Every item carries the same label; agreement is undefined, not perfect.
        return float("nan")
    return (observed - expected) / (1 - expected)


def retrieval_metrics(annotations: pd.DataFrame) -> pd.DataFrame:
    """Per-variant Precision@5, nDCG@5 and reciprocal rank from the annotations."""
    rows = []
    for mutation, group in annotations.groupby("mutation", sort=True):
        ordered = group.sort_values("rank")
        relevance = [RELEVANCE_SCALE.get(str(value), 0) for value in ordered["relevance"]]
        rows.append({
            "mutation": mutation, "judged_passages": len(relevance),
            "precision_at_5": precision_at_k(relevance),
            "ndcg_at_5": ndcg_at_k(relevance),
            "reciprocal_rank": reciprocal_rank(relevance),
            "exact_variant_claims": int((ordered["specificity"] == "exact_variant").sum()),
            "conflicts_flagged": int(ordered["conflict"].astype(str).eq("yes").sum()),
        })
    return pd.DataFrame(rows)


def citation_accuracy(annotations: pd.DataFrame) -> dict[str, float | int | None]:
    """How often a cited passage actually supports the claim it was cited for."""
    cited = annotations[annotations["cited_in_claim"].astype(str).eq("yes")]
    if not len(cited):
        return {"cited_passages": 0, "citation_accuracy": None,
                "exact_claim_accuracy": None}
    entailed = cited["entailment"].astype(str).eq("supports")
    exact = cited[cited["specificity"].astype(str).eq("exact_variant")]
    return {
        "cited_passages": int(len(cited)),
        "citation_accuracy": float(entailed.mean()),
        "exact_claim_accuracy": (float(exact["entailment"].astype(str).eq("supports").mean())
                                 if len(exact) else None),
    }


def benchmark_gate(sheet: pd.DataFrame, repeats: pd.DataFrame,
                   repeat_key: pd.DataFrame | None = None) -> dict:
    """Evaluate the publication gates without filling missing human labels.

    A blank or partially annotated sheet is a blocked gate, never an implicit
    pass.  The repeated block is joined through its separate key only at audit
    time; the reviewer-facing CSV remains blinded.
    """
    required = HUMAN_REQUIRED_COLUMNS
    annotation_complete = all(
        column in sheet and sheet[column].fillna("").astype(str).str.strip().ne("").all()
        for column in required
    ) and all(
        column in repeats and repeats[column].fillna("").astype(str).str.strip().ne("").all()
        for column in required
    )
    system_citation_complete = all(
        column in sheet and sheet[column].fillna("").astype(str).str.strip().ne("").all()
        for column in SYSTEM_CITATION_COLUMNS
    ) and all(
        column in repeats and repeats[column].fillna("").astype(str).str.strip().ne("").all()
        for column in SYSTEM_CITATION_COLUMNS
    ) and sheet.get("cited_in_claim", pd.Series(dtype=str)).astype(str).isin({"yes", "no"}).all() \
        and repeats.get("cited_in_claim", pd.Series(dtype=str)).astype(str).isin({"yes", "no"}).all()
    structural_checks = {
        "sheet_120_passages": len(sheet) == 120,
        "repeat_24_passages": len(repeats) == 24,
        "primary_layer_only": ("corpus_layer" in sheet and
                                sheet["corpus_layer"].astype(str).eq("primary_evidence").all()),
        "max_two_passages_per_pmid": (not sheet.groupby("mutation")["pmid"].apply(
            lambda values: values.astype(str).value_counts().max() > 2
        ).any()) if len(sheet) else False,
        "three_pmids_when_available": True,
        "unique_item_ids": (sheet["item_id"].astype(str).is_unique and
                             repeats["item_id"].astype(str).is_unique),
        "system_citation_key_complete": bool(system_citation_complete),
    }
    if system_citation_complete:
        structural_checks["system_citation_keys_unique"] = bool(
            sheet["system_citation_key"].astype(str).is_unique and
            repeats["system_citation_key"].astype(str).is_unique
        )
    if "panel" in sheet:
        structural_checks["production_top5_panel"] = bool(
            sheet["panel"].astype(str).eq("production_top5").all()
        )
    if "pmid" in sheet and len(sheet):
        per_variant = sheet.groupby("mutation")["pmid"].nunique()
        structural_checks["three_pmids_when_available"] = bool((per_variant >= 3).all())

    metrics = {
        "precision_at_5": None,
        "citation_precision": None,
        "exact_claim_precision": None,
        "intra_rater_kappa": None,
    }
    reasons = []
    if not annotation_complete:
        reasons.extend(["human annotations are incomplete",
                        "all six human annotation fields are required"])
    if not system_citation_complete:
        reasons.append("system citation key is incomplete or was not generated")
    if not all(structural_checks.values()):
        reasons.extend(key for key, value in structural_checks.items() if not value)
    if annotation_complete:
        per_variant = retrieval_metrics(sheet)
        metrics["precision_at_5"] = float(per_variant["precision_at_5"].mean())
        citation = citation_accuracy(sheet)
        metrics["citation_precision"] = citation["citation_accuracy"]
        metrics["exact_claim_precision"] = citation["exact_claim_accuracy"]
        if repeat_key is None:
            reasons.append("blind repeat key is missing")
        else:
            joined = repeat_key.merge(
                sheet[["item_id", *required]], left_on="original_item_id", right_on="item_id",
                validate="one_to_one", suffixes=("_repeat_key", "_original"),
            ).merge(
                repeats[["item_id", *required]], left_on="repeat_item_id", right_on="item_id",
                validate="one_to_one", suffixes=("_original", "_repeat"),
            )
            metrics["intra_rater_kappa"] = cohen_kappa(
                joined["relevance_repeat"].tolist(), joined["relevance_original"].tolist()
            )
        if pd.isna(metrics["intra_rater_kappa"]):
            reasons.append("intra-rater kappa is undefined")
        for name, threshold in (("precision_at_5", .70), ("citation_precision", 1.0),
                                ("exact_claim_precision", 1.0), ("intra_rater_kappa", .60)):
            value = metrics[name]
            if pd.isna(value) or value < threshold:
                reasons.append(f"{name} below gate")
    passed = not reasons and all(
        metrics[name] is not None and metrics[name] >= threshold
        for name, threshold in (
            ("precision_at_5", .70), ("citation_precision", 1.0),
            ("exact_claim_precision", 1.0), ("intra_rater_kappa", .60),
        )
    )
    structural_checks = {key: bool(value) for key, value in structural_checks.items()}
    return {
        "status": "passed" if passed else "blocked",
        "promotion_allowed": passed,
        "metrics": metrics,
        "thresholds": {
            "precision_at_5": .70, "citation_precision": 1.0,
            "exact_claim_precision": 1.0, "intra_rater_kappa": .60,
        },
        "annotation_complete": bool(annotation_complete),
        "structural_checks": structural_checks,
        "reasons": reasons,
    }
