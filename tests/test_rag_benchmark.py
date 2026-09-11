"""Contract tests for the human RAG evaluation design and its metrics."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.literature.benchmark import (
    ADJUDICATED_STRATA, BENCHMARK_VARIANTS, RELEVANCE_SCALE, STRATA, allocate,
    assign_stratum, blind_repeat_plan, citation_accuracy, cohen_kappa,
    corpus_mentions, motif_positions, ndcg_at_k, precision_at_k, reciprocal_rank,
    retrieval_metrics, sample_variants,
)

MOTIFS = {"p_loop": (464, 471)}


def catalog(n_per_stratum: int = 20) -> pd.DataFrame:
    """Positions 100+ are mentioned exactly, 200+ at the residue only, 300+ never."""
    rows = []
    for index in range(n_per_stratum):
        rows.append({"mutation": f"A{100 + index}V", "pos": 100 + index,
                     "adjudicated_conflict": False})
        rows.append({"mutation": f"A{200 + index}V", "pos": 200 + index,
                     "adjudicated_conflict": False})
        rows.append({"mutation": f"A{300 + index}V", "pos": 300 + index,
                     "adjudicated_conflict": False})
        rows.append({"mutation": f"A{464}V", "pos": 464, "adjudicated_conflict": False})
        rows.append({"mutation": f"C{100 + index}W", "pos": 100 + index,
                     "adjudicated_conflict": True})
    return pd.DataFrame(rows).drop_duplicates("mutation").reset_index(drop=True)


def mentions(n: int = 20) -> dict[int, set[str]]:
    by_position = {100 + index: {f"A{100 + index}V"} for index in range(n)}
    by_position.update({200 + index: {f"Q{200 + index}K"} for index in range(n)})
    return by_position


def test_corpus_mentions_indexes_every_position_seen():
    found = corpus_mentions(["BRAF V600E and G466V drive signalling", "no variants here"])
    assert found[600] == {"V600E"}
    assert found[466] == {"G466V"}


def test_corpus_mentions_ignores_drug_like_tokens():
    assert corpus_mentions(["treated with S3I-201 and PLX4720"]) == {}


def test_motif_positions_expand_configured_ranges():
    assert motif_positions(MOTIFS) == set(range(464, 472))


def test_an_exact_mention_outranks_a_same_residue_mention():
    row = {"mutation": "A100V", "pos": 100}
    assert assign_stratum(row, {100: {"A100V", "A100T"}}) == "exact_variant_in_corpus"


def test_a_different_substitution_at_the_same_residue_is_its_own_stratum():
    row = {"mutation": "A100V", "pos": 100}
    assert assign_stratum(row, {100: {"A100T"}}) == "same_residue_in_corpus"


def test_an_unmentioned_variant_inside_a_motif_is_region_only():
    row = {"mutation": "A464V", "pos": 464}
    assert assign_stratum(row, {}, motif_positions(MOTIFS)) == "functional_region_only"


def test_an_unmentioned_variant_outside_every_motif_has_no_corpus_mention():
    assert assign_stratum({"mutation": "A300V", "pos": 300}, {}, motif_positions(MOTIFS)) \
        == "no_corpus_mention"


def test_an_adjudicated_conflict_outranks_every_lexical_signal():
    row = {"mutation": "A100V", "pos": 100, "adjudicated_conflict": True}
    assert assign_stratum(row, {100: {"A100V"}}) == "conflicting_evidence"


def test_conflict_is_declared_as_not_derivable_from_text():
    assert "conflicting_evidence" in ADJUDICATED_STRATA


def test_sampling_returns_the_planned_number_of_variants():
    selected, design = sample_variants(catalog(), mentions(), motif_positions(MOTIFS))
    assert len(selected) == BENCHMARK_VARIANTS == design["sampled_total"]
    assert selected["mutation"].nunique() == BENCHMARK_VARIANTS


def test_sampling_is_deterministic():
    first, _ = sample_variants(catalog(), mentions(), motif_positions(MOTIFS))
    second, _ = sample_variants(catalog(), mentions(), motif_positions(MOTIFS))
    assert first["mutation"].tolist() == second["mutation"].tolist()


def test_a_different_seed_changes_the_sample():
    default, _ = sample_variants(catalog(), mentions(), motif_positions(MOTIFS))
    other, _ = sample_variants(catalog(), mentions(), motif_positions(MOTIFS), seed="other")
    assert set(default["mutation"]) != set(other["mutation"])


def test_every_stratum_is_represented_when_available():
    selected, _ = sample_variants(catalog(), mentions(), motif_positions(MOTIFS))
    assert set(selected["stratum"]) == set(STRATA)


def test_an_empty_stratum_is_reported_rather_than_absorbed_silently():
    plain = catalog()
    plain["adjudicated_conflict"] = False
    _, design = sample_variants(plain, mentions(), motif_positions(MOTIFS))
    assert "conflicting_evidence" in design["empty_strata"]
    assert design["available_per_stratum"]["conflicting_evidence"] == 0
    assert design["target_per_stratum"]["conflicting_evidence"] > 0


def test_a_short_stratum_is_redistributed_not_silently_dropped():
    allocated = allocate(24, STRATA, {**{s: 20 for s in STRATA}, "conflicting_evidence": 1})
    assert allocated["conflicting_evidence"] == 1
    assert sum(allocated.values()) == 24


def test_allocation_reports_a_total_it_cannot_reach():
    allocated = allocate(24, STRATA, {stratum: 2 for stratum in STRATA})
    assert sum(allocated.values()) == 10


def test_blind_repeats_are_deterministic_and_renamed():
    items = [f"I{index:03d}" for index in range(120)]
    plan = blind_repeat_plan(items)
    assert len(plan) == 24
    assert plan["repeat_item_id"].tolist() == [f"R{i:03d}" for i in range(1, 25)]
    assert set(plan["original_item_id"]) <= set(items)
    assert blind_repeat_plan(items).equals(plan)


def test_repeat_order_does_not_mirror_the_original_order():
    """A repeat sheet in original order would be recognisable as repeats."""
    items = [f"I{index:03d}" for index in range(120)]
    plan = blind_repeat_plan(items)
    assert plan["original_item_id"].tolist() != sorted(plan["original_item_id"])


def test_precision_counts_only_relevant_grades():
    assert precision_at_k([3, 2, 1, 0, 0]) == 0.4
    assert precision_at_k([0, 0, 0, 0, 0]) == 0.0
    assert precision_at_k([3, 3, 3, 3, 3]) == 1.0


def test_ndcg_rewards_putting_the_best_passage_first():
    assert ndcg_at_k([3, 2, 1, 0, 0]) == 1.0
    assert ndcg_at_k([0, 1, 2, 3, 0]) < 1.0


def test_ndcg_is_undefined_when_nothing_is_relevant():
    assert math.isnan(ndcg_at_k([0, 0, 0, 0, 0]))


def test_reciprocal_rank_finds_the_first_relevant_passage():
    assert reciprocal_rank([0, 0, 2, 0, 0]) == pytest.approx(1 / 3)
    assert reciprocal_rank([3, 0, 0]) == 1.0
    assert reciprocal_rank([0, 1, 1]) == 0.0


def test_kappa_is_one_for_identical_varied_labels():
    assert cohen_kappa(["a", "b", "a", "c"], ["a", "b", "a", "c"]) == 1.0


def test_kappa_is_undefined_when_every_item_shares_one_label():
    """Perfect agreement on a constant label carries no information."""
    assert math.isnan(cohen_kappa(["a", "a", "a"], ["a", "a", "a"]))


def test_kappa_is_zero_at_chance_agreement():
    first = ["a", "a", "b", "b"]
    second = ["a", "b", "a", "b"]
    assert cohen_kappa(first, second) == pytest.approx(0.0)


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohen_kappa(["a", "b"], ["a"])


def annotations() -> pd.DataFrame:
    return pd.DataFrame([
        {"mutation": "V600E", "rank": 1, "relevance": "directly_on_point",
         "specificity": "exact_variant", "conflict": "no", "cited_in_claim": "yes",
         "entailment": "supports"},
        {"mutation": "V600E", "rank": 2, "relevance": "relevant",
         "specificity": "same_residue", "conflict": "no", "cited_in_claim": "yes",
         "entailment": "neutral"},
        {"mutation": "V600E", "rank": 3, "relevance": "irrelevant",
         "specificity": "general", "conflict": "no", "cited_in_claim": "no",
         "entailment": "neutral"},
    ])


def test_metrics_are_computed_per_variant():
    metrics = retrieval_metrics(annotations()).set_index("mutation")
    assert metrics.loc["V600E", "judged_passages"] == 3
    assert metrics.loc["V600E", "precision_at_5"] == pytest.approx(2 / 3)
    assert metrics.loc["V600E", "reciprocal_rank"] == 1.0
    assert metrics.loc["V600E", "exact_variant_claims"] == 1


def test_citation_accuracy_counts_only_cited_passages():
    result = citation_accuracy(annotations())
    assert result["cited_passages"] == 2
    assert result["citation_accuracy"] == pytest.approx(0.5)
    assert result["exact_claim_accuracy"] == 1.0


def test_relevance_scale_is_ordered():
    assert list(RELEVANCE_SCALE.values()) == sorted(RELEVANCE_SCALE.values())
