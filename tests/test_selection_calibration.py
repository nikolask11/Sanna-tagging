from __future__ import annotations

import numpy as np
import pytest

from sanna_tagging.calibration import (
    assess_calibration,
    assess_sentence_calibration,
    build_sentence_features,
    fit_calibration,
    sentence_success_outcomes,
)
from sanna_tagging.modeling import ensemble_probabilities
from sanna_tagging.selection import SelectionCandidate, select_budget, select_candidate
from sanna_tagging.tokenization import AlignedWordProbabilities


def metrics(sentence_pass: float, token_accuracy: float, exact: float) -> dict[str, float]:
    return {
        "sentence_at_least_98_rate": sentence_pass,
        "token_accuracy": token_accuracy,
        "exact_match_rate": exact,
    }


def candidate(
    name: str,
    values: tuple[float, float, float],
    *,
    budget: int | None = None,
    complexity: int = 0,
    adapted_from: str | None = None,
) -> SelectionCandidate:
    return SelectionCandidate(
        name=name,
        metrics_by_seed={seed: metrics(*values) for seed in (0, 1, 2)},
        budget=budget,
        complexity=complexity,
        adapted_from=adapted_from,
    )


def aligned(rows, *, ids=None) -> AlignedWordProbabilities:
    matrices = tuple(np.asarray(row, dtype=np.float64) for row in rows)
    if ids is None:
        ids = tuple(f"s{index}" for index in range(len(matrices)))
    return AlignedWordProbabilities(
        sentence_ids=tuple(ids),
        probabilities=matrices,
        occurrence_counts=tuple(tuple(1 for _ in matrix) for matrix in matrices),
        label_names=("NOUN", "VERB"),
    )


def test_budget_selection_uses_three_seed_means_and_smallest_noninferior_budget():
    result = select_budget(
        (
            candidate("n50", (0.89, 0.900, 0.80), budget=50),
            candidate("n100", (0.90, 0.904, 0.82), budget=100),
            candidate("n200", (0.90, 0.905, 0.83), budget=200),
        ),
        sentence_tolerance=0.01,
        token_tolerance=0.005,
    )

    assert result.reference.candidate.name == "n200"
    assert result.selected.candidate.name == "n50"
    assert result.eligible_budgets == (50, 100, 200)


def test_budget_tolerances_use_independent_metric_maxima():
    result = select_budget(
        (
            candidate("sentence-best", (0.95, 0.900, 0.80), budget=50),
            candidate("token-best", (0.939, 0.920, 0.81), budget=100),
            candidate("eligible", (0.945, 0.916, 0.82), budget=200),
        ),
        sentence_tolerance=0.01,
        token_tolerance=0.005,
    )

    assert result.eligible_budgets == (200,)
    assert result.selected.candidate.name == "eligible"


def test_adaptation_noninferiority_gate_and_deterministic_complexity_tie_break():
    direct = candidate("direct", (0.90, 0.900, 0.80))
    rejected = candidate(
        "bad-adaptation",
        (0.99, 0.897, 0.99),
        complexity=1,
        adapted_from="direct",
    )
    simple = candidate(
        "simple-adaptation",
        (0.92, 0.899, 0.85),
        complexity=1,
        adapted_from="direct",
    )
    complex_arm = candidate(
        "complex-adaptation",
        (0.92, 0.899, 0.85),
        complexity=2,
        adapted_from="direct",
    )

    selected = select_candidate(
        (direct, rejected, complex_arm, simple),
        adaptation_token_noninferiority=0.002,
    )

    assert selected.rejected_by_noninferiority == ("bad-adaptation",)
    assert selected.selected.candidate.name == "simple-adaptation"


def test_three_member_ensemble_is_unweighted_probability_mean_and_strictly_aligned():
    members = (
        aligned(([[0.9, 0.1]],), ids=("s",)),
        aligned(([[0.6, 0.4]],), ids=("s",)),
        aligned(([[0.3, 0.7]],), ids=("s",)),
    )
    result = ensemble_probabilities(members)

    np.testing.assert_allclose(result.probabilities[0], [[0.6, 0.4]])
    assert result.occurrence_counts == ((3,),)
    with pytest.raises(ValueError, match="sentence IDs"):
        ensemble_probabilities((members[0], aligned(([[0.5, 0.5]],), ids=("other",))))


def test_calibration_fit_features_outcomes_and_heldout_assessment():
    rows = []
    gold = []
    for index in range(8):
        rows.append([[0.85, 0.15], [0.75, 0.25]] if index % 2 == 0 else [[0.4, 0.6], [0.75, 0.25]])
        gold.append([0, 0])
    probabilities = aligned(rows)
    second_member = aligned(
        [
            [[0.2, 0.8], [0.75, 0.25]] if index % 3 == 0 else row
            for index, row in enumerate(rows)
        ]
    )

    features = build_sentence_features(
        probabilities,
        ensemble_members=(probabilities, second_member),
        low_confidence_threshold=0.8,
    )
    outcomes = sentence_success_outcomes(probabilities, gold)
    bundle = fit_calibration(
        probabilities,
        gold,
        ensemble_members=(probabilities, second_member),
        regularization_c=1.0,
        low_confidence_threshold=0.8,
    )
    assessment = assess_calibration(
        bundle,
        probabilities,
        gold,
        ensemble_members=(probabilities, second_member),
        fixed_failure_risks=(0.05, 0.5),
    )

    assert features.names == (
        "length",
        "expected_errors",
        "low_confidence_fraction",
        "minimum_confidence",
        "ensemble_disagreement",
    )
    assert features.values.shape == (8, 5)
    assert features.values[:, 4].max() > 0
    assert outcomes.tolist() == [1, 0, 1, 0, 1, 0, 1, 0]
    assert bundle.token_temperature.fit_token_count == 16
    assert bundle.token_temperature.temperature > 0
    assert assessment.sentence_count == 8
    assert len(assessment.risk_coverage) == 8
    assert [point.target_failure_risk for point in assessment.fixed_risks] == [0.05, 0.5]
    assert 0 <= assessment.brier_score <= 1


def test_degenerate_sentence_fit_uses_smoothed_intercept_only_model():
    probabilities = aligned(([[0.99, 0.01]], [[0.98, 0.02]], [[0.97, 0.03]]))
    bundle = fit_calibration(probabilities, [[0], [0], [0]])
    features = build_sentence_features(bundle.token_temperature.transform(probabilities))
    predicted = bundle.sentence_model.predict_success_probability(features)

    assert bundle.sentence_model.fit_kind == "smoothed-intercept-only"
    assert np.allclose(predicted, (3.5 / 4.0))


def test_risk_coverage_orders_by_predicted_failure_and_handles_zero_acceptance():
    assessment = assess_sentence_calibration(
        [0.99, 0.8, 0.6],
        [1, 0, 1],
        fixed_failure_risks=(0.005, 0.2),
    )

    assert [point.accepted for point in assessment.risk_coverage] == [1, 2, 3]
    assert assessment.risk_coverage[0].empirical_failure_risk == 0.0
    assert assessment.fixed_risks[0].accepted == 0
    assert assessment.fixed_risks[0].empirical_failure_risk is None
