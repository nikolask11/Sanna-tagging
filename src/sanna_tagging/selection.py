"""Deterministic three-seed budget and adaptation selection helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Hashable, Mapping, Sequence

from .metrics import aggregate_three_seeds


class SelectionError(RuntimeError):
    """Selection inputs do not support the predeclared comparison policy."""


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    name: str
    metrics_by_seed: Mapping[Hashable, Mapping[str, Any]]
    complexity: int = 0
    budget: int | None = None
    adapted_from: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name must not be empty")
        if self.complexity < 0:
            raise ValueError("candidate complexity must be non-negative")
        if self.budget is not None and self.budget <= 0:
            raise ValueError("candidate budget must be positive")
        if self.adapted_from == self.name:
            raise ValueError("an adapted candidate cannot use itself as its baseline")


@dataclass(frozen=True, slots=True)
class AggregatedCandidate:
    candidate: SelectionCandidate
    aggregate: Mapping[str, Any]
    mean_sentence_pass: float
    mean_token_accuracy: float
    mean_exact_match: float


@dataclass(frozen=True, slots=True)
class BudgetSelection:
    selected: AggregatedCandidate
    reference: AggregatedCandidate
    eligible_budgets: tuple[int, ...]
    sentence_tolerance: float
    token_tolerance: float


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    selected: AggregatedCandidate
    ranked_eligible: tuple[AggregatedCandidate, ...]
    rejected_by_noninferiority: tuple[str, ...]
    token_noninferiority_tolerance: float


def _mean_leaf(aggregate: Mapping[str, Any], *paths: tuple[str, ...]) -> float:
    metrics = aggregate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SelectionError("three-seed aggregate is missing metrics")
    for path in paths:
        value: Any = metrics
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            if isinstance(value, Mapping):
                value = value.get("mean")
            if isinstance(value, Real) and not isinstance(value, bool):
                result = float(value)
                if math.isfinite(result):
                    return result
    joined = " or ".join(".".join(path) for path in paths)
    raise SelectionError(f"aggregate is missing a finite mean for {joined}")


def aggregate_candidate(candidate: SelectionCandidate) -> AggregatedCandidate:
    """Aggregate exactly all three seeds and expose the predeclared means."""
    aggregate = aggregate_three_seeds(candidate.metrics_by_seed)
    return AggregatedCandidate(
        candidate=candidate,
        aggregate=aggregate,
        mean_sentence_pass=_mean_leaf(
            aggregate,
            ("sentence_at_least_98_rate",),
            ("sentence_at_least_98", "rate"),
        ),
        mean_token_accuracy=_mean_leaf(aggregate, ("token_accuracy",)),
        mean_exact_match=_mean_leaf(
            aggregate,
            ("exact_match_rate",),
            ("exact_match", "rate"),
        ),
    )


def aggregate_selection_candidates(
    candidates: Sequence[SelectionCandidate],
) -> tuple[AggregatedCandidate, ...]:
    if not candidates:
        raise ValueError("at least one selection candidate is required")
    names = [candidate.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    return tuple(aggregate_candidate(candidate) for candidate in candidates)


def _policy_order(candidate: AggregatedCandidate) -> tuple[float, float, float, int, str]:
    # Lower tuple is better: mean sentence pass, token, exact, then complexity.
    return (
        -candidate.mean_sentence_pass,
        -candidate.mean_token_accuracy,
        -candidate.mean_exact_match,
        candidate.candidate.complexity,
        candidate.candidate.name,
    )


def select_budget(
    candidates: Sequence[SelectionCandidate],
    *,
    sentence_tolerance: float = 0.01,
    token_tolerance: float = 0.005,
) -> BudgetSelection:
    """Choose the smallest budget within both tolerances of the best mean result."""
    if sentence_tolerance < 0 or token_tolerance < 0:
        raise ValueError("budget tolerances must be non-negative")
    aggregated = aggregate_selection_candidates(candidates)
    if any(candidate.candidate.budget is None for candidate in aggregated):
        raise ValueError("every budget candidate must define a budget")
    budgets = [candidate.candidate.budget for candidate in aggregated]
    if len(set(budgets)) != len(budgets):
        raise ValueError("budget candidates must define unique budgets")
    reference = min(aggregated, key=_policy_order)
    best_sentence_pass = max(candidate.mean_sentence_pass for candidate in aggregated)
    best_token_accuracy = max(candidate.mean_token_accuracy for candidate in aggregated)
    eligible = tuple(
        candidate
        for candidate in aggregated
        if candidate.mean_sentence_pass >= best_sentence_pass - sentence_tolerance
        and candidate.mean_token_accuracy >= best_token_accuracy - token_tolerance
    )
    if not eligible:
        raise AssertionError("the reference budget must satisfy its own tolerances")
    selected = min(
        eligible,
        key=lambda candidate: (
            int(candidate.candidate.budget or 0),
            *_policy_order(candidate),
        ),
    )
    return BudgetSelection(
        selected=selected,
        reference=reference,
        eligible_budgets=tuple(
            sorted(int(candidate.candidate.budget or 0) for candidate in eligible)
        ),
        sentence_tolerance=sentence_tolerance,
        token_tolerance=token_tolerance,
    )


def select_candidate(
    candidates: Sequence[SelectionCandidate],
    *,
    adaptation_token_noninferiority: float = 0.002,
) -> CandidateSelection:
    """Gate adaptations, then rank sentence pass/token/exact/lower complexity."""
    if adaptation_token_noninferiority < 0:
        raise ValueError("adaptation_token_noninferiority must be non-negative")
    aggregated = aggregate_selection_candidates(candidates)
    by_name = {candidate.candidate.name: candidate for candidate in aggregated}
    eligible: list[AggregatedCandidate] = []
    rejected: list[str] = []
    for candidate in aggregated:
        baseline_name = candidate.candidate.adapted_from
        if baseline_name is None:
            eligible.append(candidate)
            continue
        baseline = by_name.get(baseline_name)
        if baseline is None:
            raise SelectionError(
                f"adapted candidate {candidate.candidate.name!r} lacks baseline {baseline_name!r}"
            )
        if (
            candidate.mean_token_accuracy
            < baseline.mean_token_accuracy - adaptation_token_noninferiority
        ):
            rejected.append(candidate.candidate.name)
        else:
            eligible.append(candidate)
    if not eligible:
        raise SelectionError("all candidates were rejected by the adaptation gate")
    ranked = tuple(sorted(eligible, key=_policy_order))
    return CandidateSelection(
        selected=ranked[0],
        ranked_eligible=ranked,
        rejected_by_noninferiority=tuple(sorted(rejected)),
        token_noninferiority_tolerance=adaptation_token_noninferiority,
    )


def select_adaptation(
    candidates: Sequence[SelectionCandidate],
    *,
    token_noninferiority: float = 0.002,
) -> CandidateSelection:
    """Named alias for the adaptation comparison policy."""
    return select_candidate(
        candidates,
        adaptation_token_noninferiority=token_noninferiority,
    )
