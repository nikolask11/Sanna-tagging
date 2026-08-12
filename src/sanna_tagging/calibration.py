"""Token temperature scaling and sentence-level selective calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import torch

from .tokenization import AlignedWordProbabilities

SENTENCE_FEATURE_NAMES: tuple[str, ...] = (
    "length",
    "expected_errors",
    "low_confidence_fraction",
    "minimum_confidence",
    "ensemble_disagreement",
)


class CalibrationError(RuntimeError):
    """Calibration inputs are incomplete, misaligned, or degenerate."""


@dataclass(frozen=True, slots=True)
class TokenTemperature:
    temperature: float
    fit_token_count: int
    nll_before: float
    nll_after: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    def transform(self, aligned: AlignedWordProbabilities) -> AlignedWordProbabilities:
        return apply_token_temperature(aligned, self.temperature)


@dataclass(frozen=True, slots=True)
class SentenceFeatures:
    sentence_ids: tuple[Hashable, ...]
    values: np.ndarray
    names: tuple[str, ...] = SENTENCE_FEATURE_NAMES

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape != (len(self.sentence_ids), len(self.names)):
            raise ValueError("sentence features must be [sentences, named features]")
        if not np.isfinite(values).all():
            raise ValueError("sentence features must be finite")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class SentenceSuccessModel:
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    regularization_c: float
    fit_kind: str = "logistic"

    def predict_success_probability(self, features: SentenceFeatures) -> np.ndarray:
        if features.names != self.feature_names:
            raise ValueError("sentence feature schema differs from the fitted calibrator")
        means = np.asarray(self.feature_means, dtype=np.float64)
        scales = np.asarray(self.feature_scales, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        standardized = (features.values - means) / scales
        scores = standardized @ coefficients + self.intercept
        probabilities = np.empty_like(scores)
        positive = scores >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
        exp_scores = np.exp(scores[~positive])
        probabilities[~positive] = exp_scores / (1.0 + exp_scores)
        return probabilities


@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    token_temperature: TokenTemperature
    sentence_model: SentenceSuccessModel
    low_confidence_threshold: float


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    accepted: int
    coverage: float
    empirical_failure_risk: float
    mean_predicted_failure_risk: float
    accepted_high_quality_yield: float


@dataclass(frozen=True, slots=True)
class FixedRiskAssessment:
    target_failure_risk: float
    accepted: int
    coverage: float
    empirical_failure_risk: float | None
    accepted_high_quality: int
    accepted_high_quality_yield: float


@dataclass(frozen=True, slots=True)
class CalibrationAssessment:
    sentence_count: int
    brier_score: float
    log_loss: float
    risk_coverage: tuple[RiskCoveragePoint, ...]
    fixed_risks: tuple[FixedRiskAssessment, ...]


def _probability_matrices(
    aligned: AlignedWordProbabilities,
) -> tuple[np.ndarray, ...]:
    matrices: list[np.ndarray] = []
    label_count = len(aligned.label_names)
    for sentence_index, value in enumerate(aligned.probabilities):
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != label_count:
            raise CalibrationError(
                f"sentence {aligned.sentence_ids[sentence_index]!r} has invalid probabilities"
            )
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise CalibrationError("token probabilities must be finite and non-negative")
        row_sums = matrix.sum(axis=1)
        if (row_sums <= 0).any():
            raise CalibrationError("every token probability vector must have positive mass")
        matrices.append(matrix / row_sums[:, None])
    if not matrices:
        raise CalibrationError("calibration requires at least one sentence")
    return tuple(matrices)


def _validate_gold_ids(
    aligned: AlignedWordProbabilities,
    gold_label_ids: Sequence[Sequence[int]],
    *,
    matrices: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, ...]:
    if len(gold_label_ids) != len(aligned.sentence_ids):
        raise ValueError("gold labels must align with calibration sentences")
    label_count = len(aligned.label_names)
    normalized_matrices = _probability_matrices(aligned) if matrices is None else matrices
    result: list[np.ndarray] = []
    for sentence_index, (probabilities, labels) in enumerate(
        zip(normalized_matrices, gold_label_ids)
    ):
        row = np.asarray(labels, dtype=np.int64)
        if row.ndim != 1 or len(row) != len(probabilities):
            raise ValueError(f"gold labels do not align at sentence {sentence_index}")
        if (row < 0).any() or (row >= label_count).any():
            raise ValueError("gold label IDs fall outside the probability label axis")
        result.append(row)
    return tuple(result)


def fit_token_temperature(
    calibration_fit: AlignedWordProbabilities,
    gold_label_ids: Sequence[Sequence[int]],
    *,
    max_iter: int = 50,
) -> TokenTemperature:
    """Fit one positive temperature by token NLL on calibration-fit only."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    matrices = _probability_matrices(calibration_fit)
    labels = _validate_gold_ids(
        calibration_fit, gold_label_ids, matrices=matrices
    )
    probabilities = np.concatenate(matrices, axis=0)
    targets = np.concatenate(labels, axis=0)
    log_probabilities = torch.as_tensor(
        np.log(np.clip(probabilities, 1e-12, 1.0)), dtype=torch.float64
    )
    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    initial_fraction = (1.0 - 0.05) / 19.95
    initial_raw = math.log(initial_fraction / (1.0 - initial_fraction))
    raw_temperature = torch.nn.Parameter(
        torch.tensor(initial_raw, dtype=torch.float64)
    )
    optimizer = torch.optim.LBFGS(
        [raw_temperature], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def temperature_value() -> torch.Tensor:
        # A bounded parameter avoids numerical collapse on small calibration sets.
        return 0.05 + 19.95 * torch.sigmoid(raw_temperature)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            log_probabilities / temperature_value(), target_tensor
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(temperature_value().detach())
    nll_before = float(torch.nn.functional.cross_entropy(log_probabilities, target_tensor))
    nll_after = float(
        torch.nn.functional.cross_entropy(log_probabilities / temperature, target_tensor)
    )
    if not all(math.isfinite(value) for value in (temperature, nll_before, nll_after)):
        raise CalibrationError("temperature fitting produced non-finite values")
    return TokenTemperature(
        temperature=temperature,
        fit_token_count=len(targets),
        nll_before=nll_before,
        nll_after=nll_after,
    )


def apply_token_temperature(
    aligned: AlignedWordProbabilities, temperature: float
) -> AlignedWordProbabilities:
    """Apply scalar log-probability temperature without changing label IDs."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    transformed: list[np.ndarray] = []
    for matrix in _probability_matrices(aligned):
        scaled = np.log(np.clip(matrix, 1e-12, 1.0)) / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        exponentiated = np.exp(scaled)
        transformed.append(exponentiated / exponentiated.sum(axis=1, keepdims=True))
    return AlignedWordProbabilities(
        sentence_ids=aligned.sentence_ids,
        probabilities=tuple(transformed),
        occurrence_counts=aligned.occurrence_counts,
        label_names=aligned.label_names,
    )


def _validate_ensemble_members(
    reference: AlignedWordProbabilities,
    reference_matrices: Sequence[np.ndarray],
    members: Sequence[AlignedWordProbabilities] | None,
) -> tuple[tuple[np.ndarray, ...], ...]:
    if members is None:
        return ()
    normalized: list[tuple[np.ndarray, ...]] = []
    for member in members:
        if member.sentence_ids != reference.sentence_ids:
            raise ValueError("ensemble sentence IDs are not aligned")
        if member.label_names != reference.label_names:
            raise ValueError("ensemble label axes are not aligned")
        matrices = _probability_matrices(member)
        for index, (candidate, expected) in enumerate(
            zip(matrices, reference_matrices)
        ):
            if candidate.shape != expected.shape:
                raise ValueError(f"ensemble word shapes differ at sentence {index}")
        normalized.append(matrices)
    return tuple(normalized)


def build_sentence_features(
    calibrated_probabilities: AlignedWordProbabilities,
    *,
    ensemble_members: Sequence[AlignedWordProbabilities] | None = None,
    low_confidence_threshold: float = 0.98,
) -> SentenceFeatures:
    """Build the fixed five-feature sentence-success design matrix."""
    if not 0 < low_confidence_threshold < 1:
        raise ValueError("low_confidence_threshold must be between zero and one")
    matrices = _probability_matrices(calibrated_probabilities)
    members = _validate_ensemble_members(
        calibrated_probabilities, matrices, ensemble_members
    )
    rows: list[tuple[float, ...]] = []
    for sentence_index, matrix in enumerate(matrices):
        confidences = matrix.max(axis=1)
        if members:
            member_labels = np.stack(
                [member[sentence_index].argmax(axis=1) for member in members], axis=0
            )
            votes = np.zeros((len(matrix), matrix.shape[1]), dtype=np.int64)
            token_indices = np.broadcast_to(
                np.arange(len(matrix)), member_labels.shape
            )
            np.add.at(votes, (token_indices, member_labels), 1)
            disagreement = float(
                np.mean(1.0 - votes.max(axis=1) / len(members))
            )
        else:
            disagreement = 0.0
        rows.append(
            (
                float(len(matrix)),
                float(np.sum(1.0 - confidences)),
                float(np.mean(confidences < low_confidence_threshold)),
                float(np.min(confidences)),
                disagreement,
            )
        )
    return SentenceFeatures(
        sentence_ids=calibrated_probabilities.sentence_ids,
        values=np.asarray(rows, dtype=np.float64),
    )


def sentence_success_outcomes(
    probabilities: AlignedWordProbabilities,
    gold_label_ids: Sequence[Sequence[int]],
) -> np.ndarray:
    """Return literal sentence >=98%-correct outcomes using integer arithmetic."""
    matrices = _probability_matrices(probabilities)
    labels = _validate_gold_ids(probabilities, gold_label_ids, matrices=matrices)
    outcomes = [
        int(100 * int(np.sum(matrix.argmax(axis=1) == gold)) >= 98 * len(gold))
        for matrix, gold in zip(matrices, labels)
    ]
    return np.asarray(outcomes, dtype=np.int64)


def fit_sentence_success_predictor(
    features: SentenceFeatures,
    success_outcomes: Sequence[int | bool],
    *,
    regularization_c: float = 1.0,
) -> SentenceSuccessModel:
    """Fit an L2-regularized logistic sentence-success predictor."""
    if regularization_c <= 0 or not math.isfinite(regularization_c):
        raise ValueError("regularization_c must be finite and positive")
    outcomes = np.asarray(success_outcomes, dtype=np.int64)
    if outcomes.shape != (len(features.sentence_ids),) or not np.isin(outcomes, (0, 1)).all():
        raise ValueError("success outcomes must be one binary value per sentence")
    unique_outcomes = np.unique(outcomes)
    if len(unique_outcomes) == 1:
        # The event can be nearly universal (or absent) in a small fit split.
        # Use a predeclared Jeffreys-smoothed intercept-only model rather than
        # inspecting assessment labels or aborting an otherwise valid run.
        successes = int(outcomes.sum())
        probability = (successes + 0.5) / (len(outcomes) + 1.0)
        return SentenceSuccessModel(
            feature_names=features.names,
            feature_means=tuple(float(value) for value in features.values.mean(axis=0)),
            feature_scales=tuple(1.0 for _ in features.names),
            coefficients=tuple(0.0 for _ in features.names),
            intercept=math.log(probability / (1.0 - probability)),
            regularization_c=regularization_c,
            fit_kind="smoothed-intercept-only",
        )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features.values)
    standardized = scaler.transform(features.values)
    classifier = LogisticRegression(
        C=regularization_c,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    ).fit(standardized, outcomes)
    scales = np.asarray(scaler.scale_, dtype=np.float64)
    scales[scales == 0] = 1.0
    return SentenceSuccessModel(
        feature_names=features.names,
        feature_means=tuple(float(value) for value in scaler.mean_),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in classifier.coef_[0]),
        intercept=float(classifier.intercept_[0]),
        regularization_c=regularization_c,
    )


def _prepare_sentence_calibration(
    temperature: TokenTemperature,
    probabilities: AlignedWordProbabilities,
    gold_label_ids: Sequence[Sequence[int]],
    *,
    ensemble_members: Sequence[AlignedWordProbabilities] | None,
    low_confidence_threshold: float,
) -> tuple[SentenceFeatures, np.ndarray]:
    calibrated = temperature.transform(probabilities)
    features = build_sentence_features(
        calibrated,
        ensemble_members=ensemble_members,
        low_confidence_threshold=low_confidence_threshold,
    )
    return features, sentence_success_outcomes(calibrated, gold_label_ids)


def fit_calibration(
    calibration_fit: AlignedWordProbabilities,
    gold_label_ids: Sequence[Sequence[int]],
    *,
    ensemble_members: Sequence[AlignedWordProbabilities] | None = None,
    regularization_c: float = 1.0,
    low_confidence_threshold: float = 0.98,
) -> CalibrationBundle:
    """Fit token and sentence calibration exclusively on calibration-fit data."""
    temperature = fit_token_temperature(calibration_fit, gold_label_ids)
    features, outcomes = _prepare_sentence_calibration(
        temperature,
        calibration_fit,
        gold_label_ids,
        ensemble_members=ensemble_members,
        low_confidence_threshold=low_confidence_threshold,
    )
    sentence_model = fit_sentence_success_predictor(
        features, outcomes, regularization_c=regularization_c
    )
    return CalibrationBundle(
        token_temperature=temperature,
        sentence_model=sentence_model,
        low_confidence_threshold=low_confidence_threshold,
    )


def risk_coverage_curve(
    success_probabilities: Sequence[float], success_outcomes: Sequence[int | bool]
) -> tuple[RiskCoveragePoint, ...]:
    probabilities, outcomes = _assessment_arrays(success_probabilities, success_outcomes)
    predicted_risk = 1.0 - probabilities
    order = np.lexsort((np.arange(len(probabilities)), predicted_risk))
    ordered_outcomes = outcomes[order]
    ordered_risks = predicted_risk[order]
    cumulative_failures = np.cumsum(1 - ordered_outcomes)
    cumulative_successes = np.cumsum(ordered_outcomes)
    cumulative_predicted_risk = np.cumsum(ordered_risks)
    return tuple(
        RiskCoveragePoint(
            accepted=index + 1,
            coverage=(index + 1) / len(outcomes),
            empirical_failure_risk=float(cumulative_failures[index] / (index + 1)),
            mean_predicted_failure_risk=float(
                cumulative_predicted_risk[index] / (index + 1)
            ),
            accepted_high_quality_yield=float(cumulative_successes[index] / len(outcomes)),
        )
        for index in range(len(outcomes))
    )


def _assessment_arrays(
    success_probabilities: Sequence[float], success_outcomes: Sequence[int | bool]
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(success_probabilities, dtype=np.float64)
    outcomes = np.asarray(success_outcomes, dtype=np.int64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("assessment requires one or more success probabilities")
    if outcomes.shape != probabilities.shape or not np.isin(outcomes, (0, 1)).all():
        raise ValueError("assessment outcomes must be aligned binary values")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("success probabilities must be finite values in [0, 1]")
    return probabilities, outcomes


def assess_sentence_calibration(
    success_probabilities: Sequence[float],
    success_outcomes: Sequence[int | bool],
    *,
    fixed_failure_risks: Sequence[float] = (0.01, 0.02, 0.05),
) -> CalibrationAssessment:
    """Assess held-out sentence calibration and selective high-quality yield."""
    probabilities, outcomes = _assessment_arrays(success_probabilities, success_outcomes)
    risks = tuple(float(value) for value in fixed_failure_risks)
    if not risks or len(set(risks)) != len(risks) or any(
        not math.isfinite(value) or not 0 < value < 1 for value in risks
    ):
        raise ValueError("fixed_failure_risks must be unique values between zero and one")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    brier = float(np.mean((probabilities - outcomes) ** 2))
    log_loss = float(
        -np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped))
    )
    predicted_failure = 1.0 - probabilities
    fixed: list[FixedRiskAssessment] = []
    for target in risks:
        accepted_mask = predicted_failure <= target
        accepted = int(np.sum(accepted_mask))
        high_quality = int(np.sum(outcomes[accepted_mask]))
        fixed.append(
            FixedRiskAssessment(
                target_failure_risk=target,
                accepted=accepted,
                coverage=accepted / len(outcomes),
                empirical_failure_risk=(
                    float(np.mean(1 - outcomes[accepted_mask])) if accepted else None
                ),
                accepted_high_quality=high_quality,
                accepted_high_quality_yield=high_quality / len(outcomes),
            )
        )
    return CalibrationAssessment(
        sentence_count=len(outcomes),
        brier_score=brier,
        log_loss=log_loss,
        risk_coverage=risk_coverage_curve(probabilities, outcomes),
        fixed_risks=tuple(fixed),
    )


def assess_calibration(
    bundle: CalibrationBundle,
    calibration_assessment: AlignedWordProbabilities,
    gold_label_ids: Sequence[Sequence[int]],
    *,
    ensemble_members: Sequence[AlignedWordProbabilities] | None = None,
    fixed_failure_risks: Sequence[float] = (0.01, 0.02, 0.05),
) -> CalibrationAssessment:
    """Apply a fitted bundle and assess only on calibration-assessment data."""
    features, outcomes = _prepare_sentence_calibration(
        bundle.token_temperature,
        calibration_assessment,
        gold_label_ids,
        ensemble_members=ensemble_members,
        low_confidence_threshold=bundle.low_confidence_threshold,
    )
    success_probabilities = bundle.sentence_model.predict_success_probability(features)
    return assess_sentence_calibration(
        success_probabilities,
        outcomes,
        fixed_failure_risks=fixed_failure_risks,
    )
