"""Word-complete UPOS metrics and strict three-seed aggregation."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Hashable

from .config import UPOS_TAGS

# Inclusive sentence-length ranges. ``None`` means no upper bound.
DEFAULT_LENGTH_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-40", 21, 40),
    ("41+", 41, None),
)
ERROR_BUCKETS = ("0", "1", "2", "3+")


@dataclass(slots=True)
class _TagCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    support: int = 0


@dataclass(slots=True)
class _GroupCounts:
    sentences: int = 0
    tokens: int = 0
    correct: int = 0
    missing: int = 0
    at_least_98: int = 0
    exact: int = 0

    def add(self, *, total: int, correct: int, missing: int) -> None:
        self.sentences += 1
        self.tokens += total
        self.correct += correct
        self.missing += missing
        self.at_least_98 += int(100 * correct >= 98 * total)
        self.exact += int(correct == total)

    def result(self) -> dict[str, int | float]:
        return {
            "sentences": self.sentences,
            "tokens": self.tokens,
            "correct": self.correct,
            "error": self.tokens - self.correct,
            "errors": self.tokens - self.correct,
            "missing": self.missing,
            "token_accuracy": _ratio(self.correct, self.tokens),
            "sentence_at_least_98_count": self.at_least_98,
            "sentence_at_least_98_rate": _ratio(self.at_least_98, self.sentences),
            "exact_match_count": self.exact,
            "exact_match_rate": _ratio(self.exact, self.sentences),
        }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _validate_bins(
    bins: Sequence[tuple[str, int, int | None]],
) -> tuple[tuple[str, int, int | None], ...]:
    normalized = tuple(bins)
    if not normalized:
        raise ValueError("length_bins must not be empty")
    names: set[str] = set()
    previous_end = 0
    for name, start, end in normalized:
        if not name or name in names:
            raise ValueError("length-bin names must be non-empty and unique")
        if start != previous_end + 1 or (end is not None and end < start):
            raise ValueError("length bins must be contiguous inclusive ranges starting at one")
        names.add(name)
        if end is None:
            previous_end = start - 1
            if (name, start, end) != normalized[-1]:
                raise ValueError("only the final length bin may be open-ended")
        else:
            previous_end = end
    if normalized[-1][2] is not None:
        raise ValueError("the final length bin must be open-ended")
    return normalized


def _length_bin_name(
    length: int, bins: Sequence[tuple[str, int, int | None]]
) -> str:
    for name, start, end in bins:
        if length >= start and (end is None or length <= end):
            return name
    raise AssertionError(f"no length bin covers sentence length {length}")


def _normalize_predictions(
    predictions: Sequence[Sequence[str | None] | None],
    sentence_index: int,
    gold_length: int,
) -> tuple[str | None, ...]:
    if sentence_index >= len(predictions) or predictions[sentence_index] is None:
        return (None,) * gold_length
    value = predictions[sentence_index]
    if value is None:
        return (None,) * gold_length
    row = tuple(value)
    if len(row) > gold_length:
        raise ValueError(
            f"sentence {sentence_index} has {len(row)} predictions for {gold_length} gold words"
        )
    return row + (None,) * (gold_length - len(row))


def compute_metrics(
    gold: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str | None] | None],
    *,
    tags: Sequence[str] = UPOS_TAGS,
    length_bins: Sequence[tuple[str, int, int | None]] = DEFAULT_LENGTH_BINS,
) -> dict[str, Any]:
    """Compute metrics while treating every absent prediction as a word error.

    Sentence-level 98% success uses the literal integer event
    ``100 * correct >= 98 * total``; no rounded floating-point accuracy is used.
    """
    canonical_tags = tuple(tags)
    if not canonical_tags or len(set(canonical_tags)) != len(canonical_tags):
        raise ValueError("tags must be a non-empty unique sequence")
    tag_set = set(canonical_tags)
    bins = _validate_bins(length_bins)
    if len(predictions) > len(gold):
        raise ValueError("predictions contain more sentences than gold")

    overall = _GroupCounts()
    by_length = {name: _GroupCounts() for name, _, _ in bins}
    tag_counts = {tag: _TagCounts() for tag in canonical_tags}
    error_buckets = {name: 0 for name in ERROR_BUCKETS}

    for sentence_index, gold_sentence_value in enumerate(gold):
        gold_sentence = tuple(gold_sentence_value)
        if not gold_sentence:
            raise ValueError(f"gold sentence {sentence_index} is empty")
        unknown_gold = [tag for tag in gold_sentence if tag not in tag_set]
        if unknown_gold:
            raise ValueError(f"gold sentence {sentence_index} contains unknown tag {unknown_gold[0]!r}")
        predicted_sentence = _normalize_predictions(
            predictions, sentence_index, len(gold_sentence)
        )

        correct = 0
        missing = 0
        for gold_tag, predicted_tag in zip(gold_sentence, predicted_sentence):
            counts = tag_counts[gold_tag]
            counts.support += 1
            if predicted_tag is None:
                missing += 1
                counts.false_negative += 1
                continue
            if predicted_tag not in tag_set:
                raise ValueError(
                    f"prediction for sentence {sentence_index} contains unknown tag {predicted_tag!r}"
                )
            if predicted_tag == gold_tag:
                correct += 1
                counts.true_positive += 1
            else:
                counts.false_negative += 1
                tag_counts[predicted_tag].false_positive += 1

        total = len(gold_sentence)
        errors = total - correct
        bucket = str(errors) if errors < 3 else "3+"
        error_buckets[bucket] += 1
        overall.add(total=total, correct=correct, missing=missing)
        by_length[_length_bin_name(total, bins)].add(
            total=total, correct=correct, missing=missing
        )

    per_tag: dict[str, dict[str, int | float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for tag in canonical_tags:
        counts = tag_counts[tag]
        precision = _ratio(
            counts.true_positive, counts.true_positive + counts.false_positive
        )
        recall = _ratio(counts.true_positive, counts.true_positive + counts.false_negative)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_tag[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": counts.support,
            "true_positive": counts.true_positive,
            "false_positive": counts.false_positive,
            "false_negative": counts.false_negative,
        }

    canonical_gold = [list(sentence) for sentence in gold]
    corpus_payload = json.dumps(
        canonical_gold,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    result = overall.result()
    result.update(
        {
            "evaluation_corpus_sha256": hashlib.sha256(corpus_payload).hexdigest(),
            "total": result.pop("tokens"),
            "sentence_count": result.pop("sentences"),
            "sentence_at_least_98": {
                "count": result["sentence_at_least_98_count"],
                "total": overall.sentences,
                "rate": result["sentence_at_least_98_rate"],
            },
            "exact_match": {
                "count": result["exact_match_count"],
                "total": overall.sentences,
                "rate": result["exact_match_rate"],
            },
            "sentence_error_buckets": error_buckets,
            "macro": {
                "precision": sum(precisions) / len(precisions),
                "recall": sum(recalls) / len(recalls),
                "f1": sum(f1_scores) / len(f1_scores),
                "support": overall.tokens,
            },
            "per_tag": per_tag,
            "length_bins": {name: counts.result() for name, counts in by_length.items()},
        }
    )
    return result


def evaluate_predictions(
    gold: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str | None] | None],
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility alias with an explicit evaluation-oriented name."""
    return compute_metrics(gold, predictions, **kwargs)


def _numeric_leaf(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _aggregate_node(values: Sequence[Any], path: str) -> Any:
    first = values[0]
    if _numeric_leaf(first):
        if not all(_numeric_leaf(value) for value in values):
            raise ValueError(f"seed metric schemas differ at {path}")
        numeric = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"seed metrics must be finite at {path}")
        return {
            "values": numeric,
            "mean": statistics.fmean(numeric),
            "population_variance": statistics.pvariance(numeric),
            "population_std": statistics.pstdev(numeric),
            "min": min(numeric),
            "max": max(numeric),
        }
    if isinstance(first, Mapping):
        keys = tuple(first.keys())
        key_set = set(keys)
        if not all(isinstance(value, Mapping) and set(value.keys()) == key_set for value in values):
            raise ValueError(f"seed metric schemas differ at {path}")
        return {
            key: _aggregate_node(
                [value[key] for value in values], f"{path}.{key}" if path else str(key)
            )
            for key in keys
        }
    if isinstance(first, str) or first is None:
        if not all(value == first for value in values):
            raise ValueError(f"non-numeric seed metadata differs at {path}")
        return first
    raise TypeError(f"unsupported seed metric value at {path}: {type(first).__name__}")


def _corpus_signature(report: Mapping[str, Any]) -> tuple[Any, ...] | None:
    if "total" not in report or "sentence_count" not in report:
        return None
    corpus_hash = report.get("evaluation_corpus_sha256")
    if not isinstance(corpus_hash, str) or len(corpus_hash) != 64:
        raise ValueError(
            "complete metric reports require evaluation_corpus_sha256 for seed alignment"
        )
    per_tag = report.get("per_tag")
    supports = (
        tuple(sorted((tag, values.get("support")) for tag, values in per_tag.items()))
        if isinstance(per_tag, Mapping)
        else ()
    )
    length_bins = report.get("length_bins")
    bin_sizes = (
        tuple(
            sorted(
                (name, values.get("sentences"), values.get("tokens"))
                for name, values in length_bins.items()
            )
        )
        if isinstance(length_bins, Mapping)
        else ()
    )
    return corpus_hash, report["total"], report["sentence_count"], supports, bin_sizes


def aggregate_three_seeds(
    metrics_by_seed: Mapping[Hashable, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exactly three aligned seed reports using population dispersion."""
    if not isinstance(metrics_by_seed, Mapping) or len(metrics_by_seed) != 3:
        raise ValueError("exactly three seed metric mappings are required")
    seeds = tuple(metrics_by_seed.keys())
    reports = tuple(metrics_by_seed.values())
    if not all(isinstance(report, Mapping) for report in reports):
        raise TypeError("every seed value must be a metric mapping")
    signatures = tuple(_corpus_signature(report) for report in reports)
    if signatures[0] is not None and any(
        signature != signatures[0] for signature in signatures[1:]
    ):
        raise ValueError("all three seed reports must cover the same gold corpus")
    return {
        "seed_count": 3,
        "seeds": list(seeds),
        "metrics": _aggregate_node(reports, ""),
    }


def aggregate_seed_metrics(
    metrics_by_seed: Mapping[Hashable, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compatibility alias that retains the strict three-seed contract."""
    return aggregate_three_seeds(metrics_by_seed)
