from __future__ import annotations

import numpy as np
import pytest

from sanna_tagging.metrics import aggregate_three_seeds, compute_metrics
from sanna_tagging.tokenization import (
    IGNORE_INDEX,
    TokenizationError,
    aggregate_window_probabilities,
    tokenize_pretokenized,
)


class FakeEncoding(dict):
    def __init__(self, fields, word_id_rows):
        super().__init__(fields)
        self._word_id_rows = word_id_rows

    def word_ids(self, batch_index):
        return self._word_id_rows[batch_index]


class OverflowTokenizer:
    is_fast = True

    def __init__(self, *, omit_last=False):
        self.omit_last = omit_last

    def __call__(self, words, **kwargs):
        assert kwargs == {
            "is_split_into_words": True,
            "truncation": True,
            "max_length": 6,
            "stride": 2,
            "return_overflowing_tokens": True,
            "padding": "max_length",
            "return_attention_mask": True,
        }
        second_last = None if self.omit_last else 4
        word_ids = [
            [None, 0, 0, 1, 2, None],
            [None, 2, 3, 3, second_last, None],
        ]
        return FakeEncoding(
            {
                "input_ids": [[0, 10, 11, 12, 13, 2], [0, 13, 14, 15, 16, 2]],
                "attention_mask": [[1] * 6, [1] * 6],
                "overflow_to_sample_mapping": [0, 0],
            },
            word_ids,
        )


def test_overflow_alignment_regroups_sentences_and_labels_each_word_once():
    tokenized = tokenize_pretokenized(
        [["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"]],
        OverflowTokenizer(),
        max_length=6,
        stride=2,
        labels=[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
        sentence_ids=["first", "second"],
    )

    assert tokenized.windows_by_sentence == ((0, 1), (2, 3))
    assert tokenized.word_counts == (5, 5)
    assert sum(label != IGNORE_INDEX for row in tokenized.labels or () for label in row) == 10
    assert len(tokenized.occurrences) == 12  # overlap word 2 appears in both windows.


def test_overflow_probabilities_average_duplicate_contexts_without_dropping_words():
    tokenized = tokenize_pretokenized(
        [["a", "b", "c", "d", "e"]],
        OverflowTokenizer(),
        max_length=6,
        stride=2,
        sentence_ids=["sentence"],
    )
    probabilities = np.full((2, 6, 2), 0.5)
    probabilities[0, 4] = [0.8, 0.2]
    probabilities[1, 1] = [0.4, 0.6]

    aligned = aggregate_window_probabilities(tokenized, probabilities, label_names=("A", "B"))

    np.testing.assert_allclose(aligned.probabilities[0][2], [0.6, 0.4])
    assert aligned.occurrence_counts == ((1, 1, 2, 1, 1),)
    assert aligned.probabilities[0].shape == (5, 2)


def test_tokenizer_fails_closed_when_overflow_misses_a_source_word():
    with pytest.raises(TokenizationError, match="source words have no tokenized occurrence"):
        tokenize_pretokenized(
            [["a", "b", "c", "d", "e"]],
            OverflowTokenizer(omit_last=True),
            max_length=6,
            stride=2,
        )


def test_literal_49_50_sentence_boundary_and_missing_predictions():
    tags = ("NOUN", "VERB")
    gold_49 = [["NOUN"] * 49]
    pred_49 = [["NOUN"] * 48 + ["VERB"]]
    gold_50 = [["NOUN"] * 50]
    pred_50 = [["NOUN"] * 49 + ["VERB"]]

    result_49 = compute_metrics(gold_49, pred_49, tags=tags)
    result_50 = compute_metrics(gold_50, pred_50, tags=tags)
    missing = compute_metrics([["NOUN", "VERB"], ["NOUN"]], [["NOUN"]], tags=tags)

    assert result_49["sentence_at_least_98_rate"] == 0.0
    assert result_50["sentence_at_least_98_rate"] == 1.0
    assert missing["missing"] == 2
    assert missing["error"] == 2
    assert missing["token_accuracy"] == pytest.approx(1 / 3)
    assert missing["sentence_error_buckets"] == {"0": 0, "1": 2, "2": 0, "3+": 0}


def test_three_seed_aggregation_requires_exactly_three_aligned_reports():
    reports = {
        seed: compute_metrics([["NOUN"]], [["NOUN"]], tags=("NOUN",))
        for seed in (0, 1, 2)
    }
    aggregate = aggregate_three_seeds(reports)
    assert aggregate["seed_count"] == 3
    assert aggregate["metrics"]["token_accuracy"]["mean"] == 1.0

    with pytest.raises(ValueError, match="exactly three"):
        aggregate_three_seeds({0: reports[0], 1: reports[1]})
    changed = dict(reports)
    changed[2] = compute_metrics([["NOUN", "NOUN"]], [["NOUN", "NOUN"]], tags=("NOUN",))
    same_summary = {
        0: compute_metrics(
            [["NOUN", "VERB"], ["VERB", "NOUN"]],
            [["NOUN", "VERB"], ["VERB", "NOUN"]],
            tags=("NOUN", "VERB"),
        ),
        1: compute_metrics(
            [["NOUN", "VERB"], ["VERB", "NOUN"]],
            [["NOUN", "VERB"], ["VERB", "NOUN"]],
            tags=("NOUN", "VERB"),
        ),
        2: compute_metrics(
            [["VERB", "NOUN"], ["NOUN", "VERB"]],
            [["VERB", "NOUN"], ["NOUN", "VERB"]],
            tags=("NOUN", "VERB"),
        ),
    }
    with pytest.raises(ValueError, match="same gold corpus"):
        aggregate_three_seeds(same_summary)
