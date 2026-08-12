"""Overflow-safe word alignment for token classification."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from os import PathLike
import re
from typing import Any, Hashable, Mapping, Sequence

IGNORE_INDEX = -100
_PINNED_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_TOKENIZER_METADATA_KEYS = {
    "overflow_to_sample_mapping",
    "num_truncated_tokens",
    "offset_mapping",
    "special_tokens_mask",
    "length",
}


class TokenizationError(RuntimeError):
    """Pretokenized inputs could not be aligned without losing source words."""


@dataclass(frozen=True, slots=True)
class WordOccurrence:
    """The token position used for one source word in one overflow window."""

    window_index: int
    token_index: int
    sentence_index: int
    sentence_id: Hashable
    word_index: int


@dataclass(frozen=True, slots=True)
class TokenizedWindows:
    """Fixed-length model features plus complete source-word alignment metadata."""

    features: tuple[Mapping[str, tuple[int, ...]], ...]
    labels: tuple[tuple[int, ...], ...] | None
    sentence_ids: tuple[Hashable, ...]
    word_counts: tuple[int, ...]
    window_sentence_indices: tuple[int, ...]
    word_ids: tuple[tuple[int | None, ...], ...]
    occurrences: tuple[WordOccurrence, ...]
    gold_label_ids: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        window_count = len(self.features)
        if len(self.window_sentence_indices) != window_count or len(self.word_ids) != window_count:
            raise ValueError("window metadata must align with features")
        if self.labels is not None and len(self.labels) != window_count:
            raise ValueError("window labels must align with features")
        if len(self.sentence_ids) != len(self.word_counts):
            raise ValueError("sentence IDs must align with source word counts")
        if self.gold_label_ids is not None and len(self.gold_label_ids) != len(self.sentence_ids):
            raise ValueError("gold labels must align with source sentences")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, tuple[int, ...] | int]:
        item: dict[str, tuple[int, ...] | int] = dict(self.features[index])
        if self.labels is not None:
            item["labels"] = self.labels[index]
        item["sentence_index"] = self.window_sentence_indices[index]
        return item

    @property
    def num_sentences(self) -> int:
        return len(self.sentence_ids)

    @property
    def num_words(self) -> int:
        return sum(self.word_counts)

    @property
    def windows_by_sentence(self) -> tuple[tuple[int, ...], ...]:
        grouped: list[list[int]] = [[] for _ in self.sentence_ids]
        for window_index, sentence_index in enumerate(self.window_sentence_indices):
            grouped[sentence_index].append(window_index)
        return tuple(tuple(indices) for indices in grouped)


@dataclass(frozen=True, slots=True)
class AlignedWordProbabilities:
    """Probability matrices in original sentence and word order."""

    sentence_ids: tuple[Hashable, ...]
    probabilities: tuple[Any, ...]
    occurrence_counts: tuple[tuple[int, ...], ...]
    label_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.sentence_ids)
            == len(self.probabilities)
            == len(self.occurrence_counts)
        ):
            raise ValueError("aligned probability fields must have equal sentence counts")
        label_count: int | None = None
        for matrix, counts in zip(self.probabilities, self.occurrence_counts):
            if len(matrix) != len(counts):
                raise ValueError("each probability row must correspond to one source word")
            shape = getattr(matrix, "shape", None)
            width = int(shape[1]) if shape is not None and len(shape) == 2 else len(matrix[0])
            if label_count is None:
                label_count = width
            elif width != label_count:
                raise ValueError("all aligned matrices must use the same label axis")
        if label_count is None or label_count < 1:
            raise ValueError("aligned probabilities must include at least one label")
        if len(self.label_names) != label_count:
            raise ValueError("explicit label names must align with the probability label axis")
        if len(set(self.label_names)) != len(self.label_names):
            raise ValueError("probability label names must be unique")


def create_fast_tokenizer(
    model_name_or_path: str | PathLike[str],
    *,
    revision: str,
    add_prefix_space: bool,
    tokenizer_class: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Load a fast tokenizer with an explicit immutable revision and spacing policy."""
    if not isinstance(revision, str) or not _PINNED_REVISION_RE.fullmatch(revision):
        raise ValueError("revision must be a full 40-character pinned commit")
    if not isinstance(add_prefix_space, bool):
        raise TypeError("add_prefix_space must be an explicit bool")
    if tokenizer_class is None:
        from transformers import AutoTokenizer

        tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        str(model_name_or_path),
        revision=revision,
        add_prefix_space=add_prefix_space,
        use_fast=True,
        **kwargs,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise TokenizationError("word alignment requires a fast tokenizer")
    return tokenizer


def _as_windows(value: Any) -> list[list[int]]:
    values = value.tolist() if hasattr(value, "tolist") else list(value)
    if not values:
        return []
    first = values[0]
    if isinstance(first, Integral):
        return [list(values)]
    return [list(row) for row in values]


def _encoding_word_ids(encoding: Any, batch_index: int) -> list[int | None]:
    method = getattr(encoding, "word_ids", None)
    if method is None:
        raise TokenizationError("fast tokenizer output does not expose word_ids()")
    try:
        values = method(batch_index=batch_index)
    except TypeError:
        values = method(batch_index)
    if values is None:
        raise TokenizationError(f"tokenizer returned no word IDs for window {batch_index}")
    return list(values)


def _validate_inputs(
    words: Sequence[Sequence[str]],
    labels: Sequence[Sequence[str | int]] | None,
    sentence_ids: Sequence[Hashable] | None,
    *,
    max_length: int,
    stride: int,
) -> tuple[tuple[Hashable, ...], tuple[tuple[str, ...], ...]]:
    if max_length < 3:
        raise ValueError("max_length must leave room for content and special tokens")
    if stride < 0 or stride >= max_length - 2:
        raise ValueError("stride must satisfy 0 <= stride < max_length - 2")
    normalized_words = tuple(tuple(sentence) for sentence in words)
    if not normalized_words:
        raise ValueError("at least one sentence is required")
    for sentence in normalized_words:
        if not sentence or any(not isinstance(word, str) for word in sentence):
            raise ValueError("every sentence must contain one or more string words")
    if labels is not None:
        if len(labels) != len(normalized_words):
            raise ValueError("labels must align with sentences")
        for sentence_words, sentence_labels in zip(normalized_words, labels):
            if len(sentence_words) != len(sentence_labels):
                raise ValueError("every source word must have exactly one gold label")
    ids = tuple(range(len(normalized_words))) if sentence_ids is None else tuple(sentence_ids)
    if len(ids) != len(normalized_words):
        raise ValueError("sentence_ids must align with sentences")
    try:
        unique_ids = set(ids)
    except TypeError as error:
        raise TypeError("sentence IDs must be hashable") from error
    if len(unique_ids) != len(ids):
        raise ValueError("sentence IDs must be unique")
    return ids, normalized_words


def _encode_gold_labels(
    labels: Sequence[Sequence[str | int]] | None,
    label2id: Mapping[str, int] | None,
) -> tuple[tuple[int, ...], ...] | None:
    if labels is None:
        return None
    encoded: list[tuple[int, ...]] = []
    for sentence in labels:
        row: list[int] = []
        for label in sentence:
            if isinstance(label, str):
                if label2id is None:
                    raise ValueError("label2id is required for string labels")
                try:
                    value = label2id[label]
                except KeyError as error:
                    raise ValueError(f"unknown label {label!r}") from error
            elif isinstance(label, Integral) and not isinstance(label, bool):
                value = int(label)
            else:
                raise TypeError(f"labels must be strings or integer IDs, got {label!r}")
            if value < 0:
                raise ValueError("gold label IDs must be non-negative")
            row.append(int(value))
        encoded.append(tuple(row))
    return tuple(encoded)


def tokenize_pretokenized(
    words: Sequence[Sequence[str]],
    tokenizer: Any,
    *,
    max_length: int,
    stride: int,
    labels: Sequence[Sequence[str | int]] | None = None,
    label2id: Mapping[str, int] | None = None,
    sentence_ids: Sequence[Hashable] | None = None,
) -> TokenizedWindows:
    """Tokenize words into overlapping windows without dropping or double-labeling words.

    Every word contributes one training target globally. For inference, one
    occurrence per word per window is retained so probabilities can be averaged
    over all overlapping contexts.
    """
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise TokenizationError("pretokenized alignment requires a fast tokenizer")
    ids, normalized_words = _validate_inputs(
        words, labels, sentence_ids, max_length=max_length, stride=stride
    )
    gold_label_ids = _encode_gold_labels(labels, label2id)

    features: list[Mapping[str, tuple[int, ...]]] = []
    window_labels: list[tuple[int, ...]] = []
    window_sentence_indices: list[int] = []
    all_word_ids: list[tuple[int | None, ...]] = []
    occurrences: list[WordOccurrence] = []
    labeled_words: set[tuple[int, int]] = set()

    for sentence_index, sentence_words in enumerate(normalized_words):
        encoding = tokenizer(
            list(sentence_words),
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding="max_length",
            return_attention_mask=True,
        )
        if "input_ids" not in encoding:
            raise TokenizationError("tokenizer output is missing input_ids")
        input_windows = _as_windows(encoding["input_ids"])
        if not input_windows:
            raise TokenizationError(f"tokenizer produced no windows for sentence {ids[sentence_index]!r}")

        encoded_fields: dict[str, list[list[int]]] = {}
        for key, value in encoding.items():
            if key in _TOKENIZER_METADATA_KEYS:
                continue
            try:
                rows = _as_windows(value)
            except (TypeError, ValueError):
                continue
            if len(rows) == len(input_windows) and all(
                len(row) == len(input_windows[index]) for index, row in enumerate(rows)
            ):
                encoded_fields[key] = rows
        if "input_ids" not in encoded_fields:
            encoded_fields["input_ids"] = input_windows

        for local_window_index, input_ids in enumerate(input_windows):
            word_ids = _encoding_word_ids(encoding, local_window_index)
            if len(word_ids) != len(input_ids):
                raise TokenizationError("word IDs do not align with token IDs")
            global_window_index = len(features)
            feature = {
                key: tuple(rows[local_window_index]) for key, rows in encoded_fields.items()
            }
            features.append(feature)
            window_sentence_indices.append(sentence_index)
            all_word_ids.append(tuple(word_ids))

            first_token_by_word: dict[int, int] = {}
            for token_index, word_index in enumerate(word_ids):
                if word_index is None:
                    continue
                if not isinstance(word_index, Integral) or not 0 <= word_index < len(sentence_words):
                    raise TokenizationError(
                        f"invalid word index {word_index!r} for sentence {ids[sentence_index]!r}"
                    )
                word_index = int(word_index)
                first_token_by_word.setdefault(word_index, token_index)
            for word_index, token_index in first_token_by_word.items():
                occurrences.append(
                    WordOccurrence(
                        window_index=global_window_index,
                        token_index=token_index,
                        sentence_index=sentence_index,
                        sentence_id=ids[sentence_index],
                        word_index=word_index,
                    )
                )

            if gold_label_ids is not None:
                aligned = [IGNORE_INDEX] * len(word_ids)
                for word_index, token_index in first_token_by_word.items():
                    key = (sentence_index, word_index)
                    if key not in labeled_words:
                        aligned[token_index] = gold_label_ids[sentence_index][word_index]
                        labeled_words.add(key)
                window_labels.append(tuple(aligned))

    result = TokenizedWindows(
        features=tuple(features),
        labels=tuple(window_labels) if gold_label_ids is not None else None,
        sentence_ids=ids,
        word_counts=tuple(len(sentence) for sentence in normalized_words),
        window_sentence_indices=tuple(window_sentence_indices),
        word_ids=tuple(all_word_ids),
        occurrences=tuple(occurrences),
        gold_label_ids=gold_label_ids,
    )
    assert_complete_word_coverage(result)
    if gold_label_ids is not None:
        expected = result.num_words
        assigned = sum(label != IGNORE_INDEX for row in result.labels or () for label in row)
        if assigned != expected:
            raise TokenizationError(
                f"expected exactly {expected} training labels, assigned {assigned}"
            )
    return result


def tokenize_sentences(
    sentences: Sequence[Any],
    tokenizer: Any,
    *,
    max_length: int,
    stride: int,
    label2id: Mapping[str, int] | None = None,
    sentence_ids: Sequence[Hashable] | None = None,
    labeled: bool | None = None,
) -> TokenizedWindows:
    """Adapt :class:`Sentence` objects to :func:`tokenize_pretokenized`."""
    words = [tuple(token.form for token in sentence.tokens) for sentence in sentences]
    if labeled is None:
        labeled = label2id is not None
    labels: list[tuple[str, ...]] | None = None
    if labeled:
        labels = []
        for sentence in sentences:
            try:
                labels.append(tuple(token.upos for token in sentence.tokens))
            except AttributeError as error:
                raise TokenizationError("labeled tokenization received an unlabeled token") from error
    if sentence_ids is None:
        sentence_ids = tuple(
            (
                sentence.provenance.source_file,
                sentence.provenance.sentence_index,
                sentence.provenance.sent_id,
            )
            for sentence in sentences
        )
    return tokenize_pretokenized(
        words,
        tokenizer,
        max_length=max_length,
        stride=stride,
        labels=labels,
        label2id=label2id,
        sentence_ids=sentence_ids,
    )


def assert_complete_word_coverage(tokenized: TokenizedWindows) -> None:
    """Raise if any source word has no retained inference occurrence."""
    covered = {(item.sentence_index, item.word_index) for item in tokenized.occurrences}
    missing = [
        (tokenized.sentence_ids[sentence_index], word_index)
        for sentence_index, count in enumerate(tokenized.word_counts)
        for word_index in range(count)
        if (sentence_index, word_index) not in covered
    ]
    if missing:
        preview = ", ".join(repr(item) for item in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise TokenizationError(f"{len(missing)} source words have no tokenized occurrence: {preview}{suffix}")


def aggregate_window_probabilities(
    tokenized: TokenizedWindows,
    window_probabilities: Any,
    *,
    label_names: Sequence[str],
) -> AlignedWordProbabilities:
    """Average class probabilities across every occurrence of each source word."""
    import numpy as np

    if hasattr(window_probabilities, "detach"):
        window_probabilities = window_probabilities.detach().cpu().numpy()
    probabilities = np.asarray(window_probabilities, dtype=np.float64)
    if probabilities.ndim != 3:
        raise ValueError("window probabilities must have shape [windows, tokens, labels]")
    if probabilities.shape[0] != len(tokenized):
        raise ValueError("probability window count does not match tokenization")
    if probabilities.shape[2] < 1:
        raise ValueError("probabilities must include at least one label")
    normalized_label_names = tuple(label_names)
    if len(normalized_label_names) != probabilities.shape[2]:
        raise ValueError("label_names must align with the probability label axis")
    if len(set(normalized_label_names)) != len(normalized_label_names):
        raise ValueError("label_names must be unique")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and non-negative")

    sums = [np.zeros((count, probabilities.shape[2]), dtype=np.float64) for count in tokenized.word_counts]
    counts = [np.zeros(count, dtype=np.int64) for count in tokenized.word_counts]
    for occurrence in tokenized.occurrences:
        if occurrence.token_index >= probabilities.shape[1]:
            raise ValueError("probability token dimension does not cover an occurrence")
        vector = probabilities[occurrence.window_index, occurrence.token_index]
        sums[occurrence.sentence_index][occurrence.word_index] += vector
        counts[occurrence.sentence_index][occurrence.word_index] += 1

    missing = [
        (tokenized.sentence_ids[sentence_index], word_index)
        for sentence_index, sentence_counts in enumerate(counts)
        for word_index, count in enumerate(sentence_counts)
        if count == 0
    ]
    if missing:
        raise TokenizationError(f"inference probabilities miss {len(missing)} source words")

    averaged = tuple(total / count[:, None] for total, count in zip(sums, counts))
    return AlignedWordProbabilities(
        sentence_ids=tokenized.sentence_ids,
        probabilities=averaged,
        occurrence_counts=tuple(tuple(int(value) for value in row) for row in counts),
        label_names=normalized_label_names,
    )
