"""Canonical UPOS model initialization, training, prediction, and ensembling."""

from __future__ import annotations

import copy
import math
import os
import random
import re
from dataclasses import dataclass
from os import PathLike
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .config import UPOS_TAGS
from .metrics import compute_metrics
from .tokenization import (
    IGNORE_INDEX,
    AlignedWordProbabilities,
    TokenizedWindows,
    aggregate_window_probabilities,
)

LABEL2ID: dict[str, int] = {label: index for index, label in enumerate(UPOS_TAGS)}
ID2LABEL: dict[int, str] = {index: label for label, index in LABEL2ID.items()}
_PINNED_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class ModelingError(RuntimeError):
    """A model, checkpoint, or training batch violates the tagging contract."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 16
    effective_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    min_epochs: int = 3
    max_epochs: int = 15
    patience: int = 3
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.effective_batch_size < self.batch_size:
            raise ValueError("effective_batch_size must be at least batch_size")
        if self.effective_batch_size % self.batch_size:
            raise ValueError("effective_batch_size must be divisible by batch_size")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must satisfy 0 <= ratio < 1")
        if self.min_epochs <= 0 or self.max_epochs < self.min_epochs:
            raise ValueError("epochs must satisfy 0 < min_epochs <= max_epochs")
        if self.patience <= 0 or self.gradient_clip <= 0:
            raise ValueError("patience and gradient_clip must be positive")

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.effective_batch_size // self.batch_size


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    optimizer_steps: int
    train_loss: float
    dev_metrics: Mapping[str, Any]
    selection_rank: tuple[float, float, float]
    is_best: bool


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: Any
    best_epoch: int
    best_step: int
    optimizer_steps: int
    best_metrics: Mapping[str, Any]
    history: tuple[EpochRecord, ...]
    stopped_early: bool

    @property
    def best_steps(self) -> int:
        """Plural alias used by artifact schemas that record update counts."""
        return self.best_step


@dataclass(frozen=True, slots=True)
class PredictionResult:
    sentence_ids: tuple[Any, ...]
    label_ids: tuple[tuple[int, ...], ...]
    tags: tuple[tuple[str, ...], ...]
    confidences: tuple[tuple[float, ...], ...]
    probabilities: tuple[Any, ...]


def canonical_label_maps(
    labels: Sequence[str] = UPOS_TAGS,
) -> tuple[dict[str, int], dict[int, str]]:
    """Return deterministic contiguous IDs for a canonical label sequence."""
    normalized = tuple(labels)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("labels must be non-empty and unique")
    label2id = {label: index for index, label in enumerate(normalized)}
    return label2id, {index: label for label, index in label2id.items()}


def _is_local_checkpoint(value: str | PathLike[str]) -> bool:
    return os.path.exists(os.fspath(value))


def _source_label_map(config: Any) -> dict[str, int] | None:
    value = getattr(config, "label2id", None)
    if not isinstance(value, Mapping):
        return None
    try:
        return {str(label): int(index) for label, index in value.items()}
    except (TypeError, ValueError):
        return None


def _reset_or_reorder_classifier(
    model: Any,
    source_label2id: Mapping[str, int] | None,
    target_label2id: Mapping[str, int],
) -> None:
    if source_label2id == target_label2id:
        return
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        raise ModelingError("token classifier model does not expose a classifier head")
    if (
        source_label2id is not None
        and set(source_label2id) == set(target_label2id)
        and set(source_label2id.values()) == set(range(len(target_label2id)))
    ):
        outputs = [
            module
            for module in classifier.modules()
            if isinstance(module, torch.nn.Linear) and module.out_features == len(target_label2id)
        ]
        if outputs:
            output = outputs[-1]
            order = torch.as_tensor(
                [source_label2id[label] for label in target_label2id],
                dtype=torch.long,
                device=output.weight.device,
            )
            with torch.no_grad():
                output.weight.copy_(output.weight.index_select(0, order))
                if output.bias is not None:
                    output.bias.copy_(output.bias.index_select(0, order))
            return
    initializer = getattr(model, "_init_weights", None)
    if callable(initializer):
        classifier.apply(initializer)
    else:
        for module in classifier.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()


def initialize_token_classifier(
    model_name_or_path: str | PathLike[str],
    *,
    revision: str | None = None,
    labels: Sequence[str] = UPOS_TAGS,
    model_class: Any | None = None,
    config_class: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Initialize a canonical classifier from a pinned Hub or local checkpoint.

    A local checkpoint may be an adapted masked-language-model checkpoint or a
    transferred token-classification checkpoint. Mismatched heads are recreated;
    compatible transferred heads are retained.
    """
    label2id, id2label = canonical_label_maps(labels)
    local = _is_local_checkpoint(model_name_or_path)
    if not local and (
        not isinstance(revision, str) or not _PINNED_REVISION_RE.fullmatch(revision)
    ):
        raise ValueError("Hub model initialization requires a full 40-character commit")
    source_label2id: dict[str, int] | None = None
    source_num_labels: int | None = None
    if model_class is None:
        from transformers import AutoConfig, AutoModelForTokenClassification

        model_class = AutoModelForTokenClassification
        config_class = config_class or AutoConfig
    elif config_class is None:
        raise ValueError(
            "config_class is required with an injected model_class so source label "
            "metadata cannot bypass head validation"
        )
    if config_class is not None:
        config_kwargs = {
            key: kwargs[key]
            for key in (
                "cache_dir",
                "force_download",
                "local_files_only",
                "token",
                "trust_remote_code",
                "subfolder",
            )
            if key in kwargs
        }
        if revision is not None:
            config_kwargs["revision"] = revision
        source_config = config_class.from_pretrained(
            str(model_name_or_path), **config_kwargs
        )
        source_label2id = _source_label_map(source_config)
        source_num_labels = getattr(source_config, "num_labels", None)
    load_kwargs: dict[str, Any] = {
        **kwargs,
        "num_labels": len(label2id),
        "label2id": label2id,
        "id2label": id2label,
        "ignore_mismatched_sizes": True,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    model = model_class.from_pretrained(str(model_name_or_path), **load_kwargs)
    if source_num_labels == len(label2id):
        _reset_or_reorder_classifier(model, source_label2id, label2id)
    config = getattr(model, "config", None)
    if config is not None:
        config.num_labels = len(label2id)
        config.label2id = dict(label2id)
        config.id2label = dict(id2label)
    return model


def initialize_model(*args: Any, **kwargs: Any) -> Any:
    """Compatibility alias for :func:`initialize_token_classifier`."""
    return initialize_token_classifier(*args, **kwargs)


def set_seed(seed: int) -> None:
    """Seed Python and Torch without requiring a particular accelerator."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sentence_balanced_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sentence_ids: torch.Tensor | Sequence[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Average token CE within each source sentence, then average sentences."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits/labels must have shapes [batch, tokens, labels]/[batch, tokens]")
    if len(sentence_ids) != logits.shape[0]:
        raise ValueError("one source sentence ID is required per window")
    if not torch.is_tensor(sentence_ids):
        sentence_ids = torch.as_tensor(sentence_ids, device=logits.device)
    else:
        sentence_ids = sentence_ids.to(logits.device)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape_as(labels)
    valid = labels.ne(ignore_index)
    sentence_losses: list[torch.Tensor] = []
    for sentence_id in torch.unique(sentence_ids, sorted=False):
        sentence_mask = sentence_ids.eq(sentence_id).unsqueeze(1) & valid
        if sentence_mask.any():
            sentence_losses.append(token_losses[sentence_mask].mean())
    if not sentence_losses:
        raise ModelingError("a sentence-balanced batch contains no training labels")
    return torch.stack(sentence_losses).mean()


def _model_logits(model: Any, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    output = model(**inputs)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise ModelingError("model forward output does not expose logits")


def _batch_tensors(
    tokenized: TokenizedWindows,
    indices: Sequence[int],
    device: torch.device,
    *,
    include_labels: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor]:
    if not indices:
        raise ValueError("cannot create an empty model batch")
    keys = tuple(tokenized.features[indices[0]].keys())
    if not keys or any(tuple(tokenized.features[index].keys()) != keys for index in indices):
        raise ModelingError("all windows in a batch must have identical model input fields")
    inputs = {
        key: torch.as_tensor(
            [tokenized.features[index][key] for index in indices],
            dtype=torch.long,
            device=device,
        )
        for key in keys
    }
    labels = None
    if include_labels:
        if tokenized.labels is None:
            raise ModelingError("training data has no aligned labels")
        labels = torch.as_tensor(
            [tokenized.labels[index] for index in indices],
            dtype=torch.long,
            device=device,
        )
    sentence_ids = torch.as_tensor(
        [tokenized.window_sentence_indices[index] for index in indices],
        dtype=torch.long,
        device=device,
    )
    return inputs, labels, sentence_ids


def _parameter_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay and not no_decay:
        raise ModelingError("model has no trainable parameters")
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def create_optimizer_and_scheduler(
    model: Any,
    *,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Create Torch AdamW and a version-stable linear warmup/decay schedule."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = int(total_steps * warmup_ratio)
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, weight_decay),
        lr=learning_rate,
    )

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / float(decay_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    return optimizer, scheduler


def selection_rank(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """Lexicographic rank: sentence >=98%, token accuracy, exact match."""
    sentence_value = metrics.get("sentence_at_least_98_rate")
    if sentence_value is None and isinstance(metrics.get("sentence_at_least_98"), Mapping):
        sentence_value = metrics["sentence_at_least_98"].get("rate")
    exact_value = metrics.get("exact_match_rate")
    if exact_value is None and isinstance(metrics.get("exact_match"), Mapping):
        exact_value = metrics["exact_match"].get("rate")
    token_value = metrics.get("token_accuracy")
    if sentence_value is None or token_value is None or exact_value is None:
        raise ValueError("metrics lack sentence>=98, token accuracy, or exact-match rate")
    rank = (float(sentence_value), float(token_value), float(exact_value))
    if not all(math.isfinite(value) for value in rank):
        raise ValueError("selection metrics must be finite")
    return rank


def _snapshot_state(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in model.state_dict().items()
    }


def _training_label_counts(tokenized: TokenizedWindows) -> tuple[int, ...]:
    if tokenized.labels is None:
        raise ModelingError("training data has no aligned labels")
    counts = [0] * tokenized.num_sentences
    for window_index, row in enumerate(tokenized.labels):
        sentence_index = tokenized.window_sentence_indices[window_index]
        counts[sentence_index] += sum(label != IGNORE_INDEX for label in row)
    if tuple(counts) != tokenized.word_counts:
        raise ModelingError(
            "training labels must cover every source word exactly once per sentence"
        )
    return tuple(counts)


def _scaled_sentence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sentence_ids: torch.Tensor,
    label_counts: Sequence[int],
    group_sentence_count: int,
) -> torch.Tensor:
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape_as(labels)
    valid = labels.ne(IGNORE_INDEX)
    row_denominators = torch.as_tensor(
        [label_counts[int(sentence_id)] * group_sentence_count for sentence_id in sentence_ids],
        dtype=token_losses.dtype,
        device=token_losses.device,
    )
    return ((token_losses * valid) / row_denominators.unsqueeze(1)).sum()


def predict_probabilities(
    model: Any,
    tokenized: TokenizedWindows,
    *,
    batch_size: int = 32,
    device: str | torch.device | None = None,
    labels: Sequence[str] = UPOS_TAGS,
) -> AlignedWordProbabilities:
    """Predict every window and average all aligned source-word occurrences."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    normalized_labels = tuple(labels)
    if not normalized_labels or len(set(normalized_labels)) != len(normalized_labels):
        raise ValueError("prediction labels must be non-empty and unique")
    target_device = torch.device(device) if device is not None else get_device()
    model.to(target_device)
    model.eval()
    window_probabilities: list[torch.Tensor] = []
    with torch.no_grad():
        for offset in range(0, len(tokenized), batch_size):
            indices = tuple(range(offset, min(offset + batch_size, len(tokenized))))
            inputs, _, _ = _batch_tensors(
                tokenized, indices, target_device, include_labels=False
            )
            logits = _model_logits(model, inputs)
            if logits.ndim != 3 or logits.shape[-1] != len(normalized_labels):
                raise ModelingError(
                    "token classifier logits must be [windows, tokens, configured labels]"
                )
            window_probabilities.append(torch.softmax(logits.float(), dim=-1).cpu())
    if not window_probabilities:
        raise ModelingError("cannot predict an empty tokenization")
    return aggregate_window_probabilities(
        tokenized,
        torch.cat(window_probabilities, dim=0),
        label_names=normalized_labels,
    )


def predictions_from_probabilities(
    aligned: AlignedWordProbabilities,
    *,
    id2label: Mapping[int, str] = ID2LABEL,
) -> PredictionResult:
    """Convert aligned probability vectors into labels and confidences."""
    import numpy as np

    label_rows: list[tuple[int, ...]] = []
    tag_rows: list[tuple[str, ...]] = []
    confidence_rows: list[tuple[float, ...]] = []
    probabilities: list[Any] = []
    try:
        configured_label_names = tuple(id2label[index] for index in range(len(aligned.label_names)))
    except KeyError as error:
        raise ValueError("id2label must define every contiguous probability column") from error
    if configured_label_names != aligned.label_names:
        raise ValueError("id2label order does not match the aligned probability label axis")
    for matrix_value in aligned.probabilities:
        matrix = np.asarray(matrix_value, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] < 1:
            raise ValueError("each aligned probability matrix must be [words, labels]")
        label_ids = tuple(int(value) for value in matrix.argmax(axis=1))
        try:
            tags = tuple(id2label[label_id] for label_id in label_ids)
        except KeyError as error:
            raise ValueError(f"probability label ID {error.args[0]} has no configured tag") from error
        label_rows.append(label_ids)
        tag_rows.append(tags)
        confidence_rows.append(tuple(float(value) for value in matrix.max(axis=1)))
        probabilities.append(matrix)
    return PredictionResult(
        sentence_ids=aligned.sentence_ids,
        label_ids=tuple(label_rows),
        tags=tuple(tag_rows),
        confidences=tuple(confidence_rows),
        probabilities=tuple(probabilities),
    )


def predict(
    model: Any,
    tokenized: TokenizedWindows,
    *,
    batch_size: int = 32,
    device: str | torch.device | None = None,
    id2label: Mapping[int, str] = ID2LABEL,
) -> PredictionResult:
    return predictions_from_probabilities(
        predict_probabilities(
            model,
            tokenized,
            batch_size=batch_size,
            device=device,
            labels=tuple(id2label[index] for index in sorted(id2label)),
        ),
        id2label=id2label,
    )


def evaluate_token_classifier(
    model: Any,
    tokenized: TokenizedWindows,
    *,
    batch_size: int = 32,
    device: str | torch.device | None = None,
    labels: Sequence[str] = UPOS_TAGS,
) -> dict[str, Any]:
    if tokenized.gold_label_ids is None:
        raise ModelingError("evaluation requires complete source-word gold labels")
    _, id2label = canonical_label_maps(labels)
    prediction = predict(
        model, tokenized, batch_size=batch_size, device=device, id2label=id2label
    )
    try:
        gold = tuple(
            tuple(id2label[label_id] for label_id in sentence)
            for sentence in tokenized.gold_label_ids
        )
    except KeyError as error:
        raise ModelingError(f"gold label ID {error.args[0]} is outside the canonical map") from error
    return compute_metrics(gold, prediction.tags, tags=labels)


def train_token_classifier(
    model: Any,
    train_data: TokenizedWindows,
    dev_data: TokenizedWindows,
    *,
    config: TrainingConfig = TrainingConfig(),
    seed: int = 0,
    device: str | torch.device | None = None,
    labels: Sequence[str] = UPOS_TAGS,
    evaluate_fn: Callable[[Any, TokenizedWindows, torch.device], Mapping[str, Any]] | None = None,
) -> TrainingResult:
    """Fine-tune with exact sentence balancing, accumulation, and early stopping."""
    if train_data.labels is None:
        raise ModelingError("training tokenization has no labels")
    if train_data.num_sentences <= 0:
        raise ModelingError("training data has no source sentences")
    set_seed(seed)
    target_device = torch.device(device) if device is not None else get_device()
    model.to(target_device)

    updates_per_epoch = math.ceil(
        train_data.num_sentences / config.effective_batch_size
    )
    total_steps = updates_per_epoch * config.max_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        total_steps=total_steps,
    )
    optimizer.zero_grad(set_to_none=True)

    generator = random.Random(seed)
    label_counts = _training_label_counts(train_data)
    windows_by_sentence = train_data.windows_by_sentence
    history: list[EpochRecord] = []
    best_state: dict[str, Any] | None = None
    best_metrics: Mapping[str, Any] | None = None
    best_rank: tuple[float, float, float] | None = None
    best_epoch = 0
    best_step = 0
    optimizer_steps = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        sentence_order = list(range(train_data.num_sentences))
        generator.shuffle(sentence_order)
        weighted_loss_sum = 0.0
        seen_sentences = 0

        for group_offset in range(
            0, len(sentence_order), config.effective_batch_size
        ):
            sentence_indices = tuple(
                sentence_order[group_offset : group_offset + config.effective_batch_size]
            )
            group_sentence_count = len(sentence_indices)
            window_indices = tuple(
                window_index
                for sentence_index in sentence_indices
                for window_index in windows_by_sentence[sentence_index]
            )
            if not window_indices:
                raise ModelingError("a source-sentence group has no tokenized windows")
            group_loss = 0.0
            for window_offset in range(0, len(window_indices), config.batch_size):
                window_batch = window_indices[
                    window_offset : window_offset + config.batch_size
                ]
                inputs, batch_labels, window_sentence_ids = _batch_tensors(
                    train_data,
                    window_batch,
                    target_device,
                    include_labels=True,
                )
                assert batch_labels is not None
                logits = _model_logits(model, inputs)
                scaled_loss = _scaled_sentence_loss(
                    logits,
                    batch_labels,
                    window_sentence_ids,
                    label_counts,
                    group_sentence_count,
                )
                scaled_loss.backward()
                group_loss += float(scaled_loss.detach()) * group_sentence_count

            weighted_loss_sum += group_loss
            seen_sentences += group_sentence_count
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if seen_sentences != train_data.num_sentences:
            raise ModelingError("an epoch did not train every source sentence exactly once")
        train_loss = weighted_loss_sum / seen_sentences
        if evaluate_fn is None:
            dev_metrics = evaluate_token_classifier(
                model,
                dev_data,
                batch_size=config.batch_size,
                device=target_device,
                labels=labels,
            )
        else:
            dev_metrics = dict(evaluate_fn(model, dev_data, target_device))
        rank = selection_rank(dev_metrics)
        improved = best_rank is None or rank > best_rank
        if improved:
            best_rank = rank
            best_metrics = copy.deepcopy(dev_metrics)
            best_state = _snapshot_state(model)
            best_epoch = epoch
            best_step = optimizer_steps
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            EpochRecord(
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                train_loss=train_loss,
                dev_metrics=copy.deepcopy(dev_metrics),
                selection_rank=rank,
                is_best=improved,
            )
        )
        if epoch >= config.min_epochs and epochs_without_improvement >= config.patience:
            stopped_early = True
            break

    if best_state is None or best_metrics is None:
        raise ModelingError("training completed without a valid dev checkpoint")
    model.load_state_dict(best_state)
    model.to(target_device)
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_step=best_step,
        optimizer_steps=optimizer_steps,
        best_metrics=best_metrics,
        history=tuple(history),
        stopped_early=stopped_early,
    )


def train_model(*args: Any, **kwargs: Any) -> TrainingResult:
    """Compatibility alias for :func:`train_token_classifier`."""
    return train_token_classifier(*args, **kwargs)


def ensemble_probabilities(
    members: Sequence[AlignedWordProbabilities],
    *,
    weights: Sequence[float] | None = None,
) -> AlignedWordProbabilities:
    """Average probability vectors after strict sentence/word/label alignment checks."""
    import numpy as np

    if not members:
        raise ValueError("at least one ensemble member is required")
    reference = members[0]
    if weights is None:
        normalized_weights = np.full(len(members), 1.0 / len(members), dtype=np.float64)
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        if normalized_weights.shape != (len(members),):
            raise ValueError("weights must align with ensemble members")
        if not np.isfinite(normalized_weights).all() or (normalized_weights < 0).any():
            raise ValueError("ensemble weights must be finite and non-negative")
        total = float(normalized_weights.sum())
        if total <= 0:
            raise ValueError("ensemble weights must have a positive sum")
        normalized_weights = normalized_weights / total

    for member in members[1:]:
        if member.sentence_ids != reference.sentence_ids:
            raise ValueError("ensemble sentence IDs are not aligned")
        if member.label_names != reference.label_names:
            raise ValueError("ensemble probability label axes are not aligned")
        if len(member.probabilities) != len(reference.probabilities):
            raise ValueError("ensemble sentence counts are not aligned")

    averaged: list[Any] = []
    occurrence_counts: list[tuple[int, ...]] = []
    for sentence_index, reference_value in enumerate(reference.probabilities):
        reference_matrix = np.asarray(reference_value, dtype=np.float64)
        matrices = []
        for member in members:
            matrix = np.asarray(member.probabilities[sentence_index], dtype=np.float64)
            if matrix.shape != reference_matrix.shape:
                raise ValueError(
                    f"ensemble probability shape mismatch at sentence {reference.sentence_ids[sentence_index]!r}"
                )
            if not np.isfinite(matrix).all() or (matrix < 0).any():
                raise ValueError("ensemble probabilities must be finite and non-negative")
            matrices.append(matrix)
        averaged.append(
            sum(weight * matrix for weight, matrix in zip(normalized_weights, matrices))
        )
        occurrence_counts.append(
            tuple(
                sum(member.occurrence_counts[sentence_index][word_index] for member in members)
                for word_index in range(reference_matrix.shape[0])
            )
        )
    return AlignedWordProbabilities(
        sentence_ids=reference.sentence_ids,
        probabilities=tuple(averaged),
        occurrence_counts=tuple(occurrence_counts),
        label_names=reference.label_names,
    )


def ensemble_predictions(
    members: Sequence[AlignedWordProbabilities],
    *,
    weights: Sequence[float] | None = None,
    id2label: Mapping[int, str] = ID2LABEL,
) -> PredictionResult:
    return predictions_from_probabilities(
        ensemble_probabilities(members, weights=weights), id2label=id2label
    )
