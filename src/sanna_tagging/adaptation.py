"""Leakage-safe Czech LAPT and bounded Slovak source transfer."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

import torch

from . import modeling as _modeling
from .artifacts import atomic_write_json, sha256_json
from .config import UPOS_TAGS
from .data import Sentence, Token, ordered_form_hash
from .modeling import (
    LABEL2ID,
    create_optimizer_and_scheduler,
    get_device,
    initialize_token_classifier,
    selection_rank,
    set_seed,
)
from .tokenization import IGNORE_INDEX, TokenizedWindows

_PINNED_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_LAPT_SEAL = object()


class AdaptationError(RuntimeError):
    """An adaptation input or checkpoint violates the experiment contract."""


@dataclass(frozen=True, slots=True)
class _LAPTTextRow:
    forms: tuple[str, ...]
    identity: str


@dataclass(frozen=True, slots=True)
class LAPTTrainingText:
    """Opaque FORM-only Czech training text accepted by the LAPT trainer.

    Instances can only be produced by :func:`project_czech_training_forms`.
    They contain neither token objects nor a split selector, so downstream LAPT
    APIs cannot be passed labeled, development, or test prepared artifacts.
    """

    _rows: tuple[_LAPTTextRow, ...]
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _LAPT_SEAL:
            raise TypeError("LAPTTrainingText must be created by project_czech_training_forms")

    def __len__(self) -> int:
        return len(self._rows)


@dataclass(frozen=True, slots=True)
class LAPTConfig:
    max_updates: int = 2000
    eval_every: int = 100
    holdout_sentences: int = 1000
    holdout_seed: int = 173
    masking_probability: float = 0.15
    batch_size: int = 16
    effective_batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    gradient_clip: float = 1.0
    max_length: int = 256
    seed: int = 173

    def __post_init__(self) -> None:
        if not 1 <= self.max_updates <= 2000:
            raise ValueError("max_updates must be in 1..2000")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.holdout_sentences <= 0:
            raise ValueError("holdout_sentences must be positive")
        if not 0 < self.masking_probability < 1:
            raise ValueError("masking_probability must be between zero and one")
        if self.batch_size <= 0 or self.effective_batch_size < self.batch_size:
            raise ValueError("effective_batch_size must be at least batch_size")
        if self.effective_batch_size % self.batch_size:
            raise ValueError("effective_batch_size must be divisible by batch_size")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must satisfy 0 <= ratio < 1")
        if self.gradient_clip <= 0 or self.max_length < 3:
            raise ValueError("gradient_clip must be positive and max_length at least three")

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.effective_batch_size // self.batch_size


@dataclass(frozen=True, slots=True)
class MLMEvaluation:
    step: int
    loss: float
    is_best: bool


@dataclass(frozen=True, slots=True)
class LAPTResult:
    model: Any
    checkpoint_dir: Path
    encoder_dir: Path
    metadata_path: Path
    best_step: int
    best_holdout_loss: float
    optimizer_steps: int
    history: tuple[MLMEvaluation, ...]


@dataclass(frozen=True, slots=True)
class TransferConfig:
    updates: int = 750
    eval_every: int = 50
    batch_size: int = 16
    effective_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    gradient_clip: float = 1.0
    seed: int = 404

    def __post_init__(self) -> None:
        if not 1 <= self.updates <= 750:
            raise ValueError("updates must be in 1..750")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.batch_size <= 0 or self.effective_batch_size < self.batch_size:
            raise ValueError("effective_batch_size must be at least batch_size")
        if self.effective_batch_size % self.batch_size:
            raise ValueError("effective_batch_size must be divisible by batch_size")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must satisfy 0 <= ratio < 1")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")


@dataclass(frozen=True, slots=True)
class TransferEvaluation:
    step: int
    train_loss: float
    dev_metrics: Mapping[str, Any]
    selection_rank: tuple[float, float, float]
    is_best: bool


@dataclass(frozen=True, slots=True)
class TransferResult:
    model: Any
    initialization: Literal["base", "lapt"]
    best_step: int
    optimizer_steps: int
    best_metrics: Mapping[str, Any]
    history: tuple[TransferEvaluation, ...]
    optimizer_state_preserved: Literal[False] = False


def project_czech_training_forms(sentences: Sequence[Sentence]) -> LAPTTrainingText:
    """Accept already FORM-only cleaned Czech train data at the LAPT boundary."""
    if not sentences:
        raise ValueError("Czech LAPT requires at least two cleaned training sentences")
    rows: list[_LAPTTextRow] = []
    seen: set[str] = set()
    for sentence in sentences:
        if type(sentence) is not Sentence:
            raise TypeError("LAPT projection accepts only prepared Sentence objects")
        provenance = sentence.provenance
        if provenance.split.casefold() != "train":
            raise AdaptationError("LAPT projection accepts the training split only")
        source = provenance.source_file.replace("\\", "/").casefold()
        if not source.startswith("czech/train/"):
            raise AdaptationError("LAPT projection accepts cleaned Czech training sources only")
        if any(type(token) is not Token for token in sentence.tokens):
            raise AdaptationError(
                "LAPT projection accepts plain FORM-only Token objects; labels are forbidden"
            )
        forms = sentence.forms
        if not forms or any(not isinstance(form, str) or not form for form in forms):
            raise AdaptationError("every LAPT sentence must contain non-empty FORM values")
        form_hash = sentence.form_hash
        if form_hash in seen:
            raise AdaptationError("LAPT text must be cleaned of duplicate FORM sentences")
        seen.add(form_hash)
        identity = sha256_json(
            {
                "form_hash": form_hash,
                "source_file": provenance.source_file,
                "sentence_index": provenance.sentence_index,
                "sent_id": provenance.sent_id,
            }
        )
        rows.append(_LAPTTextRow(forms=forms, identity=identity))
    if len(rows) < 2:
        raise ValueError("Czech LAPT requires at least two cleaned training sentences")
    return LAPTTrainingText(tuple(rows), _LAPT_SEAL)


def _require_lapt_text(value: Any) -> LAPTTrainingText:
    if type(value) is not LAPTTrainingText or value._seal is not _LAPT_SEAL:
        raise TypeError("train_lapt accepts only project_czech_training_forms output")
    return value


def split_lapt_holdout(
    text: LAPTTrainingText, *, holdout_sentences: int, seed: int
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    """Deterministically split the opaque training text; no external dev is accepted."""
    corpus = _require_lapt_text(text)
    if not 1 <= holdout_sentences < len(corpus):
        raise ValueError("holdout_sentences must leave at least one LAPT training sentence")
    ranked = sorted(
        range(len(corpus._rows)),
        key=lambda index: (
            hashlib.sha256(f"{seed}\0{corpus._rows[index].identity}".encode("utf-8")).digest(),
            corpus._rows[index].identity,
        ),
    )
    held_out = set(ranked[:holdout_sentences])
    train = tuple(row.forms for index, row in enumerate(corpus._rows) if index not in held_out)
    holdout = tuple(row.forms for index, row in enumerate(corpus._rows) if index in held_out)
    return train, holdout


def initialize_masked_lm(
    model_name_or_path: str | os.PathLike[str],
    *,
    revision: str | None = None,
    model_class: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Initialize AutoModelForMaskedLM from a pinned Hub commit or local path."""
    local = os.path.exists(os.fspath(model_name_or_path))
    if not local and (
        not isinstance(revision, str) or not _PINNED_REVISION_RE.fullmatch(revision)
    ):
        raise ValueError("Hub masked-LM initialization requires a full 40-character commit")
    if model_class is None:
        from transformers import AutoModelForMaskedLM

        model_class = AutoModelForMaskedLM
    load_kwargs = dict(kwargs)
    if revision is not None:
        load_kwargs["revision"] = revision
    return model_class.from_pretrained(str(model_name_or_path), **load_kwargs)


def _to_int_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    values = list(value)
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise AdaptationError("LAPT tokenizer returned an unexpected batch dimension")
        values = list(values[0])
    return [int(item) for item in values]


def _tokenize_lapt_rows(
    rows: Sequence[Sequence[str]], tokenizer: Any, *, max_length: int
) -> tuple[dict[str, tuple[int, ...]], ...]:
    encoded_rows: list[dict[str, tuple[int, ...]]] = []
    for forms in rows:
        encoding = tokenizer(
            list(forms),
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        if "input_ids" not in encoding:
            raise AdaptationError("LAPT tokenizer output is missing input_ids")
        input_ids = _to_int_list(encoding["input_ids"])
        if not input_ids:
            raise AdaptationError("LAPT tokenizer produced an empty sentence")
        row: dict[str, tuple[int, ...]] = {"input_ids": tuple(input_ids)}
        for key, value in encoding.items():
            if key == "input_ids":
                continue
            try:
                values = _to_int_list(value)
            except (TypeError, ValueError):
                continue
            if len(values) == len(input_ids):
                row[key] = tuple(values)
        if "attention_mask" not in row:
            row["attention_mask"] = tuple(1 for _ in input_ids)
        if "special_tokens_mask" not in row:
            method = getattr(tokenizer, "get_special_tokens_mask", None)
            if not callable(method):
                raise AdaptationError("tokenizer must expose special-token masking")
            row["special_tokens_mask"] = tuple(
                int(value) for value in method(input_ids, already_has_special_tokens=True)
            )
        encoded_rows.append(row)
    return tuple(encoded_rows)


def _collate_mlm(
    rows: Sequence[Mapping[str, Sequence[int]]], tokenizer: Any
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not rows:
        raise ValueError("cannot collate an empty MLM batch")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise AdaptationError("LAPT tokenizer must define pad_token_id")
    width = max(len(row["input_ids"]) for row in rows)
    model_keys = sorted(set.intersection(*(set(row) for row in rows)) - {"special_tokens_mask"})
    inputs: dict[str, torch.Tensor] = {}
    for key in model_keys:
        pad_value = int(pad_id) if key == "input_ids" else 0
        inputs[key] = torch.as_tensor(
            [list(row[key]) + [pad_value] * (width - len(row[key])) for row in rows],
            dtype=torch.long,
        )
    special = torch.as_tensor(
        [
            list(row["special_tokens_mask"])
            + [1] * (width - len(row["special_tokens_mask"]))
            for row in rows
        ],
        dtype=torch.bool,
    )
    return inputs, special


def _mask_mlm_batch(
    inputs: Mapping[str, torch.Tensor],
    special_tokens_mask: torch.Tensor,
    tokenizer: Any,
    *,
    probability: float,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise AdaptationError("LAPT tokenizer must define mask_token_id")
    input_ids = inputs["input_ids"].clone()
    labels = input_ids.clone()
    attention = inputs.get("attention_mask", torch.ones_like(input_ids)).bool()
    candidates = attention & ~special_tokens_mask
    selected = (torch.rand(input_ids.shape, generator=generator) < probability) & candidates
    if candidates.any() and not selected.any():
        candidate_indices = candidates.nonzero(as_tuple=False)
        choice = int(torch.randint(len(candidate_indices), (1,), generator=generator))
        selected[tuple(candidate_indices[choice].tolist())] = True
    if not selected.any():
        raise AdaptationError("MLM batch contains no maskable tokens")
    labels[~selected] = IGNORE_INDEX

    selected_indices = selected.nonzero(as_tuple=False)
    replacement_draw = torch.rand(len(selected_indices), generator=generator)
    mask_indices = selected_indices[replacement_draw < 0.8]
    input_ids[mask_indices[:, 0], mask_indices[:, 1]] = int(mask_id)
    random_indices = selected_indices[
        (replacement_draw >= 0.8) & (replacement_draw < 0.9)
    ]
    if len(random_indices):
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            try:
                vocab_size = len(tokenizer)
            except TypeError as error:
                raise AdaptationError("LAPT tokenizer must expose its vocabulary size") from error
        random_tokens = torch.randint(
            int(vocab_size), (len(random_indices),), generator=generator, dtype=torch.long
        )
        input_ids[random_indices[:, 0], random_indices[:, 1]] = random_tokens
    masked_inputs = dict(inputs)
    masked_inputs["input_ids"] = input_ids
    return masked_inputs, labels


def _extract_loss(output: Any) -> torch.Tensor:
    if hasattr(output, "loss"):
        loss = output.loss
    elif isinstance(output, Mapping) and "loss" in output:
        loss = output["loss"]
    elif isinstance(output, (tuple, list)) and output:
        loss = output[0]
    else:
        raise AdaptationError("masked-LM output does not expose loss")
    if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
        raise AdaptationError("masked-LM loss must be a finite scalar tensor")
    return loss


def _shuffled_groups(
    size: int, group_size: int, rng: random.Random
) -> Iterator[tuple[int, ...]]:
    while True:
        order = list(range(size))
        rng.shuffle(order)
        for offset in range(0, size, group_size):
            yield tuple(order[offset : offset + group_size])


def _evaluate_mlm(
    model: Any,
    rows: Sequence[Mapping[str, Sequence[int]]],
    tokenizer: Any,
    *,
    config: LAPTConfig,
    device: torch.device,
    step: int,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 1_000_003 * step)
    losses: list[tuple[float, int]] = []
    with torch.no_grad():
        for offset in range(0, len(rows), config.batch_size):
            raw_inputs, special = _collate_mlm(
                rows[offset : offset + config.batch_size], tokenizer
            )
            masked, labels = _mask_mlm_batch(
                raw_inputs,
                special,
                tokenizer,
                probability=config.masking_probability,
                generator=generator,
            )
            batch_count = labels.shape[0]
            output = model(
                **{key: value.to(device) for key, value in masked.items()},
                labels=labels.to(device),
            )
            losses.append((float(_extract_loss(output).detach().cpu()), batch_count))
    return sum(loss * count for loss, count in losses) / sum(count for _, count in losses)


def _save_component(component: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save = getattr(component, "save_pretrained", None)
    if callable(save):
        save(str(directory))
    elif hasattr(component, "state_dict"):
        torch.save(component.state_dict(), directory / "pytorch_model.bin")
    else:
        raise AdaptationError("model component cannot be checkpointed")


def train_lapt(
    model: Any,
    tokenizer: Any,
    training_text: LAPTTrainingText,
    output_dir: str | os.PathLike[str],
    *,
    config: LAPTConfig = LAPTConfig(),
    model_source: str,
    model_revision: str | None,
    device: str | torch.device | None = None,
) -> LAPTResult:
    """Run bounded dynamic-mask LAPT with an internal training-only holdout."""
    corpus = _require_lapt_text(training_text)
    train_forms, holdout_forms = split_lapt_holdout(
        corpus,
        holdout_sentences=config.holdout_sentences,
        seed=config.holdout_seed,
    )
    train_rows = _tokenize_lapt_rows(train_forms, tokenizer, max_length=config.max_length)
    holdout_rows = _tokenize_lapt_rows(holdout_forms, tokenizer, max_length=config.max_length)
    set_seed(config.seed)
    target_device = torch.device(device) if device is not None else get_device()
    model.to(target_device)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        total_steps=config.max_updates,
    )
    optimizer.zero_grad(set_to_none=True)
    batch_stream = _shuffled_groups(
        len(train_rows), config.batch_size, random.Random(config.seed)
    )
    mask_generator = torch.Generator().manual_seed(config.seed)
    output_root = Path(output_dir)
    checkpoint_dir = output_root / "checkpoint"
    encoder_dir = output_root / "encoder"
    metadata_path = output_root / "metadata.json"
    temporary_state = output_root / ".best-state.pt"
    output_root.mkdir(parents=True, exist_ok=True)

    history: list[MLMEvaluation] = []
    best_loss = math.inf
    best_step = 0
    try:
        for step in range(1, config.max_updates + 1):
            model.train()
            for _ in range(config.gradient_accumulation_steps):
                indices = next(batch_stream)
                raw_inputs, special = _collate_mlm(
                    [train_rows[index] for index in indices], tokenizer
                )
                masked, labels = _mask_mlm_batch(
                    raw_inputs,
                    special,
                    tokenizer,
                    probability=config.masking_probability,
                    generator=mask_generator,
                )
                output = model(
                    **{key: value.to(target_device) for key, value in masked.items()},
                    labels=labels.to(target_device),
                )
                loss = _extract_loss(output) / config.gradient_accumulation_steps
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % config.eval_every == 0 or step == config.max_updates:
                holdout_loss = _evaluate_mlm(
                    model,
                    holdout_rows,
                    tokenizer,
                    config=config,
                    device=target_device,
                    step=step,
                )
                improved = holdout_loss < best_loss
                if improved:
                    best_loss = holdout_loss
                    best_step = step
                    torch.save(model.state_dict(), temporary_state)
                history.append(MLMEvaluation(step=step, loss=holdout_loss, is_best=improved))
        if best_step == 0 or not temporary_state.is_file():
            raise AdaptationError("LAPT completed without a held-out checkpoint")
        try:
            best_state = torch.load(temporary_state, map_location="cpu", weights_only=True)
        except TypeError:  # Torch versions before weights_only support.
            best_state = torch.load(temporary_state, map_location="cpu")
        model.load_state_dict(best_state)
        model.to(target_device)
    finally:
        temporary_state.unlink(missing_ok=True)

    _save_component(model, checkpoint_dir)
    encoder = getattr(model, "base_model", None)
    _save_component(encoder if encoder is not None else model, encoder_dir)
    tokenizer_save = getattr(tokenizer, "save_pretrained", None)
    if callable(tokenizer_save):
        tokenizer_save(str(checkpoint_dir))
        tokenizer_save(str(encoder_dir))

    train_hashes = [ordered_form_hash(forms) for forms in train_forms]
    holdout_hashes = [ordered_form_hash(forms) for forms in holdout_forms]
    metadata = {
        "schema_version": 1,
        "stage": "czech-lapt",
        "model_source": model_source,
        "model_revision": model_revision,
        "config": asdict(config),
        "training_sentences": len(train_forms),
        "holdout_sentences": len(holdout_forms),
        "training_form_hashes": train_hashes,
        "holdout_form_hashes": holdout_hashes,
        "best_step": best_step,
        "optimizer_steps": config.max_updates,
        "best_holdout_loss": best_loss,
        "history": [asdict(record) for record in history],
        "labels_accessed": False,
        "holdout_source": "deterministic-czech-training-only",
        "masking": "dynamic-80-10-10",
    }
    atomic_write_json(metadata_path, metadata)
    return LAPTResult(
        model=model,
        checkpoint_dir=checkpoint_dir,
        encoder_dir=encoder_dir,
        metadata_path=metadata_path,
        best_step=best_step,
        best_holdout_loss=best_loss,
        optimizer_steps=config.max_updates,
        history=tuple(history),
    )


def initialize_transfer_classifier(
    model_name_or_path: str | os.PathLike[str],
    *,
    initialization: Literal["base", "lapt"],
    revision: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create the canonical 17-label source classifier from base or LAPT init."""
    if initialization not in ("base", "lapt"):
        raise ValueError("initialization must be 'base' or 'lapt'")
    if initialization == "lapt" and not os.path.exists(os.fspath(model_name_or_path)):
        raise ValueError("LAPT initialization must be a local saved checkpoint")
    model = initialize_token_classifier(
        model_name_or_path,
        revision=revision,
        labels=UPOS_TAGS,
        **kwargs,
    )
    _assert_canonical_classifier(model)
    return model


def _assert_canonical_classifier(model: Any) -> None:
    config = getattr(model, "config", None)
    if config is None:
        raise AdaptationError("transfer model must expose label configuration")
    source = getattr(config, "label2id", None)
    source_ids = getattr(config, "id2label", None)
    try:
        normalized = {str(label): int(index) for label, index in source.items()}
        normalized_ids = {int(index): str(label) for index, label in source_ids.items()}
    except (AttributeError, TypeError, ValueError) as error:
        raise AdaptationError("transfer model has invalid label ID metadata") from error
    expected_ids = {index: label for label, index in LABEL2ID.items()}
    if (
        normalized != LABEL2ID
        or normalized_ids != expected_ids
        or int(getattr(config, "num_labels", -1)) != len(UPOS_TAGS)
    ):
        raise AdaptationError("transfer requires the canonical 17-label UPOS ID map")


def _validate_transfer_data(data: TokenizedWindows, *, role: str) -> None:
    if data.labels is None or data.gold_label_ids is None:
        raise AdaptationError(f"Slovak {role} data must carry complete canonical labels")
    found_label = False
    for row in data.gold_label_ids:
        for label in row:
            found_label = True
            if not 0 <= label < len(UPOS_TAGS):
                raise AdaptationError(
                    f"Slovak {role} labels fall outside canonical UPOS IDs"
                )
    if not found_label:
        raise AdaptationError(f"Slovak {role} data has no gold labels")


def train_slovak_transfer(
    model: Any,
    slovak_train: TokenizedWindows,
    slovak_dev: TokenizedWindows,
    *,
    initialization: Literal["base", "lapt"],
    config: TransferConfig = TransferConfig(),
    device: str | torch.device | None = None,
    evaluate_fn: Callable[[Any, TokenizedWindows, torch.device], Mapping[str, Any]] | None = None,
) -> TransferResult:
    """Train exactly the configured bounded updates and select only on Slovak dev."""
    if initialization not in ("base", "lapt"):
        raise ValueError("initialization must be 'base' or 'lapt'")
    _assert_canonical_classifier(model)
    _validate_transfer_data(slovak_train, role="train")
    _validate_transfer_data(slovak_dev, role="dev")
    set_seed(config.seed)
    target_device = torch.device(device) if device is not None else get_device()
    model.to(target_device)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        total_steps=config.updates,
    )
    optimizer.zero_grad(set_to_none=True)
    label_counts = _modeling._training_label_counts(slovak_train)
    groups = _shuffled_groups(
        slovak_train.num_sentences,
        config.effective_batch_size,
        random.Random(config.seed),
    )
    history: list[TransferEvaluation] = []
    best_state: dict[str, Any] | None = None
    best_metrics: Mapping[str, Any] | None = None
    best_rank: tuple[float, float, float] | None = None
    best_step = 0
    interval_loss = torch.zeros((), device=target_device)
    interval_updates = 0

    for step in range(1, config.updates + 1):
        model.train()
        sentence_indices = next(groups)
        group_sentence_count = len(sentence_indices)
        window_indices = tuple(
            window_index
            for sentence_index in sentence_indices
            for window_index in slovak_train.windows_by_sentence[sentence_index]
        )
        update_loss = torch.zeros((), device=target_device)
        for offset in range(0, len(window_indices), config.batch_size):
            batch = window_indices[offset : offset + config.batch_size]
            inputs, labels, sentence_ids = _modeling._batch_tensors(
                slovak_train, batch, target_device, include_labels=True
            )
            assert labels is not None
            logits = _modeling._model_logits(model, inputs)
            loss = _modeling._scaled_sentence_loss(
                logits,
                labels,
                sentence_ids,
                label_counts,
                group_sentence_count,
            )
            loss.backward()
            update_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        interval_loss += update_loss
        interval_updates += 1

        if step % config.eval_every == 0 or step == config.updates:
            if evaluate_fn is None:
                metrics = _modeling.evaluate_token_classifier(
                    model,
                    slovak_dev,
                    batch_size=config.batch_size,
                    device=target_device,
                    labels=UPOS_TAGS,
                )
            else:
                metrics = dict(evaluate_fn(model, slovak_dev, target_device))
            rank = selection_rank(metrics)
            improved = best_rank is None or rank > best_rank
            if improved:
                best_rank = rank
                best_metrics = copy.deepcopy(metrics)
                best_state = _modeling._snapshot_state(model)
                best_step = step
            history.append(
                TransferEvaluation(
                    step=step,
                    train_loss=float((interval_loss / interval_updates).cpu()),
                    dev_metrics=copy.deepcopy(metrics),
                    selection_rank=rank,
                    is_best=improved,
                )
            )
            interval_loss = torch.zeros((), device=target_device)
            interval_updates = 0

    if best_state is None or best_metrics is None:
        raise AdaptationError("Slovak transfer completed without a dev checkpoint")
    model.load_state_dict(best_state)
    model.to(target_device)
    model.zero_grad(set_to_none=True)
    # The optimizer is intentionally not returned. Later Czech refinement through
    # train_token_classifier necessarily constructs a fresh optimizer/scheduler.
    _assert_canonical_classifier(model)
    return TransferResult(
        model=model,
        initialization=initialization,
        best_step=best_step,
        optimizer_steps=config.updates,
        best_metrics=best_metrics,
        history=tuple(history),
    )


def model_for_czech_refinement(result: TransferResult) -> Any:
    """Return the transferred model after enforcing a fresh-optimizer boundary."""
    if type(result) is not TransferResult or result.optimizer_state_preserved is not False:
        raise TypeError("Czech refinement requires a completed TransferResult")
    _assert_canonical_classifier(result.model)
    result.model.zero_grad(set_to_none=True)
    return result.model
