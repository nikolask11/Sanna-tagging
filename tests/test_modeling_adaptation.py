from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from conftest import labeled_sentence, unlabeled_sentence
from sanna_tagging.adaptation import (
    AdaptationError,
    LAPTTrainingText,
    TransferResult,
    model_for_czech_refinement,
    project_czech_training_forms,
    split_lapt_holdout,
)
from sanna_tagging.config import UPOS_TAGS
from sanna_tagging.modeling import (
    TrainingConfig,
    canonical_label_maps,
    sentence_balanced_loss,
    train_token_classifier,
)
from sanna_tagging.runner import V2ExecutionBackend
from sanna_tagging.tokenization import TokenizedWindows, WordOccurrence


def tiny_tokenized() -> TokenizedWindows:
    return TokenizedWindows(
        features=({"input_ids": (1, 1)},),
        labels=((0, 0),),
        sentence_ids=("s",),
        word_counts=(2,),
        window_sentence_indices=(0,),
        word_ids=((0, 1),),
        occurrences=(
            WordOccurrence(0, 0, 0, "s", 0),
            WordOccurrence(0, 1, 0, "s", 1),
        ),
        gold_label_ids=((0, 0),),
    )


class ScalarClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids):
        positive = self.weight * input_ids.float()
        return {"logits": torch.stack((positive, -positive), dim=-1)}


def test_sentence_balanced_loss_weights_sentences_not_tokens():
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 0.0]],
            [[3.0, 0.0], [0.0, 0.0]],
            [[0.0, 2.0], [2.0, 0.0]],
        ]
    )
    labels = torch.tensor([[0, -100], [1, -100], [0, 0]])
    actual = sentence_balanced_loss(logits, labels, [10, 10, 20])

    sentence_10 = torch.stack(
        (F.cross_entropy(logits[0, 0][None], torch.tensor([0])),
         F.cross_entropy(logits[1, 0][None], torch.tensor([1])))
    ).mean()
    sentence_20 = torch.stack(
        (F.cross_entropy(logits[2, 0][None], torch.tensor([0])),
         F.cross_entropy(logits[2, 1][None], torch.tensor([0])))
    ).mean()
    assert actual == pytest.approx(float((sentence_10 + sentence_20) / 2))


def test_training_restores_best_dev_checkpoint_not_last_epoch():
    model = ScalarClassifier()
    observed_weights = []
    scores = iter((1.0, 0.8, 0.7))

    def evaluate(candidate, _data, _device):
        observed_weights.append(float(candidate.weight.detach()))
        score = next(scores)
        return {
            "sentence_at_least_98_rate": score,
            "token_accuracy": score,
            "exact_match_rate": score,
        }

    result = train_token_classifier(
        model,
        tiny_tokenized(),
        tiny_tokenized(),
        config=TrainingConfig(
            batch_size=1,
            effective_batch_size=1,
            learning_rate=0.1,
            weight_decay=0.0,
            warmup_ratio=0.0,
            min_epochs=1,
            max_epochs=3,
            patience=3,
            gradient_clip=1.0,
        ),
        seed=4,
        device="cpu",
        labels=("A", "B"),
        evaluate_fn=evaluate,
    )

    assert result.best_epoch == 1
    assert len(observed_weights) == 3
    assert float(result.model.weight.detach()) == pytest.approx(observed_weights[0])
    assert float(result.model.weight.detach()) != pytest.approx(observed_weights[-1])


def test_lapt_projection_is_forms_only_czech_train_and_internal_holdout():
    sentences = tuple(
        unlabeled_sentence((f"form-{index}",), sentence_index=index)
        for index in range(5)
    )
    projected = project_czech_training_forms(sentences)
    train_a, holdout_a = split_lapt_holdout(projected, holdout_sentences=2, seed=19)
    train_b, holdout_b = split_lapt_holdout(projected, holdout_sentences=2, seed=19)

    assert (train_a, holdout_a) == (train_b, holdout_b)
    assert len(train_a) == 3 and len(holdout_a) == 2
    assert not hasattr(projected, "labels")
    with pytest.raises(TypeError, match="created by project_czech_training_forms"):
        split_lapt_holdout(LAPTTrainingText((), object()), holdout_sentences=1, seed=0)
    with pytest.raises(AdaptationError, match="training split only"):
        project_czech_training_forms(
            (unlabeled_sentence(("dev",), split="dev", source_file="czech/dev/x", sentence_index=0),)
        )
    with pytest.raises(AdaptationError, match="labels are forbidden"):
        project_czech_training_forms((labeled_sentence(("labeled",), sentence_index=0),))


def test_transfer_result_enforces_fresh_optimizer_boundary_and_clears_gradients():
    label2id, id2label = canonical_label_maps(UPOS_TAGS)
    model = torch.nn.Linear(2, len(UPOS_TAGS))
    model.config = SimpleNamespace(
        label2id=label2id,
        id2label=id2label,
        num_labels=len(UPOS_TAGS),
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    result = TransferResult(
        model=model,
        initialization="base",
        best_step=1,
        optimizer_steps=1,
        best_metrics={},
        history=(),
    )

    returned = model_for_czech_refinement(result)

    assert returned is model
    assert result.optimizer_state_preserved is False
    assert all(parameter.grad is None for parameter in model.parameters())


def test_backend_seeds_before_random_classifier_initialization(monkeypatch, tmp_path, v2_config):
    backend = V2ExecutionBackend()
    monkeypatch.setattr(backend, "_sentences", lambda *_args, **_kwargs: (object(),))
    monkeypatch.setattr(
        "sanna_tagging.tokenization.tokenize_sentences", lambda *_args, **_kwargs: object()
    )
    draws = []

    def fake_model_tokenizer(**_kwargs):
        draws.append(float(torch.rand(())))
        return object(), object(), {}

    monkeypatch.setattr(backend, "_model_tokenizer", fake_model_tokenizer)
    monkeypatch.setattr(backend, "_save_model", lambda *_args, **_kwargs: None)
    fake_training = SimpleNamespace(
        model=object(),
        best_metrics={"token_accuracy": 1.0},
        best_epoch=1,
        best_step=1,
        optimizer_steps=1,
        stopped_early=False,
        history=(),
    )
    monkeypatch.setattr(
        "sanna_tagging.modeling.train_token_classifier",
        lambda *_args, **_kwargs: fake_training,
    )
    arguments = {
        "job": {"seed": 73, "model_family": "slovakbert"},
        "train_path": tmp_path / "train",
        "dev_path": tmp_path / "dev",
        "initialization_manifest": None,
        "output_dir": tmp_path / "checkpoint",
        "config": v2_config,
        "tiny": True,
    }

    backend.train_supervised(**arguments)
    backend.train_supervised(**arguments)

    assert draws[0] == draws[1]
