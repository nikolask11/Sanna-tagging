from __future__ import annotations

from pathlib import Path

import pytest

from sanna_tagging.config import DEFAULT_CONFIG, load_config
from sanna_tagging.data import (
    LabeledToken,
    Sentence,
    SentenceProvenance,
    Token,
    TokenProvenance,
)


@pytest.fixture
def v2_config() -> dict:
    return load_config(DEFAULT_CONFIG)


def labeled_sentence(
    forms: tuple[str, ...],
    *,
    tags: tuple[str, ...] | None = None,
    source_file: str = "czech/train/synthetic.conllu",
    split: str = "train",
    sentence_index: int = 0,
) -> Sentence:
    if tags is None:
        tags = tuple("NOUN" for _ in forms)
    return Sentence(
        tokens=tuple(
            LabeledToken(
                form=form,
                upos=tag,
                provenance=TokenProvenance(conllu_id=index, line_number=index + 1),
            )
            for index, (form, tag) in enumerate(zip(forms, tags, strict=True), start=1)
        ),
        provenance=SentenceProvenance(
            source_file=source_file,
            split=split,
            sentence_index=sentence_index,
            sent_id=f"s{sentence_index}",
        ),
        comments=(f"# sent_id = s{sentence_index}",),
    )


def unlabeled_sentence(
    forms: tuple[str, ...],
    *,
    source_file: str = "czech/train/synthetic.conllu",
    split: str = "train",
    sentence_index: int = 0,
) -> Sentence:
    return Sentence(
        tokens=tuple(
            Token(
                form=form,
                provenance=TokenProvenance(conllu_id=index, line_number=index + 1),
            )
            for index, form in enumerate(forms, start=1)
        ),
        provenance=SentenceProvenance(
            source_file=source_file,
            split=split,
            sentence_index=sentence_index,
            sent_id=f"s{sentence_index}",
        ),
        comments=(f"# sent_id = s{sentence_index}",),
    )


def write_conllu(path: Path, rows: list[tuple[str, str]]) -> Path:
    blocks = []
    for sentence_index, (form, tag) in enumerate(rows):
        blocks.append(
            "\n".join(
                (
                    f"# sent_id = s{sentence_index}",
                    f"1\t{form}\t_\t{tag}\t_\t_\t0\troot\t_\t_",
                )
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n\n", encoding="utf-8")
    return path
