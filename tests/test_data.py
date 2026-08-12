from __future__ import annotations

import json
from dataclasses import replace

import pytest

from conftest import labeled_sentence, write_conllu
from sanna_tagging.config import UPOS_TAGS
from sanna_tagging.data import (
    ConlluFormatError,
    DataLeakageError,
    LabeledToken,
    Token,
    _deduplicate,
    assert_disjoint_form_hashes,
    deterministic_partitions,
    nested_gold_samples,
    ordered_form_hash,
    parse_labeled_conllu,
    parse_unlabeled_conllu,
    prepare_data,
)


def test_parser_preserves_provenance_and_skips_non_syntactic_rows(tmp_path):
    source = tmp_path / "sample.conllu"
    source.write_text(
        "# sent_id = x-1\n"
        "# text = Do domu.\n"
        "1-2\tDo\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "1\tDo\t_\tADP\t_\t_\t2\tcase\t_\t_\n"
        "2\tdomu\t_\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "2.1\telided\t_\tVERB\t_\t_\t_\t_\t_\t_\n"
        "3\t.\t_\tPUNCT\t_\t_\t2\tpunct\t_\t_\n\n",
        encoding="utf-8",
    )

    (sentence,) = parse_labeled_conllu(source, split="train", source_file="pinned/file")

    assert sentence.forms == ("Do", "domu", ".")
    assert tuple(token.upos for token in sentence.tokens) == ("ADP", "NOUN", "PUNCT")
    assert sentence.provenance.source_file == "pinned/file"
    assert sentence.provenance.sent_id == "x-1"
    assert [token.provenance.line_number for token in sentence.tokens] == [4, 5, 7]
    assert all(isinstance(token, LabeledToken) for token in sentence.tokens)


def test_unlabeled_parser_never_retains_upos_and_labeled_test_is_forbidden(tmp_path):
    source = write_conllu(tmp_path / "test.conllu", [("tajné", "VERB")])

    (sentence,) = parse_unlabeled_conllu(source, split="test")

    assert type(sentence.tokens[0]) is Token
    assert not hasattr(sentence.tokens[0], "upos")
    with pytest.raises(DataLeakageError, match="may not parse test UPOS"):
        parse_labeled_conllu(source, split="test")


def test_parser_rejects_noncanonical_upos(tmp_path):
    source = write_conllu(tmp_path / "bad.conllu", [("word", "NOT_A_UPOS")])
    with pytest.raises(ConlluFormatError, match="invalid gold UPOS"):
        parse_labeled_conllu(source, split="train")
    assert len(UPOS_TAGS) == 17
    assert UPOS_TAGS == tuple(sorted(UPOS_TAGS))


def test_sampling_and_partitioning_are_deterministic_and_label_independent():
    original = tuple(
        labeled_sentence((f"form-{index}",), tags=("NOUN",), sentence_index=index)
        for index in range(12)
    )
    changed_labels = tuple(
        replace(
            sentence,
            tokens=(replace(sentence.tokens[0], upos="VERB"),),
        )
        for sentence in original
    )

    first = nested_gold_samples(original, (3, 7, 12), seed=31)
    second = nested_gold_samples(changed_labels, (3, 7, 12), seed=31)
    assert {size: tuple(item.form_hash for item in rows) for size, rows in first.items()} == {
        size: tuple(item.form_hash for item in rows) for size, rows in second.items()
    }
    assert {item.form_hash for item in first[3]} <= {item.form_hash for item in first[7]}
    assert deterministic_partitions(original, {"a": 0.5, "b": 0.25, "c": 0.25}, seed=9) == (
        deterministic_partitions(original, {"a": 0.5, "b": 0.25, "c": 0.25}, seed=9)
    )


def test_deduplication_keeps_first_and_leakage_guard_uses_ordered_forms():
    first = labeled_sentence(("same", "order"), sentence_index=0)
    duplicate = labeled_sentence(("same", "order"), sentence_index=1)
    reversed_sentence = labeled_sentence(("order", "same"), sentence_index=2)

    assert ordered_form_hash(first.forms) == ordered_form_hash(duplicate.forms)
    assert ordered_form_hash(first.forms) != ordered_form_hash(reversed_sentence.forms)
    assert _deduplicate((first, duplicate, reversed_sentence)) == (first, reversed_sentence)
    with pytest.raises(DataLeakageError, match="overlap"):
        assert_disjoint_form_hashes({"train": (first,), "dev": (duplicate,)})


def test_prepare_deduplicates_by_test_then_dev_and_serializes_test_unlabeled(
    tmp_path, v2_config
):
    config = v2_config
    config["seed_sizes"] = [1, 2]
    config["random_seeds"] = [0]
    config["data"]["dev_cap"] = 2
    config["data"]["dev_partitions"] = {"selection": 0.5, "calibration_fit": 0.5}

    czech_train = write_conllu(
        tmp_path / "raw" / "cs-train.conllu",
        [("train-only", "NOUN"), ("dev-copy", "VERB"), ("test-copy", "ADJ"), ("train-2", "NOUN")],
    )
    czech_dev = write_conllu(
        tmp_path / "raw" / "cs-dev.conllu",
        [("dev-copy", "VERB"), ("dev-only", "NOUN"), ("test-copy", "ADJ")],
    )
    czech_test = write_conllu(
        tmp_path / "raw" / "cs-test.conllu", [("test-copy", "ADJ")]
    )
    slovak_train = write_conllu(
        tmp_path / "raw" / "sk-train.conllu", [("sk-train", "NOUN")]
    )
    slovak_dev = write_conllu(
        tmp_path / "raw" / "sk-dev.conllu", [("sk-dev", "NOUN")]
    )
    raw = {
        "czech": {"train": (czech_train,), "dev": (czech_dev,), "test": (czech_test,)},
        "slovak": {"train": (slovak_train,), "dev": (slovak_dev,)},
    }

    manifest = prepare_data(config, raw, tmp_path / "prepared", run_fingerprint="f" * 20)

    assert manifest["test_labels_exposed"] is False
    assert manifest["splits"]["czech/train"]["sentences"] == 2
    assert manifest["splits"]["czech/dev_clean"]["sentences"] == 2
    test_record = json.loads(
        (tmp_path / "prepared/czech/test.jsonl").read_text(encoding="utf-8").strip()
    )
    assert "upos" not in test_record["tokens"][0]
    lapt_records = [
        json.loads(line)
        for line in (tmp_path / "prepared/czech/lapt_train_forms.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert lapt_records
    assert all("upos" not in token for row in lapt_records for token in row["tokens"])
    assert manifest["splits"]["czech/lapt_train_forms"]["labels_included"] is False
    assert (
        manifest["splits"]["czech/lapt_train_forms"]["ordered_form_hashes"]
        == manifest["splits"]["czech/train"]["ordered_form_hashes"]
    )
    audit = json.loads((tmp_path / "prepared/data_audit.json").read_text(encoding="utf-8"))
    assert audit["removed"]["czech/train"] == 2
    assert audit["removed"]["czech/dev"] == 1
