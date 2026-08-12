from __future__ import annotations

import inspect

import pytest

from sanna_tagging.artifacts import (
    ArtifactIntegrityError,
    ResumeMismatchError,
    atomic_output_path,
    sha256_file,
    strict_resume,
    write_manifest,
)
from sanna_tagging.config import compute_run_fingerprint
from sanna_tagging.data import DownloadIntegrityError, download_pinned
from sanna_tagging.reporting import (
    FinalizationAlreadyAttempted,
    FinalizationMismatch,
    create_draft_report,
    finalize_official_test,
)
from sanna_tagging.runner import verify_checkpoint_manifest, write_checkpoint_manifest


def test_fingerprint_binds_config_commit_and_dependency_lock(tmp_path, v2_config):
    lock = tmp_path / "requirements.lock"
    lock.write_text("dependency==1\n", encoding="utf-8")
    config_a = dict(v2_config)
    config_b = dict(v2_config)
    config_b["_config_path"] = "/a/different/machine/config.yaml"

    first = compute_run_fingerprint(config_a, code_commit="a" * 40, lock_file=lock)
    same = compute_run_fingerprint(config_b, code_commit="a" * 40, lock_file=lock)
    new_commit = compute_run_fingerprint(config_a, code_commit="b" * 40, lock_file=lock)
    lock.write_text("dependency==2\n", encoding="utf-8")
    new_lock = compute_run_fingerprint(config_a, code_commit="a" * 40, lock_file=lock)

    assert first == same
    assert len(first) == 20
    assert len({first, new_commit, new_lock}) == 3


def test_valid_pinned_cache_is_reused_offline_and_corruption_fails_closed(
    tmp_path, monkeypatch
):
    destination = tmp_path / "cached.bin"
    destination.write_bytes(b"immutable")
    digest = sha256_file(destination)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("valid cache attempted network access"),
    )

    assert download_pinned("https://invalid.example/file", destination, digest) == destination
    destination.write_bytes(b"changed")
    with pytest.raises(DownloadIntegrityError, match="cached SHA-256 mismatch"):
        download_pinned("https://invalid.example/file", destination, digest)


def test_atomic_output_never_publishes_partial_bytes(tmp_path):
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")

    with pytest.raises(RuntimeError, match="interrupt"):
        with atomic_output_path(destination) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("interrupt")

    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob(".artifact.bin.*.tmp")) == []


def test_strict_resume_accepts_exact_state_and_rejects_tamper_or_partial_output(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    input_path.write_text("input", encoding="utf-8")
    output_path.write_text("output", encoding="utf-8")
    arguments = {
        "stage": "unit",
        "run_fingerprint": "f" * 20,
        "inputs": {"input": input_path},
        "outputs": {"output": output_path},
        "parameters": {"value": 1},
        "relative_to": tmp_path,
    }
    write_manifest(manifest_path, **arguments)

    assert strict_resume(manifest_path, **arguments) is True
    output_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        strict_resume(manifest_path, **arguments)

    partial = tmp_path / "partial"
    partial.write_text("orphan", encoding="utf-8")
    with pytest.raises(ResumeMismatchError, match="without commit manifest"):
        strict_resume(
            tmp_path / "missing-manifest.json",
            stage="unit",
            run_fingerprint="f" * 20,
            inputs={},
            outputs={"partial": partial},
            relative_to=tmp_path,
        )


def test_checkpoint_manifest_rejects_unrecorded_files(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"model")
    manifest = tmp_path / "checkpoint.manifest.json"
    write_checkpoint_manifest(
        manifest,
        checkpoint_dir=checkpoint,
        run_fingerprint="f" * 20,
        metadata={"load_subdir": "."},
    )

    verify_checkpoint_manifest(manifest, expected_fingerprint="f" * 20)
    (checkpoint / "injected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="file set differs"):
        verify_checkpoint_manifest(manifest, expected_fingerprint="f" * 20)


def test_draft_api_has_no_test_gold_capability():
    parameters = inspect.signature(create_draft_report).parameters

    assert not any("test" in name for name in parameters)
    assert "_parse_official_test_gold" not in create_draft_report.__code__.co_names
    assert "official" not in " ".join(create_draft_report.__code__.co_names)


def test_finalization_requires_exact_report_lock_before_gold_parser(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    wrong_lock = tmp_path / "selection.lock.json"
    wrong_lock.write_text("{}", encoding="utf-8")
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("{}\n", encoding="utf-8")
    official = tmp_path / "official.conllu"
    official.write_text("not opened", encoding="utf-8")
    monkeypatch.setattr(
        "sanna_tagging.reporting._parse_official_test_gold",
        lambda *_args: pytest.fail("gold parser reached without authorization"),
    )

    with pytest.raises(FinalizationMismatch, match="exact lock"):
        finalize_official_test(
            run_dir=run_dir,
            run_fingerprint="f" * 20,
            selection_lock_path=wrong_lock,
            prepared_unlabeled_test_path=prepared,
            official_gold_test_path=official,
            expected_gold_sha256="0" * 64,
            output_dir=run_dir / "final",
            evaluator=lambda *_args: {},
        )


def test_existing_finalization_claim_rejects_before_any_input_access(tmp_path):
    output = tmp_path / "final"
    output.mkdir()
    with pytest.raises(FinalizationAlreadyAttempted, match="already attempted"):
        finalize_official_test(
            run_dir=tmp_path,
            run_fingerprint="f" * 20,
            selection_lock_path=tmp_path / "missing-lock",
            prepared_unlabeled_test_path=tmp_path / "missing-prepared",
            official_gold_test_path=tmp_path / "missing-gold",
            expected_gold_sha256="0" * 64,
            output_dir=output,
            evaluator=lambda *_args: {},
        )
