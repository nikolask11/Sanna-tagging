from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_conllu
from sanna_tagging.artifacts import (
    ArtifactIntegrityError,
    ResumeMismatchError,
    load_manifest,
    sha256_file,
)
from sanna_tagging.cli import main
from sanna_tagging.reporting import finalize_official_test
from sanna_tagging.runner import (
    CalibrationExecution,
    CheckpointExecution,
    ExperimentRunner,
    RunnerError,
    build_stage_plan,
)


class OfflineBackend:
    def __init__(self):
        self.compute_calls = 0
        self.calibration_calls = 0
        self.lapt_train_path = None

    @staticmethod
    def _metrics():
        return {
            "sentence_at_least_98_rate": 0.9,
            "token_accuracy": 0.95,
            "exact_match_rate": 0.8,
        }

    def _checkpoint(self, output_dir: Path, *, metadata=None, metrics=None):
        self.compute_calls += 1
        output_dir.mkdir(parents=True)
        (output_dir / "model.bin").write_bytes(f"offline-{self.compute_calls}".encode())
        return CheckpointExecution(
            checkpoint_dir=output_dir,
            metrics=metrics or self._metrics(),
            metadata={"load_subdir": ".", **(metadata or {})},
        )

    def train_supervised(self, *, output_dir, **_kwargs):
        return self._checkpoint(output_dir)

    def train_lapt(self, *, output_dir, train_path, **_kwargs):
        self.lapt_train_path = Path(train_path)
        records = [json.loads(line) for line in self.lapt_train_path.read_text().splitlines()]
        assert records and all("upos" not in token for row in records for token in row["tokens"])
        return self._checkpoint(
            output_dir,
            metadata={"labels_accessed": False},
            metrics={"best_holdout_loss": 1.0, "best_step": 1},
        )

    def train_transfer(self, *, output_dir, job, **_kwargs):
        return self._checkpoint(
            output_dir,
            metadata={
                "initialization": job["initialization"],
                "optimizer_state_preserved": False,
            },
        )

    def calibrate_ensemble(self, **_kwargs):
        self.calibration_calls += 1
        return CalibrationExecution(
            calibration={
                "fit_partition": "calibration_fit",
                "assessment_partition": "calibration_assessment",
                "temperature": 1.0,
            },
            ensemble={
                "method": "unweighted-arithmetic-mean-probabilities",
                "member_count": 3,
                "selection_metrics": self._metrics(),
            },
        )

    def benchmark_ensemble(self, **_kwargs):
        return {"selection_influence": False, "sentences_per_second": 1.0}

    def evaluate_official_test(self, **_kwargs):
        raise AssertionError("offline Task #5 tests must never evaluate official test labels")


class XlmrWinningBackend(OfflineBackend):
    def train_supervised(self, *, output_dir, job, **_kwargs):
        metrics = self._metrics()
        if job["model_family"] == "xlmr":
            metrics = {
                "sentence_at_least_98_rate": 0.99,
                "token_accuracy": 0.99,
                "exact_match_rate": 0.99,
            }
        return self._checkpoint(output_dir, metrics=metrics)


def offline_raw_files(tmp_path):
    raw = tmp_path / "offline-raw"
    czech_train = write_conllu(
        raw / "cs-train.conllu",
        [(f"train-{index}", "NOUN") for index in range(805)],
    )
    czech_dev = write_conllu(
        raw / "cs-dev.conllu",
        [(f"dev-{index}", "NOUN") for index in range(12)],
    )
    czech_test = write_conllu(
        raw / "cs-test.conllu",
        [(f"test-{index}", "NOUN") for index in range(3)],
    )
    slovak_train = write_conllu(
        raw / "sk-train.conllu",
        [(f"sk-train-{index}", "NOUN") for index in range(4)],
    )
    slovak_dev = write_conllu(
        raw / "sk-dev.conllu",
        [(f"sk-dev-{index}", "NOUN") for index in range(3)],
    )
    return {
        "czech": {"train": (czech_train,), "dev": (czech_dev,), "test": (czech_test,)},
        "slovak": {"train": (slovak_train,), "dev": (slovak_dev,)},
    }


def test_stage_plan_contains_exact_30_job_protocol(v2_config):
    plan = build_stage_plan(v2_config)

    assert plan["counts"] == {
        "budget_supervised": 15,
        "lapt": 1,
        "slovak_transfer": 2,
        "xlmr_reference_supervised": 3,
        "adapted_supervised": 9,
        "supervised_total": 29,
        "compute_total": 30,
    }
    assert len(plan["jobs"]) == 30
    assert len({job["id"] for job in plan["jobs"]}) == 30
    draft = next(stage for stage in plan["stages"] if stage["command"] == "report-draft")
    assert draft["protected_test_labels"] == "not accepted by this stage"


def test_cli_dry_run_prints_30_job_plan_without_writes(tmp_path, capsys):
    runs_root = tmp_path / "runs"

    result = main(
        [
            "--runs-root",
            str(runs_root),
            "--dry-run",
            "plan",
        ]
    )

    assert result == 0
    assert not runs_root.exists()
    line = capsys.readouterr().out.strip()
    assert line.startswith("SANNA_RESULT ")
    record = json.loads(line.removeprefix("SANNA_RESULT "))
    assert record["record_type"] == "plan"
    assert record["plan"]["counts"]["compute_total"] == 30
    assert len(record["plan"]["jobs"]) == 30


def test_prepare_rejects_raw_cache_inside_portable_run(tmp_path):
    runner = ExperimentRunner(
        runs_root=tmp_path / "runs", tiny=True, code_commit="r" * 40
    )
    with pytest.raises(RunnerError, match="outside the portable run"):
        runner.prepare(raw_root=runner.layout.run_dir / "raw")


def test_non_direct_winner_prunes_selected_direct_checkpoints_and_resumes(
    tmp_path, monkeypatch
):
    raw_files = offline_raw_files(tmp_path)
    monkeypatch.setattr(
        "sanna_tagging.data.download_corpora",
        lambda _config, _raw_dir: raw_files,
    )
    backend = XlmrWinningBackend()
    runner = ExperimentRunner(
        runs_root=tmp_path / "runs",
        backend=backend,
        tiny=True,
        code_commit="x" * 40,
    )

    runner.prepare()
    runner.budget()
    adaptation = runner.adapt()

    assert adaptation["selection"]["winning_arm"] == "xlmr-reference"
    model_paths = list(runner.layout.run_dir.rglob("model.bin"))
    assert len(model_paths) == 3
    assert all(
        path.is_relative_to(runner.layout.adapt_dir / "reference")
        for path in model_paths
    )
    assert not list(runner.layout.budget_dir.rglob("model.bin"))
    calls_after_selection = backend.compute_calls

    # Simulate interruption after authorized deletion but before the final
    # pruning commit marker; the durable plan must permit idempotent recovery.
    budget_pruning_commit = runner.layout.budget_dir / "pruning.manifest.json"
    budget_pruning_commit.unlink()
    runner.budget()
    assert budget_pruning_commit.is_file()

    runner.budget()
    runner.adapt()

    assert backend.compute_calls == calls_after_selection == 30
    assert len(list(runner.layout.run_dir.rglob("model.bin"))) == 3


def test_mocked_offline_e2e_freezes_selection_lock_and_resumes_atomically(
    tmp_path, monkeypatch
):
    raw_files = offline_raw_files(tmp_path)
    download_roots = []

    def fake_download(_config, raw_dir):
        download_roots.append(Path(raw_dir).resolve())
        return raw_files

    monkeypatch.setattr("sanna_tagging.data.download_corpora", fake_download)
    backend = OfflineBackend()
    runner = ExperimentRunner(
        runs_root=tmp_path / "runs",
        backend=backend,
        tiny=True,
        code_commit="c" * 40,
    )

    prepared = runner.prepare()
    assert download_roots == [runner.layout.raw_dir.resolve()]
    assert not runner.layout.raw_dir.is_relative_to(runner.layout.run_dir)
    assert not runner.layout.raw_dir.is_relative_to(runner.layout.runs_root)
    assert not list(runner.layout.run_dir.rglob("*.conllu"))
    budget = runner.budget()
    assert len(list(runner.layout.run_dir.rglob("model.bin"))) == 3
    assert (runner.layout.budget_dir / "pruning.manifest.json").is_file()
    adaptation = runner.adapt()
    assert len(list(runner.layout.run_dir.rglob("model.bin"))) == 3
    assert (runner.layout.adapt_dir / "pruning.manifest.json").is_file()
    draft = runner.report_draft()

    assert prepared["test_labels_exposed"] is False
    assert budget["selected_budget"] == 50
    assert adaptation["selection"]["winning_arm"] == "slovakbert-direct"
    assert backend.lapt_train_path == (
        runner.layout.prepared_dir / "czech" / "lapt_train_forms.jsonl"
    )
    assert draft["status"] == "draft-selection-frozen-test-not-evaluated"
    assert draft["test_labels_accessed"] is False
    assert backend.compute_calls == 30
    assert backend.calibration_calls == 1

    lock_path = runner.layout.report_dir / "selection.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["selection_frozen"] is True
    assert lock["final_test_evaluated"] is False
    assert lock["data"]["test_labels_exposed"] is False
    assert len(lock["checkpoints"]) == 3
    assert len(set(lock["checkpoint_hashes"])) == 3
    assert all(not Path(entry["checkpoint_manifest"]["path"]).is_absolute() for entry in lock["checkpoints"])
    assert not runner.layout.final_dir.exists()

    runner.prepare()
    runner.budget()
    runner.adapt()
    resumed_draft = runner.report_draft()
    assert resumed_draft == draft
    assert backend.compute_calls == 30
    assert backend.calibration_calls == 1

    lock_bytes = lock_path.read_bytes()
    observed_seed_paths = None

    def synthetic_evaluator(checkpoint_by_seed, _official_test, _calibration):
        nonlocal observed_seed_paths
        observed_seed_paths = dict(checkpoint_by_seed)
        return {
            "seed_metrics": [
                {
                    "seed": seed,
                    "tagging": {
                        "token_accuracy": 0.90 + seed / 100,
                        "sentence_at_least_98_rate": 0.80 + seed / 100,
                        "exact_match_rate": 0.70 + seed / 100,
                    },
                }
                for seed in checkpoint_by_seed
            ],
            "ensemble": {
                "tagging": {
                    "token_accuracy": 0.95,
                    "sentence_at_least_98_rate": 0.90,
                    "exact_match_rate": 0.85,
                },
                "calibration": {"synthetic": True},
            },
        }

    synthetic_gold = raw_files["czech"]["test"][0]
    synthetic_final_dir = runner.layout.run_dir / "synthetic-final"
    final_report = finalize_official_test(
        run_dir=runner.layout.run_dir,
        run_fingerprint=runner.fingerprint,
        selection_lock_path=lock_path,
        prepared_unlabeled_test_path=runner.layout.prepared_dir / "czech" / "test.jsonl",
        official_gold_test_path=synthetic_gold,
        expected_gold_sha256=sha256_file(synthetic_gold),
        output_dir=synthetic_final_dir,
        evaluator=synthetic_evaluator,
    )
    assert list(observed_seed_paths) == [0, 1, 2]
    assert final_report["selection"]["winning_arm"] == lock["winning_arm"]
    assert lock_path.read_bytes() == lock_bytes
    assert (synthetic_final_dir / "test_metrics.json").is_file()
    final_markdown = (synthetic_final_dir / "REPORT_FINAL.md").read_text(encoding="utf-8")
    assert all(name in final_markdown for name in ("Seed 0", "Seed 1", "Seed 2", "ensemble"))
    assert "not changed after selection.lock.json was frozen" in final_markdown
    final_manifest = load_manifest(synthetic_final_dir / "artifact-manifest.json")
    assert {"test_metrics", "final_report_markdown"} <= set(final_manifest["outputs"])

    selected_job_manifest = (
        runner.layout.budget_dir / "jobs" / "n50-seed0" / "artifact-manifest.json"
    )
    selected_job_manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises((ArtifactIntegrityError, ResumeMismatchError)):
        runner.adapt()
    assert backend.compute_calls == 30
