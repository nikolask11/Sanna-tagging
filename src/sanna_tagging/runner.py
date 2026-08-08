"""Manifest-driven orchestration for the Czech v2 experiment DAG."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, TextIO

from .artifacts import (
    ArtifactIntegrityError,
    ResumeMismatchError,
    atomic_write_bytes,
    atomic_write_json,
    file_record,
    load_manifest,
    sha256_file,
    strict_resume,
    verify_file_record,
    verify_manifest_files,
    write_manifest,
)
from .config import (
    DEFAULT_CONFIG,
    LOCK_FILE,
    REPO_ROOT,
    UPOS_TAGS,
    canonical_config,
    compute_run_fingerprint,
    expand_experiment_plan,
    load_config,
)
from .selection import (
    SelectionCandidate,
    aggregate_candidate,
    select_budget,
    select_candidate,
)

RUNNER_SCHEMA_VERSION = 1
ADAPTED_VARIANTS = (
    "slovakbert-czech-lapt",
    "slovakbert-slovak-upos",
    "slovakbert-czech-lapt-slovak-upos",
)
WINNING_ARM_NAMES = (
    "slovakbert-direct",
    "xlmr-reference",
    *ADAPTED_VARIANTS,
)
ARM_COMPLEXITY = {
    "slovakbert-direct": 0,
    "xlmr-reference": 0,
    "slovakbert-czech-lapt": 1,
    "slovakbert-slovak-upos": 1,
    "slovakbert-czech-lapt-slovak-upos": 2,
}


class RunnerError(RuntimeError):
    """The requested stage cannot safely run from the available artifacts."""


class StageDependencyError(RunnerError):
    """A required upstream stage is absent or does not match this run."""


@dataclass(frozen=True, slots=True)
class CheckpointExecution:
    """Backend result whose checkpoint directory is sealed by the runner."""

    checkpoint_dir: Path
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CalibrationExecution:
    calibration: Mapping[str, Any]
    ensemble: Mapping[str, Any]


class ExecutionBackend(Protocol):
    """Dependency-injection boundary for GPU work and synthetic smoke backends."""

    def train_supervised(
        self,
        *,
        job: Mapping[str, Any],
        train_path: Path,
        dev_path: Path,
        initialization_manifest: Path | None,
        output_dir: Path,
        config: Mapping[str, Any],
        tiny: bool,
    ) -> CheckpointExecution: ...

    def train_lapt(
        self,
        *,
        train_path: Path,
        output_dir: Path,
        config: Mapping[str, Any],
        tiny: bool,
    ) -> CheckpointExecution: ...

    def train_transfer(
        self,
        *,
        job: Mapping[str, Any],
        slovak_train_path: Path,
        slovak_dev_path: Path,
        initialization_manifest: Path | None,
        output_dir: Path,
        config: Mapping[str, Any],
        tiny: bool,
    ) -> CheckpointExecution: ...

    def calibrate_ensemble(
        self,
        *,
        checkpoint_manifests: Sequence[Path],
        selection_path: Path,
        calibration_fit_path: Path,
        calibration_assessment_path: Path,
        config: Mapping[str, Any],
        tiny: bool,
    ) -> CalibrationExecution: ...

    def benchmark_ensemble(
        self,
        *,
        checkpoint_manifests: Sequence[Path],
        data_path: Path,
        config: Mapping[str, Any],
        tiny: bool,
    ) -> Mapping[str, Any]: ...

    def evaluate_official_test(
        self,
        *,
        checkpoint_by_seed: Mapping[int, Path],
        official_test: Any,
        calibration: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RunLayout:
    runs_root: Path
    run_name: str
    fingerprint: str

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_name / self.fingerprint

    @property
    def plan_dir(self) -> Path:
        return self.run_dir / "plan"

    @property
    def raw_dir(self) -> Path:
        """Ephemeral protected cache excluded from Kaggle's portable output."""
        configured = os.environ.get("SANNA_RAW_CACHE")
        root = Path(configured).expanduser() if configured else Path(tempfile.gettempdir())
        return root.resolve() / "sanna-tagging-raw-cache" / self.fingerprint

    @property
    def prepared_dir(self) -> Path:
        return self.run_dir / "prepared"

    @property
    def budget_dir(self) -> Path:
        return self.run_dir / "budget"

    @property
    def adapt_dir(self) -> Path:
        return self.run_dir / "adapt"

    @property
    def report_dir(self) -> Path:
        return self.run_dir / "report"

    @property
    def benchmark_dir(self) -> Path:
        return self.run_dir / "benchmark"

    @property
    def final_dir(self) -> Path:
        return self.run_dir / "final"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("run artifacts may not contain non-finite floats")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"JSON artifact must be an object: {path}")
    return value


def _portable(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_stage_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact, ordered v2 stage DAG and all 30 compute jobs."""
    expanded = expand_experiment_plan(dict(config))
    jobs = expanded["jobs"]
    counts = {
        "budget_supervised": sum(job["stage"] == "budget" for job in jobs),
        "lapt": sum(job["stage"] == "lapt" for job in jobs),
        "slovak_transfer": sum(job["stage"] == "transfer" for job in jobs),
        "xlmr_reference_supervised": sum(job["stage"] == "reference" for job in jobs),
        "adapted_supervised": sum(job["stage"] == "adapt" for job in jobs),
        "supervised_total": expanded["supervised_invocations"],
        "compute_total": len(jobs),
    }
    expected = {
        "budget_supervised": 15,
        "lapt": 1,
        "slovak_transfer": 2,
        "xlmr_reference_supervised": 3,
        "adapted_supervised": 9,
        "supervised_total": 29,
        "compute_total": 30,
    }
    if counts != expected:
        raise RunnerError(
            f"configuration does not expand to the accepted v2 job plan: {counts!r}"
        )
    if len(config["seed_sizes"]) != 5 or len(config["random_seeds"]) != 3:
        raise RunnerError("v2 requires exactly five budget sizes and three seeds")

    stage_jobs: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("budget", "lapt", "transfer", "reference", "adapt")
    }
    for job in jobs:
        normalized = dict(job)
        normalized["id"] = _job_id(normalized)
        stage_jobs[job["stage"]].append(normalized)
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "run_name": config["run_name"],
        "counts": counts,
        "stages": [
            {"command": "plan", "depends_on": [], "compute_jobs": 0},
            {"command": "prepare", "depends_on": ["plan"], "compute_jobs": 0},
            {
                "command": "budget",
                "depends_on": ["prepare"],
                "compute_jobs": counts["budget_supervised"],
                "jobs": stage_jobs["budget"],
            },
            {
                "command": "adapt",
                "depends_on": ["budget"],
                "compute_jobs": (
                    counts["lapt"]
                    + counts["slovak_transfer"]
                    + counts["xlmr_reference_supervised"]
                    + counts["adapted_supervised"]
                ),
                "jobs": (
                    stage_jobs["lapt"]
                    + stage_jobs["transfer"]
                    + stage_jobs["reference"]
                    + stage_jobs["adapt"]
                ),
            },
            {
                "command": "benchmark",
                "depends_on": ["adapt"],
                "optional": True,
                "compute_jobs": 0,
            },
            {
                "command": "report-draft",
                "depends_on": ["adapt"],
                "compute_jobs": 0,
                "protected_test_labels": "not accepted by this stage",
            },
            {
                "command": "finalize-test",
                "depends_on": ["report-draft"],
                "compute_jobs": 0,
                "explicit_one_shot": True,
            },
        ],
        "jobs": [job for stage in ("budget", "lapt", "transfer", "reference", "adapt") for job in stage_jobs[stage]],
    }


def _job_id(job: Mapping[str, Any]) -> str:
    parts = [str(job["stage"]), str(job["variant"])]
    if "size" in job:
        parts.append(f"n{job['size']}")
    if "seed" in job:
        parts.append(f"seed{job['seed']}")
    return "-".join(parts)


def _checkpoint_files(checkpoint_dir: Path, *, relative_to: Path) -> dict[str, dict[str, Any]]:
    if not checkpoint_dir.is_dir():
        raise ArtifactIntegrityError(f"checkpoint directory is missing: {checkpoint_dir}")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_symlink():
            raise ArtifactIntegrityError(f"checkpoint may not contain symlinks: {path}")
        if path.is_file():
            records[path.relative_to(checkpoint_dir).as_posix()] = file_record(
                path, relative_to=relative_to
            )
    if not records:
        raise ArtifactIntegrityError(f"checkpoint contains no files: {checkpoint_dir}")
    return records


def write_checkpoint_manifest(
    manifest_path: Path,
    *,
    checkpoint_dir: Path,
    run_fingerprint: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "run_fingerprint": run_fingerprint,
        "checkpoint_root": _portable(checkpoint_dir, manifest_path.parent),
        "metadata": _jsonable(metadata),
        "files": _checkpoint_files(checkpoint_dir, relative_to=manifest_path.parent),
    }
    atomic_write_json(manifest_path, value)
    return value


def _checkpoint_manifest_structure(
    manifest_path: str | Path, *, expected_fingerprint: str | None = None
) -> tuple[dict[str, Any], Path]:
    """Validate a checkpoint manifest without requiring retained weight files."""
    manifest_path = Path(manifest_path)
    value = _read_json(manifest_path)
    if value.get("schema_version") != 1:
        raise ArtifactIntegrityError(f"unsupported checkpoint manifest: {manifest_path}")
    if expected_fingerprint is not None and value.get("run_fingerprint") != expected_fingerprint:
        raise ResumeMismatchError(f"checkpoint fingerprint mismatch: {manifest_path}")
    checkpoint_root = value.get("checkpoint_root")
    if not isinstance(checkpoint_root, str):
        raise ArtifactIntegrityError(f"checkpoint manifest lacks a root: {manifest_path}")
    root_value = Path(checkpoint_root)
    if root_value.is_absolute() or ".." in root_value.parts:
        raise ArtifactIntegrityError(f"checkpoint root must be portable: {manifest_path}")
    root = (manifest_path.parent / root_value).resolve()
    records = value.get("files")
    if not isinstance(records, dict) or not records:
        raise ArtifactIntegrityError(f"checkpoint manifest has no files: {manifest_path}")
    for name, record in records.items():
        logical_name = Path(name) if isinstance(name, str) else None
        if (
            logical_name is None
            or logical_name.is_absolute()
            or ".." in logical_name.parts
            or not isinstance(record, dict)
        ):
            raise ArtifactIntegrityError(f"malformed checkpoint file record: {manifest_path}")
        path_value = record.get("path")
        if not isinstance(path_value, str):
            raise ArtifactIntegrityError(f"malformed checkpoint file record: {manifest_path}")
        committed = Path(path_value)
        committed = committed if committed.is_absolute() else manifest_path.parent / committed
        if committed.resolve() != (root / logical_name).resolve():
            raise ArtifactIntegrityError(
                f"checkpoint file record escapes its declared root: {manifest_path}"
            )
        if not isinstance(record.get("bytes"), int) or not isinstance(record.get("sha256"), str):
            raise ArtifactIntegrityError(f"malformed checkpoint file record: {manifest_path}")
    return value, root


def verify_checkpoint_manifest(
    manifest_path: str | Path, *, expected_fingerprint: str | None = None
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    value, root = _checkpoint_manifest_structure(
        manifest_path, expected_fingerprint=expected_fingerprint
    )
    if not root.is_dir():
        raise ArtifactIntegrityError(f"checkpoint root is missing: {root}")
    records = value["files"]
    for record in records.values():
        verify_file_record(record, relative_to=manifest_path.parent)
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactIntegrityError(f"checkpoint may not contain symlinks: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != set(records):
        raise ArtifactIntegrityError(
            f"checkpoint file set differs from its manifest: {manifest_path}"
        )
    return value


def checkpoint_load_dir(manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    value = verify_checkpoint_manifest(manifest_path)
    root = Path(value["checkpoint_root"])
    root = root if root.is_absolute() else manifest_path.parent / root
    metadata = value.get("metadata", {})
    subdir = metadata.get("load_subdir", ".") if isinstance(metadata, dict) else "."
    if not isinstance(subdir, str) or Path(subdir).is_absolute() or ".." in Path(subdir).parts:
        raise ArtifactIntegrityError(f"unsafe checkpoint load_subdir: {manifest_path}")
    destination = root / subdir
    if not destination.is_dir():
        raise ArtifactIntegrityError(f"checkpoint load directory is missing: {destination}")
    return destination


class ExperimentRunner:
    """Run resumable stages while keeping selection and final test evaluation separate."""

    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_CONFIG,
        runs_root: str | Path = REPO_ROOT / "runs",
        backend: ExecutionBackend | None = None,
        dry_run: bool = False,
        tiny: bool = False,
        code_commit: str | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.config = load_config(self.config_path)
        fingerprint_config = deepcopy(self.config)
        if tiny:
            fingerprint_config["execution"] = {"mode": "tiny-nonproduction"}
        self.fingerprint = compute_run_fingerprint(
            fingerprint_config, code_commit=code_commit
        )
        self.layout = RunLayout(
            Path(runs_root).resolve(), str(self.config["run_name"]), self.fingerprint
        )
        self.dry_run = bool(dry_run)
        self.tiny = bool(tiny)
        self._backend = backend
        self.stdout = stdout or sys.stdout
        self.stage_plan = build_stage_plan(self.config)

    @property
    def backend(self) -> ExecutionBackend:
        if self._backend is None:
            self._backend = V2ExecutionBackend()
        return self._backend

    def _emit(self, record_type: str, **values: Any) -> None:
        record = {
            "record_type": record_type,
            "run_fingerprint": self.fingerprint,
            **_jsonable(values),
        }
        print(
            "SANNA_RESULT "
            + json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")),
            file=self.stdout,
            flush=True,
        )

    def plan(self) -> dict[str, Any]:
        return {
            **self.stage_plan,
            "run_fingerprint": self.fingerprint,
            "run_path": f"runs/{self.config['run_name']}/{self.fingerprint}",
            "path_under_runs_root": f"{self.config['run_name']}/{self.fingerprint}",
            "execution_mode": "tiny-nonproduction" if self.tiny else "production",
        }

    def write_plan(self) -> dict[str, Any]:
        value = self.plan()
        if self.dry_run:
            self._emit("plan", dry_run=True, plan=value)
            return value
        plan_path = self.layout.plan_dir / "plan.json"
        identity_path = self.layout.plan_dir / "run.identity.json"
        config_snapshot_path = self.layout.plan_dir / "config.snapshot.json"
        lock_snapshot_path = self.layout.plan_dir / "requirements-kaggle.lock"
        manifest_path = self.layout.plan_dir / "artifact-manifest.json"
        outputs = {
            "plan": plan_path,
            "identity": identity_path,
            "config_snapshot": config_snapshot_path,
            "dependency_lock_snapshot": lock_snapshot_path,
        }
        # The committed plan is self-contained. Its fingerprint already binds the
        # source config, code commit, and dependency lock, so downstream stage
        # handoffs do not depend on their original machine-specific paths.
        inputs: dict[str, Path] = {}
        parameters = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "execution_mode": value["execution_mode"],
            "counts": value["counts"],
            "config_sha256": sha256_file(self.config_path),
            "dependency_lock_sha256": sha256_file(LOCK_FILE),
        }
        if strict_resume(
            manifest_path,
            stage="plan",
            run_fingerprint=self.fingerprint,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            relative_to=self.layout.run_dir,
        ):
            self._emit("plan", resumed=True, plan=value)
            return value
        if self.layout.plan_dir.exists() and any(self.layout.plan_dir.iterdir()):
            raise ResumeMismatchError(f"partial plan directory: {self.layout.plan_dir}")
        atomic_write_json(plan_path, value)
        atomic_write_json(config_snapshot_path, canonical_config(self.config))
        atomic_write_bytes(lock_snapshot_path, LOCK_FILE.read_bytes())
        atomic_write_json(
            identity_path,
            {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "run_name": self.config["run_name"],
                "run_fingerprint": self.fingerprint,
                "execution_mode": value["execution_mode"],
                "canonical_config": canonical_config(self.config),
            },
        )
        write_manifest(
            manifest_path,
            stage="plan",
            run_fingerprint=self.fingerprint,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            relative_to=self.layout.run_dir,
        )
        self._emit("plan", resumed=False, plan=value)
        return value

    def prepare(self, *, raw_root: str | Path | None = None) -> Mapping[str, Any]:
        if self.dry_run:
            self._emit("prepare", dry_run=True, jobs=0)
            return {"dry_run": True}
        self.write_plan()
        from .data import download_corpora, prepare_data

        raw_dir = Path(raw_root).resolve() if raw_root is not None else self.layout.raw_dir
        if raw_dir == self.layout.runs_root or raw_dir.is_relative_to(
            self.layout.runs_root
        ):
            raise RunnerError(
                "raw acquisition, including protected Czech test gold, must stay "
                "outside the portable runs tree"
            )
        raw_files = download_corpora(self.config, raw_dir)
        dataset = prepare_data(
            self.config,
            raw_files,
            self.layout.prepared_dir,
            run_fingerprint=self.fingerprint,
        )
        self._verify_prepared()
        self._emit(
            "prepare",
            dataset_manifest=str(self.layout.prepared_dir / "dataset_manifest.json"),
            test_labels_exposed=False,
        )
        return dataset

    def _verify_prepared(self) -> dict[str, Any]:
        path = self.layout.prepared_dir / "artifact-manifest.json"
        if not path.is_file():
            raise StageDependencyError("prepare must complete before this stage")
        manifest = load_manifest(path)
        if manifest.get("stage") != "data-preparation" or manifest.get("run_fingerprint") != self.fingerprint:
            raise StageDependencyError(f"prepared data does not match run {self.fingerprint}")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict):
            raise ArtifactIntegrityError(f"prepared manifest has invalid outputs: {path}")
        # Only prepared outputs are required after a portable stage handoff. Raw
        # acquisition inputs may live in a different cache on the next machine.
        for record in outputs.values():
            if not isinstance(record, dict):
                raise ArtifactIntegrityError(f"prepared manifest has malformed output: {path}")
            verify_file_record(record, relative_to=self.layout.prepared_dir)
        return manifest

    def _fresh_job_directory(self, job_dir: Path) -> None:
        if job_dir.exists() and any(job_dir.iterdir()):
            raise ResumeMismatchError(
                f"partial job directory has no valid commit manifest: {job_dir}"
            )
        job_dir.mkdir(parents=True, exist_ok=True)

    def _prune_checkpoints(
        self,
        *,
        stage_dir: Path,
        stage: str,
        checkpoint_manifests: Sequence[Path],
        retained_manifests: Sequence[Path],
        authorization_inputs: Mapping[str, Path],
    ) -> dict[str, Any]:
        """Delete authorized weight trees while preserving their sealed checksums.

        A committed pruning plan is written before deletion, making interruption
        recovery idempotent. Result JSON, checkpoint manifests, and job manifests
        remain intact; only checkpoint roots are removed.
        """
        manifests = tuple(dict.fromkeys(Path(path).resolve() for path in checkpoint_manifests))
        retained = {Path(path).resolve() for path in retained_manifests}
        if not retained <= set(manifests):
            raise ArtifactIntegrityError(f"{stage} pruning retains an unknown checkpoint")
        plan_path = stage_dir / "pruning.plan.json"
        plan_manifest_path = stage_dir / "pruning.plan.manifest.json"
        final_manifest_path = stage_dir / "pruning.manifest.json"
        entries = []
        roots: dict[Path, Path] = {}
        for manifest_path in manifests:
            value, root = _checkpoint_manifest_structure(
                manifest_path, expected_fingerprint=self.fingerprint
            )
            if root == self.layout.run_dir or not root.is_relative_to(self.layout.run_dir):
                raise ArtifactIntegrityError(
                    f"checkpoint pruning root escapes the run directory: {root}"
                )
            roots[manifest_path] = root
            entries.append(
                {
                    "checkpoint_manifest": _portable(manifest_path, self.layout.run_dir),
                    "checkpoint_manifest_sha256": sha256_file(manifest_path),
                    "checkpoint_root": _portable(root, self.layout.run_dir),
                    "retained": manifest_path in retained,
                    "files": value["files"],
                }
            )
        plan = {
            "schema_version": 1,
            "run_fingerprint": self.fingerprint,
            "stage": stage,
            "policy": "delete-checkpoint-roots-preserve-results-and-sealed-checksums-v1",
            "entries": entries,
        }
        plan_inputs = {
            **{str(name): Path(path) for name, path in authorization_inputs.items()},
            **{
                f"checkpoint-manifest-{index}": path
                for index, path in enumerate(manifests)
            },
        }
        plan_parameters = {
            "stage": stage,
            "retained": sorted(_portable(path, self.layout.run_dir) for path in retained),
        }
        verified: set[Path] = set()
        if strict_resume(
            plan_manifest_path,
            stage=f"{stage}-checkpoint-pruning-plan",
            run_fingerprint=self.fingerprint,
            inputs=plan_inputs,
            outputs={"plan": plan_path},
            parameters=plan_parameters,
            relative_to=self.layout.run_dir,
        ):
            if _read_json(plan_path) != plan:
                raise ResumeMismatchError(f"{stage} checkpoint pruning plan changed")
        else:
            for manifest_path in manifests:
                verify_checkpoint_manifest(
                    manifest_path, expected_fingerprint=self.fingerprint
                )
                verified.add(manifest_path)
            atomic_write_json(plan_path, plan)
            write_manifest(
                plan_manifest_path,
                stage=f"{stage}-checkpoint-pruning-plan",
                run_fingerprint=self.fingerprint,
                inputs=plan_inputs,
                outputs={"plan": plan_path},
                parameters=plan_parameters,
                relative_to=self.layout.run_dir,
            )

        for manifest_path, root in roots.items():
            if manifest_path not in retained and root.exists():
                if not root.is_dir():
                    raise ArtifactIntegrityError(f"checkpoint root is not a directory: {root}")
                shutil.rmtree(root)
        for manifest_path, root in roots.items():
            if manifest_path in retained:
                if manifest_path not in verified:
                    try:
                        verify_checkpoint_manifest(
                            manifest_path, expected_fingerprint=self.fingerprint
                        )
                    except ArtifactIntegrityError:
                        if not self._checkpoint_is_authorized_pruned(manifest_path):
                            raise
            elif root.exists():
                raise ArtifactIntegrityError(f"pruned checkpoint root still exists: {root}")

        final_inputs = {
            "pruning_plan_manifest": plan_manifest_path,
            **{str(name): Path(path) for name, path in authorization_inputs.items()},
        }
        final_parameters = {**plan_parameters, "entry_count": len(entries)}
        if final_manifest_path.is_file():
            strict_resume(
                final_manifest_path,
                stage=f"{stage}-checkpoint-pruning",
                run_fingerprint=self.fingerprint,
                inputs=final_inputs,
                outputs={"plan": plan_path},
                parameters=final_parameters,
                relative_to=self.layout.run_dir,
            )
        else:
            write_manifest(
                final_manifest_path,
                stage=f"{stage}-checkpoint-pruning",
                run_fingerprint=self.fingerprint,
                inputs=final_inputs,
                outputs={"plan": plan_path},
                parameters=final_parameters,
                relative_to=self.layout.run_dir,
            )
        return plan

    def _checkpoint_is_authorized_pruned(self, manifest_path: Path) -> bool:
        target = Path(manifest_path).resolve()
        for stage_dir, stage in (
            (self.layout.budget_dir, "budget"),
            (self.layout.adapt_dir, "adapt"),
        ):
            final_manifest_path = stage_dir / "pruning.manifest.json"
            plan_manifest_path = stage_dir / "pruning.plan.manifest.json"
            if final_manifest_path.is_file():
                authority_path = final_manifest_path
                expected_stage = f"{stage}-checkpoint-pruning"
            elif plan_manifest_path.is_file():
                # The plan is the durable authorization boundary written before
                # deletion, so an interrupted deletion can resume idempotently.
                authority_path = plan_manifest_path
                expected_stage = f"{stage}-checkpoint-pruning-plan"
            else:
                continue
            authority = load_manifest(authority_path)
            if (
                authority.get("stage") != expected_stage
                or authority.get("run_fingerprint") != self.fingerprint
            ):
                raise ArtifactIntegrityError(
                    f"checkpoint pruning manifest identity mismatch: {authority_path}"
                )
            verify_manifest_files(authority, relative_to=self.layout.run_dir)
            plan = _read_json(stage_dir / "pruning.plan.json")
            entries = plan.get("entries")
            if not isinstance(entries, list):
                raise ArtifactIntegrityError(f"malformed checkpoint pruning plan: {stage_dir}")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ArtifactIntegrityError(f"malformed checkpoint pruning plan: {stage_dir}")
                value = entry.get("checkpoint_manifest")
                path = Path(value) if isinstance(value, str) else None
                if path is not None and not path.is_absolute():
                    path = self.layout.run_dir / path
                if path is not None and path.resolve() == target:
                    if entry.get("retained") is not False:
                        continue
                    if entry.get("checkpoint_manifest_sha256") != sha256_file(target):
                        raise ArtifactIntegrityError(
                            f"pruned checkpoint manifest changed: {target}"
                        )
                    _, root = _checkpoint_manifest_structure(
                        target, expected_fingerprint=self.fingerprint
                    )
                    if root.exists():
                        raise ArtifactIntegrityError(
                            f"authorized pruned checkpoint unexpectedly exists: {root}"
                        )
                    return True
        return False

    def _run_checkpoint_job(
        self,
        *,
        job: Mapping[str, Any],
        job_dir: Path,
        inputs: Mapping[str, Path],
        parameters: Mapping[str, Any],
        execute: Callable[[Path], CheckpointExecution],
    ) -> dict[str, Any]:
        result_path = job_dir / "result.json"
        checkpoint_manifest_path = job_dir / "checkpoint.manifest.json"
        artifact_manifest_path = job_dir / "artifact-manifest.json"
        outputs = {
            "result": result_path,
            "checkpoint_manifest": checkpoint_manifest_path,
        }
        if strict_resume(
            artifact_manifest_path,
            stage=str(job["stage"]),
            run_fingerprint=self.fingerprint,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            relative_to=self.layout.run_dir,
        ):
            try:
                verify_checkpoint_manifest(
                    checkpoint_manifest_path, expected_fingerprint=self.fingerprint
                )
            except ArtifactIntegrityError:
                if not self._checkpoint_is_authorized_pruned(checkpoint_manifest_path):
                    raise
            result = _read_json(result_path)
            self._emit("job", resumed=True, result=result)
            return result
        self._fresh_job_directory(job_dir)
        execution = execute(job_dir)
        checkpoint_metadata = {
            "job": dict(job),
            "execution_mode": "tiny-nonproduction" if self.tiny else "production",
            **dict(execution.metadata),
        }
        write_checkpoint_manifest(
            checkpoint_manifest_path,
            checkpoint_dir=Path(execution.checkpoint_dir),
            run_fingerprint=self.fingerprint,
            metadata=checkpoint_metadata,
        )
        result = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "run_fingerprint": self.fingerprint,
            "job": dict(job),
            "metrics": _jsonable(execution.metrics),
            "checkpoint_manifest": _portable(
                checkpoint_manifest_path, self.layout.run_dir
            ),
            "experimental_valid": not self.tiny,
            "metadata": _jsonable(execution.metadata),
        }
        atomic_write_json(result_path, result)
        write_manifest(
            artifact_manifest_path,
            stage=str(job["stage"]),
            run_fingerprint=self.fingerprint,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            relative_to=self.layout.run_dir,
        )
        self._emit("job", resumed=False, result=result)
        return result

    def _budget_job_dir(self, size: int, seed: int) -> Path:
        return self.layout.budget_dir / "jobs" / f"n{size}-seed{seed}"

    def _budget_checkpoint_manifest(self, size: int, seed: int) -> Path:
        return self._budget_job_dir(size, seed) / "checkpoint.manifest.json"

    def _verify_checkpoint_job_commit(
        self,
        *,
        job_dir: Path,
        expected_stage: str,
        expected_job: Mapping[str, Any],
        require_checkpoint: bool,
    ) -> dict[str, Any]:
        artifact_path = job_dir / "artifact-manifest.json"
        result_path = job_dir / "result.json"
        checkpoint_path = job_dir / "checkpoint.manifest.json"
        manifest = load_manifest(artifact_path)
        if (
            manifest.get("stage") != expected_stage
            or manifest.get("run_fingerprint") != self.fingerprint
        ):
            raise ArtifactIntegrityError(f"job manifest identity mismatch: {artifact_path}")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {"result", "checkpoint_manifest"}:
            raise ArtifactIntegrityError(f"job manifest outputs are malformed: {artifact_path}")
        committed = {
            name: verify_file_record(record, relative_to=self.layout.run_dir).resolve()
            for name, record in outputs.items()
            if isinstance(record, dict)
        }
        if committed != {
            "result": result_path.resolve(),
            "checkpoint_manifest": checkpoint_path.resolve(),
        }:
            raise ArtifactIntegrityError(f"job manifest commits unexpected paths: {artifact_path}")
        result = _read_json(result_path)
        if result.get("run_fingerprint") != self.fingerprint:
            raise ArtifactIntegrityError(f"job result fingerprint mismatch: {result_path}")
        actual_job = result.get("job")
        if not isinstance(actual_job, dict) or any(
            actual_job.get(name) != value for name, value in expected_job.items()
        ):
            raise ArtifactIntegrityError(f"job result identity mismatch: {result_path}")
        checkpoint, _ = _checkpoint_manifest_structure(
            checkpoint_path, expected_fingerprint=self.fingerprint
        )
        checkpoint_metadata = checkpoint.get("metadata")
        checkpoint_job = (
            checkpoint_metadata.get("job") if isinstance(checkpoint_metadata, dict) else None
        )
        if not isinstance(checkpoint_job, dict) or any(
            checkpoint_job.get(name) != value for name, value in expected_job.items()
        ):
            raise ArtifactIntegrityError(
                f"checkpoint job identity mismatch: {checkpoint_path}"
            )
        if require_checkpoint:
            verify_checkpoint_manifest(
                checkpoint_path, expected_fingerprint=self.fingerprint
            )
        return result

    def _budget_selection_contract(
        self,
    ) -> tuple[dict[str, Path], dict[str, Any]]:
        sizes = tuple(int(size) for size in self.config["seed_sizes"])
        seeds = tuple(int(seed) for seed in self.config["random_seeds"])
        inputs: dict[str, Path] = {}
        for size in sizes:
            for seed in seeds:
                base = self._budget_job_dir(size, seed)
                inputs[f"result-n{size}-seed{seed}"] = base / "result.json"
                inputs[f"manifest-n{size}-seed{seed}"] = base / "artifact-manifest.json"
        policy = self.config["selection"]
        parameters = {
            "seed_sizes": list(sizes),
            "seeds": list(seeds),
            "sentence_tolerance": float(policy["budget_sentence_tolerance"]),
            "token_tolerance": float(policy["budget_token_tolerance"]),
        }
        return inputs, parameters

    def _verify_and_prune_budget_selection(self) -> dict[str, Any]:
        selection_path = self.layout.budget_dir / "selection.json"
        selection_manifest_path = self.layout.budget_dir / "selection.manifest.json"
        inputs, parameters = self._budget_selection_contract()
        if not strict_resume(
            selection_manifest_path,
            stage="budget-selection",
            run_fingerprint=self.fingerprint,
            inputs=inputs,
            outputs={"selection": selection_path},
            parameters=parameters,
            relative_to=self.layout.run_dir,
        ):
            raise StageDependencyError("budget selection is not committed")
        selection = _read_json(selection_path)
        sizes = tuple(int(size) for size in self.config["seed_sizes"])
        seeds = tuple(int(seed) for seed in self.config["random_seeds"])
        selected_budget = int(selection.get("selected_budget", -1))
        if selected_budget not in sizes:
            raise ArtifactIntegrityError("budget selection names an unknown budget")
        manifests = []
        for size in sizes:
            for seed in seeds:
                expected_job = {
                    "stage": "budget",
                    "variant": "slovakbert",
                    "size": size,
                    "seed": seed,
                }
                expected_job["id"] = _job_id(expected_job)
                self._verify_checkpoint_job_commit(
                    job_dir=self._budget_job_dir(size, seed),
                    expected_stage="budget",
                    expected_job=expected_job,
                    require_checkpoint=False,
                )
                manifests.append(self._budget_checkpoint_manifest(size, seed))
        retained = [
            self._budget_checkpoint_manifest(selected_budget, seed) for seed in seeds
        ]
        self._prune_checkpoints(
            stage_dir=self.layout.budget_dir,
            stage="budget",
            checkpoint_manifests=manifests,
            retained_manifests=retained,
            authorization_inputs={"selection_manifest": selection_manifest_path},
        )
        return selection

    def budget(self) -> Mapping[str, Any]:
        if self.dry_run:
            jobs = [job for job in self.plan()["jobs"] if job["stage"] == "budget"]
            self._emit("budget", dry_run=True, jobs=jobs)
            return {"dry_run": True, "jobs": jobs}
        self.write_plan()
        self._verify_prepared()
        prepared_manifest = self.layout.prepared_dir / "artifact-manifest.json"
        selection_dev = self.layout.prepared_dir / "czech" / "dev" / "selection.jsonl"
        training_parameters = dict(self.config["training"])
        sizes = tuple(int(size) for size in self.config["seed_sizes"])
        seeds = tuple(int(seed) for seed in self.config["random_seeds"])
        existing_selection_manifest = self.layout.budget_dir / "selection.manifest.json"
        if existing_selection_manifest.is_file():
            selection_value = self._verify_and_prune_budget_selection()
            selected_budget = int(selection_value["selected_budget"])
            self._write_handoff(
                "budget",
                {
                    "selected_budget": selected_budget,
                    "selection": _portable(
                        self.layout.budget_dir / "selection.json", self.layout.run_dir
                    ),
                    "selected_checkpoint_manifests": [
                        _portable(
                            self._budget_checkpoint_manifest(selected_budget, seed),
                            self.layout.run_dir,
                        )
                        for seed in seeds
                    ],
                },
                {"selection_manifest": existing_selection_manifest},
            )
            self._emit("budget-selection", resumed=True, selection=selection_value)
            return selection_value
        results: dict[int, dict[int, dict[str, Any]]] = {
            size: {} for size in sizes
        }
        planned_jobs = (
            job for job in self.stage_plan["jobs"] if job["stage"] == "budget"
        )
        for planned_job in planned_jobs:
            size_int, seed_int = int(planned_job["size"]), int(planned_job["seed"])
            job = {**planned_job, "model_family": "slovakbert"}
            train_path = (
                self.layout.prepared_dir
                / "czech"
                / "gold"
                / f"seed_{seed_int}"
                / f"n_{size_int}.jsonl"
            )
            inputs = {
                "prepared_manifest": prepared_manifest,
                "train": train_path,
                "selection_dev": selection_dev,
            }
            parameters = {
                "job": job,
                "training": training_parameters,
                "model": dict(self.config["models"]["slovakbert"]),
                "tiny": self.tiny,
            }
            result = self._run_checkpoint_job(
                job=job,
                job_dir=self._budget_job_dir(size_int, seed_int),
                inputs=inputs,
                parameters=parameters,
                execute=lambda job_dir, job=job, train_path=train_path: self.backend.train_supervised(
                    job=job,
                    train_path=train_path,
                    dev_path=selection_dev,
                    initialization_manifest=None,
                    output_dir=job_dir / "checkpoint",
                    config=self.config,
                    tiny=self.tiny,
                ),
            )
            results[size_int][seed_int] = result

        candidates = [
            SelectionCandidate(
                name=f"slovakbert-n{size}",
                budget=size,
                metrics_by_seed={
                    seed: results[size][seed]["metrics"] for seed in seeds
                },
            )
            for size in sizes
        ]
        policy = self.config["selection"]
        selected = select_budget(
            candidates,
            sentence_tolerance=float(policy["budget_sentence_tolerance"]),
            token_tolerance=float(policy["budget_token_tolerance"]),
        )
        selection_path = self.layout.budget_dir / "selection.json"
        selection_manifest_path = self.layout.budget_dir / "selection.manifest.json"
        selection_inputs, selection_parameters = self._budget_selection_contract()
        if strict_resume(
            selection_manifest_path,
            stage="budget-selection",
            run_fingerprint=self.fingerprint,
            inputs=selection_inputs,
            outputs={"selection": selection_path},
            parameters=selection_parameters,
            relative_to=self.layout.run_dir,
        ):
            selection_value = _read_json(selection_path)
        else:
            selection_value = {
                "schema_version": 1,
                "run_fingerprint": self.fingerprint,
                "selected_budget": selected.selected.candidate.budget,
                "reference_budget": selected.reference.candidate.budget,
                "eligible_budgets": list(selected.eligible_budgets),
                "policy": selection_parameters,
                "candidates": [
                    {
                        "name": candidate.candidate.name,
                        "budget": candidate.candidate.budget,
                        "mean_sentence_pass": candidate.mean_sentence_pass,
                        "mean_token_accuracy": candidate.mean_token_accuracy,
                        "mean_exact_match": candidate.mean_exact_match,
                        "aggregate": _jsonable(candidate.aggregate),
                    }
                    for candidate in (
                        selected.selected,
                        *(
                            item
                            for item in [selected.reference]
                            if item.candidate.name != selected.selected.candidate.name
                        ),
                    )
                ],
                "all_aggregates": {
                    candidate.name: _jsonable(aggregate_candidate(candidate).aggregate)
                    for candidate in candidates
                },
            }
            atomic_write_json(selection_path, selection_value)
            write_manifest(
                selection_manifest_path,
                stage="budget-selection",
                run_fingerprint=self.fingerprint,
                inputs=selection_inputs,
                outputs={"selection": selection_path},
                parameters=selection_parameters,
                relative_to=self.layout.run_dir,
            )
        selection_value = self._verify_and_prune_budget_selection()
        self._write_handoff(
            "budget",
            {
                "selected_budget": selection_value["selected_budget"],
                "selection": _portable(selection_path, self.layout.run_dir),
                "selected_checkpoint_manifests": [
                    _portable(
                        self._budget_checkpoint_manifest(
                            int(selection_value["selected_budget"]), int(seed)
                        ),
                        self.layout.run_dir,
                    )
                    for seed in seeds
                ],
            },
            {"selection_manifest": selection_manifest_path},
        )
        self._emit("budget-selection", selection=selection_value)
        return selection_value

    def _write_handoff(
        self, stage: str, payload: Mapping[str, Any], inputs: Mapping[str, Path]
    ) -> dict[str, Any]:
        stage_dir = self.layout.run_dir / stage
        handoff_path = stage_dir / "handoff.json"
        manifest_path = stage_dir / "handoff.manifest.json"
        value = {
            "schema_version": 1,
            "stage": stage,
            "run_name": self.config["run_name"],
            "run_fingerprint": self.fingerprint,
            "run_path": f"runs/{self.config['run_name']}/{self.fingerprint}",
            "path_under_runs_root": f"{self.config['run_name']}/{self.fingerprint}",
            **_jsonable(payload),
        }
        parameters = {"payload": value}
        if strict_resume(
            manifest_path,
            stage=f"{stage}-handoff",
            run_fingerprint=self.fingerprint,
            inputs=inputs,
            outputs={"handoff": handoff_path},
            parameters=parameters,
            relative_to=self.layout.run_dir,
        ):
            return _read_json(handoff_path)
        atomic_write_json(handoff_path, value)
        write_manifest(
            manifest_path,
            stage=f"{stage}-handoff",
            run_fingerprint=self.fingerprint,
            inputs=inputs,
            outputs={"handoff": handoff_path},
            parameters=parameters,
            relative_to=self.layout.run_dir,
        )
        return value

    def _adapt_checkpoint_manifest(self, category: str, variant: str, seed: int | None = None) -> Path:
        name = variant if seed is None else f"{variant}-seed{seed}"
        return self.layout.adapt_dir / category / name / "checkpoint.manifest.json"

    def _adapt_job_dir(self, category: str, variant: str, seed: int | None = None) -> Path:
        return self._adapt_checkpoint_manifest(category, variant, seed).parent

    def _require_stage_artifact(
        self,
        *,
        artifact_path: Path,
        manifest_path: Path,
        expected_stage: str,
        dependency_message: str,
    ) -> dict[str, Any]:
        if not artifact_path.is_file() or not manifest_path.is_file():
            raise StageDependencyError(dependency_message)
        manifest = load_manifest(manifest_path)
        if (
            manifest.get("stage") != expected_stage
            or manifest.get("run_fingerprint") != self.fingerprint
        ):
            raise StageDependencyError(
                f"{expected_stage} artifact does not match run {self.fingerprint}"
            )
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict):
            raise ArtifactIntegrityError(
                f"{expected_stage} manifest has malformed outputs"
            )
        committed_paths = []
        for record in outputs.values():
            if not isinstance(record, dict):
                raise ArtifactIntegrityError(
                    f"{expected_stage} manifest contains a malformed output"
                )
            committed_paths.append(
                verify_file_record(
                    record, relative_to=self.layout.run_dir
                ).resolve()
            )
        if artifact_path.resolve() not in committed_paths:
            raise ArtifactIntegrityError(
                f"{expected_stage} manifest does not commit {artifact_path}"
            )
        return _read_json(artifact_path)

    def _require_budget_selection(self) -> dict[str, Any]:
        selection_manifest = self.layout.budget_dir / "selection.manifest.json"
        if not selection_manifest.is_file():
            raise StageDependencyError("budget must complete before adapt")
        return self._verify_and_prune_budget_selection()

    def _adapt_selection_contract(
        self, selected_budget: int, seeds: Sequence[int]
    ) -> tuple[dict[str, Path], dict[str, Any]]:
        inputs: dict[str, Path] = {}
        for arm in WINNING_ARM_NAMES:
            for seed in seeds:
                result = self._arm_result_path(arm, selected_budget, int(seed))
                inputs[f"result-{arm}-seed{seed}"] = result
                inputs[f"manifest-{arm}-seed{seed}"] = (
                    result.parent / "artifact-manifest.json"
                )
        parameters = {
            "selected_budget": selected_budget,
            "seeds": [int(seed) for seed in seeds],
            "token_noninferiority": float(
                self.config["selection"]["adaptation_token_noninferiority"]
            ),
            "candidate_complexity": dict(ARM_COMPLEXITY),
        }
        return inputs, parameters

    def adapt(self) -> Mapping[str, Any]:
        if self.dry_run:
            jobs = [job for job in self.plan()["jobs"] if job["stage"] != "budget"]
            self._emit("adapt", dry_run=True, jobs=jobs)
            return {"dry_run": True, "jobs": jobs}
        self.write_plan()
        self._verify_prepared()
        budget_selection = self._require_budget_selection()
        selected_budget = int(budget_selection["selected_budget"])
        seeds = tuple(int(seed) for seed in self.config["random_seeds"])
        planned_by_stage = {
            stage: [
                dict(job)
                for job in self.stage_plan["jobs"]
                if job["stage"] == stage
            ]
            for stage in ("lapt", "transfer", "reference", "adapt")
        }
        prepared_manifest = self.layout.prepared_dir / "artifact-manifest.json"
        selection_dev = self.layout.prepared_dir / "czech" / "dev" / "selection.jsonl"
        czech_lapt_train_forms = (
            self.layout.prepared_dir / "czech" / "lapt_train_forms.jsonl"
        )
        slovak_train = self.layout.prepared_dir / "slovak" / "train.jsonl"
        slovak_dev = self.layout.prepared_dir / "slovak" / "dev.jsonl"

        lapt_job = {**planned_by_stage["lapt"][0], "model_family": "slovakbert"}
        lapt_dir = self._adapt_job_dir("shared", lapt_job["variant"])
        self._run_checkpoint_job(
            job=lapt_job,
            job_dir=lapt_dir,
            inputs={
                "prepared_manifest": prepared_manifest,
                "czech_lapt_train_forms": czech_lapt_train_forms,
            },
            parameters={"job": lapt_job, "lapt": dict(self.config["lapt"]), "tiny": self.tiny},
            execute=lambda job_dir: self.backend.train_lapt(
                train_path=czech_lapt_train_forms,
                output_dir=job_dir / "checkpoint",
                config=self.config,
                tiny=self.tiny,
            ),
        )
        lapt_manifest = lapt_dir / "checkpoint.manifest.json"

        transfer_manifests: dict[str, Path] = {}
        transfer_initializations = {
            "slovakbert-slovak-upos": (None, "base"),
            "slovakbert-czech-lapt-slovak-upos": (lapt_manifest, "lapt"),
        }
        for planned_job in planned_by_stage["transfer"]:
            variant = str(planned_job["variant"])
            initialization_manifest, initialization = transfer_initializations[variant]
            job = {
                **planned_job,
                "model_family": "slovakbert",
                "initialization": initialization,
            }
            job_dir = self._adapt_job_dir("shared", variant)
            inputs = {
                "prepared_manifest": prepared_manifest,
                "slovak_train": slovak_train,
                "slovak_dev": slovak_dev,
            }
            if initialization_manifest is not None:
                inputs["initialization_checkpoint"] = initialization_manifest
            self._run_checkpoint_job(
                job=job,
                job_dir=job_dir,
                inputs=inputs,
                parameters={
                    "job": job,
                    "transfer": dict(self.config["transfer"]),
                    "tiny": self.tiny,
                },
                execute=lambda job_dir, job=job, initialization_manifest=initialization_manifest: self.backend.train_transfer(
                    job=job,
                    slovak_train_path=slovak_train,
                    slovak_dev_path=slovak_dev,
                    initialization_manifest=initialization_manifest,
                    output_dir=job_dir / "checkpoint",
                    config=self.config,
                    tiny=self.tiny,
                ),
            )
            transfer_manifests[variant] = job_dir / "checkpoint.manifest.json"

        arm_results: dict[str, dict[int, dict[str, Any]]] = {
            name: {} for name in WINNING_ARM_NAMES
        }
        for seed in seeds:
            arm_results["slovakbert-direct"][seed] = _read_json(
                self._budget_job_dir(selected_budget, seed) / "result.json"
            )

        initialization_by_arm = {
            "slovakbert-czech-lapt": lapt_manifest,
            "slovakbert-slovak-upos": transfer_manifests["slovakbert-slovak-upos"],
            "slovakbert-czech-lapt-slovak-upos": transfer_manifests[
                "slovakbert-czech-lapt-slovak-upos"
            ],
        }
        selected_job_specs = [
            (job, "xlmr-reference", "reference", "xlmr", None)
            for job in planned_by_stage["reference"]
        ] + [
            (
                job,
                str(job["variant"]),
                "arms",
                "slovakbert",
                initialization_by_arm[str(job["variant"])],
            )
            for job in planned_by_stage["adapt"]
        ]
        for planned_job, arm, category, model_family, initialization_manifest in selected_job_specs:
            seed = int(planned_job["seed"])
            train_path = (
                self.layout.prepared_dir
                / "czech"
                / "gold"
                / f"seed_{seed}"
                / f"n_{selected_budget}.jsonl"
            )
            job = {
                **planned_job,
                "arm": arm,
                "model_family": model_family,
                "size": selected_budget,
            }
            job_dir = self._adapt_job_dir(
                category, str(planned_job["variant"]), seed
            )
            inputs = {
                "prepared_manifest": prepared_manifest,
                "budget_selection": self.layout.budget_dir / "selection.json",
                "train": train_path,
                "selection_dev": selection_dev,
            }
            if initialization_manifest is not None:
                inputs["initialization_checkpoint"] = initialization_manifest
            parameters = {
                "job": job,
                "training": dict(self.config["training"]),
                "tiny": self.tiny,
            }
            if initialization_manifest is None:
                parameters["model"] = dict(self.config["models"][model_family])
            result = self._run_checkpoint_job(
                job=job,
                job_dir=job_dir,
                inputs=inputs,
                parameters=parameters,
                execute=lambda job_dir, job=job, train_path=train_path, initialization_manifest=initialization_manifest: self.backend.train_supervised(
                    job=job,
                    train_path=train_path,
                    dev_path=selection_dev,
                    initialization_manifest=initialization_manifest,
                    output_dir=job_dir / "checkpoint",
                    config=self.config,
                    tiny=self.tiny,
                ),
            )
            arm_results[arm][seed] = result

        candidates = [
            SelectionCandidate(
                name="slovakbert-direct",
                metrics_by_seed={seed: arm_results["slovakbert-direct"][seed]["metrics"] for seed in seeds},
                complexity=ARM_COMPLEXITY["slovakbert-direct"],
                budget=selected_budget,
            ),
            SelectionCandidate(
                name="xlmr-reference",
                metrics_by_seed={seed: arm_results["xlmr-reference"][seed]["metrics"] for seed in seeds},
                complexity=ARM_COMPLEXITY["xlmr-reference"],
                budget=selected_budget,
            ),
            *[
                SelectionCandidate(
                    name=arm,
                    metrics_by_seed={seed: arm_results[arm][seed]["metrics"] for seed in seeds},
                    complexity=ARM_COMPLEXITY[arm],
                    budget=selected_budget,
                    adapted_from="slovakbert-direct",
                )
                for arm in ADAPTED_VARIANTS
            ],
        ]
        selected = select_candidate(
            candidates,
            adaptation_token_noninferiority=float(
                self.config["selection"]["adaptation_token_noninferiority"]
            ),
        )
        winning_arm = selected.selected.candidate.name
        winning_manifests = [
            self._arm_checkpoint_manifest(winning_arm, selected_budget, seed)
            for seed in seeds
        ]
        selection_path = self.layout.adapt_dir / "selection.json"
        selection_manifest_path = self.layout.adapt_dir / "selection.manifest.json"
        selection_inputs, selection_parameters = self._adapt_selection_contract(
            selected_budget, seeds
        )
        if strict_resume(
            selection_manifest_path,
            stage="adaptation-selection",
            run_fingerprint=self.fingerprint,
            inputs=selection_inputs,
            outputs={"selection": selection_path},
            parameters=selection_parameters,
            relative_to=self.layout.run_dir,
        ):
            selection_value = _read_json(selection_path)
            if selection_value.get("winning_arm") != winning_arm:
                raise ResumeMismatchError("recomputed adaptation winner differs from artifact")
        else:
            selection_value = {
                "schema_version": 1,
                "run_fingerprint": self.fingerprint,
                "selected_budget": selected_budget,
                "winning_arm": winning_arm,
                "winning_checkpoint_manifests": [
                    _portable(path, self.layout.run_dir) for path in winning_manifests
                ],
                "policy": selection_parameters,
                "rejected_by_noninferiority": list(selected.rejected_by_noninferiority),
                "ranked_eligible": [item.candidate.name for item in selected.ranked_eligible],
                "candidates": {
                    item.candidate.name: {
                        "mean_sentence_pass": item.mean_sentence_pass,
                        "mean_token_accuracy": item.mean_token_accuracy,
                        "mean_exact_match": item.mean_exact_match,
                        "aggregate": _jsonable(item.aggregate),
                    }
                    for item in (
                        selected.selected,
                        *[
                            entry
                            for entry in selected.ranked_eligible
                            if entry.candidate.name != selected.selected.candidate.name
                        ],
                    )
                },
            }
            atomic_write_json(selection_path, selection_value)
            write_manifest(
                selection_manifest_path,
                stage="adaptation-selection",
                run_fingerprint=self.fingerprint,
                inputs=selection_inputs,
                outputs={"selection": selection_path},
                parameters=selection_parameters,
                relative_to=self.layout.run_dir,
            )

        calibration_dir = self.layout.adapt_dir / "calibration"
        calibration_path = calibration_dir / "calibration.json"
        ensemble_path = calibration_dir / "ensemble.json"
        calibration_manifest_path = calibration_dir / "artifact-manifest.json"
        calibration_inputs = {
            "adaptation_selection": selection_path,
            "selection_dev": selection_dev,
            "calibration_fit": self.layout.prepared_dir / "czech" / "dev" / "calibration_fit.jsonl",
            "calibration_assessment": self.layout.prepared_dir / "czech" / "dev" / "calibration_assessment.jsonl",
            **{
                f"checkpoint-seed{seed}": path
                for seed, path in zip(seeds, winning_manifests)
            },
        }
        calibration_parameters = {
            "winning_arm": winning_arm,
            "seeds": list(seeds),
            "ensemble": "unweighted-arithmetic-mean-probabilities",
            "calibration": dict(self.config["calibration"]),
            "tiny": self.tiny,
        }
        if strict_resume(
            calibration_manifest_path,
            stage="ensemble-calibration",
            run_fingerprint=self.fingerprint,
            inputs=calibration_inputs,
            outputs={"calibration": calibration_path, "ensemble": ensemble_path},
            parameters=calibration_parameters,
            relative_to=self.layout.run_dir,
        ):
            calibration_value = _read_json(calibration_path)
            ensemble_value = _read_json(ensemble_path)
        else:
            if calibration_dir.exists() and any(calibration_dir.iterdir()):
                raise ResumeMismatchError(f"partial calibration directory: {calibration_dir}")
            execution = self.backend.calibrate_ensemble(
                checkpoint_manifests=winning_manifests,
                selection_path=selection_dev,
                calibration_fit_path=calibration_inputs["calibration_fit"],
                calibration_assessment_path=calibration_inputs["calibration_assessment"],
                config=self.config,
                tiny=self.tiny,
            )
            calibration_value = {
                "schema_version": 1,
                "run_fingerprint": self.fingerprint,
                "winning_arm": winning_arm,
                **_jsonable(execution.calibration),
            }
            ensemble_value = {
                "schema_version": 1,
                "run_fingerprint": self.fingerprint,
                "winning_arm": winning_arm,
                "checkpoint_manifests": [
                    _portable(path, self.layout.run_dir) for path in winning_manifests
                ],
                **_jsonable(execution.ensemble),
            }
            atomic_write_json(calibration_path, calibration_value)
            atomic_write_json(ensemble_path, ensemble_value)
            write_manifest(
                calibration_manifest_path,
                stage="ensemble-calibration",
                run_fingerprint=self.fingerprint,
                inputs=calibration_inputs,
                outputs={"calibration": calibration_path, "ensemble": ensemble_path},
                parameters=calibration_parameters,
                relative_to=self.layout.run_dir,
            )
        self._write_handoff(
            "adapt",
            {
                "winning_arm": winning_arm,
                "selected_budget": selected_budget,
                "selection": _portable(selection_path, self.layout.run_dir),
                "checkpoint_manifests": [
                    _portable(path, self.layout.run_dir) for path in winning_manifests
                ],
                "ensemble": _portable(ensemble_path, self.layout.run_dir),
                "calibration": _portable(calibration_path, self.layout.run_dir),
            },
            {
                "selection_manifest": selection_manifest_path,
                "calibration_manifest": calibration_manifest_path,
            },
        )
        self._prune_adapt_checkpoints(
            selected_budget=selected_budget,
            winning_manifests=winning_manifests,
        )
        result = {
            "selection": selection_value,
            "ensemble": ensemble_value,
            "calibration": calibration_value,
        }
        self._emit("adaptation-selection", result=result)
        return result

    def _arm_checkpoint_manifest(self, arm: str, budget: int, seed: int) -> Path:
        if arm == "slovakbert-direct":
            return self._budget_checkpoint_manifest(budget, seed)
        if arm == "xlmr-reference":
            return self._adapt_checkpoint_manifest("reference", "xlmr", seed)
        if arm in ADAPTED_VARIANTS:
            return self._adapt_checkpoint_manifest("arms", arm, seed)
        raise RunnerError(f"unknown arm: {arm}")

    def _arm_result_path(self, arm: str, budget: int, seed: int) -> Path:
        return self._arm_checkpoint_manifest(arm, budget, seed).parent / "result.json"

    def _adapt_created_checkpoint_manifests(self) -> list[Path]:
        manifests = [
            self._adapt_checkpoint_manifest(
                "shared", "slovakbert-czech-lapt"
            ),
            self._adapt_checkpoint_manifest(
                "shared", "slovakbert-slovak-upos"
            ),
            self._adapt_checkpoint_manifest(
                "shared", "slovakbert-czech-lapt-slovak-upos"
            ),
        ]
        seeds = tuple(int(seed) for seed in self.config["random_seeds"])
        manifests.extend(
            self._adapt_checkpoint_manifest("reference", "xlmr", seed)
            for seed in seeds
        )
        manifests.extend(
            self._adapt_checkpoint_manifest("arms", arm, seed)
            for arm in ADAPTED_VARIANTS
            for seed in seeds
        )
        return manifests

    def _prune_adapt_checkpoints(
        self,
        *,
        selected_budget: int,
        winning_manifests: Sequence[Path],
    ) -> dict[str, Any]:
        seeds = tuple(int(seed) for seed in self.config["random_seeds"])
        candidates = [
            *self._adapt_created_checkpoint_manifests(),
            *[
                self._budget_checkpoint_manifest(selected_budget, seed)
                for seed in seeds
            ],
        ]
        retained = [Path(path) for path in winning_manifests]
        return self._prune_checkpoints(
            stage_dir=self.layout.adapt_dir,
            stage="adapt",
            checkpoint_manifests=candidates,
            retained_manifests=retained,
            authorization_inputs={
                "selection_manifest": self.layout.adapt_dir / "selection.manifest.json",
                "calibration_manifest": self.layout.adapt_dir
                / "calibration"
                / "artifact-manifest.json",
                "handoff_manifest": self.layout.adapt_dir / "handoff.manifest.json",
            },
        )

    def _require_adapt_handoff(self) -> dict[str, Any]:
        return self._require_stage_artifact(
            artifact_path=self.layout.adapt_dir / "handoff.json",
            manifest_path=self.layout.adapt_dir / "handoff.manifest.json",
            expected_stage="adapt-handoff",
            dependency_message="adapt must complete before this stage",
        )

    def _handoff_paths(
        self, handoff: Mapping[str, Any], *, verify: bool = True
    ) -> list[Path]:
        values = handoff.get("checkpoint_manifests")
        expected_count = len(self.config["random_seeds"])
        if not isinstance(values, list) or len(values) != expected_count:
            raise ArtifactIntegrityError(
                "adapt handoff must contain one checkpoint per configured seed"
            )
        paths = []
        for value in values:
            if not isinstance(value, str):
                raise ArtifactIntegrityError("checkpoint handoff paths must be strings")
            path = Path(value)
            path = path if path.is_absolute() else self.layout.run_dir / path
            if verify:
                verify_checkpoint_manifest(path, expected_fingerprint=self.fingerprint)
            paths.append(path)
        return paths

    def benchmark(self) -> Mapping[str, Any]:
        if self.dry_run:
            self._emit("benchmark", dry_run=True, optional=True)
            return {"dry_run": True}
        handoff = self._require_adapt_handoff()
        checkpoints = self._handoff_paths(handoff)
        benchmark_path = self.layout.benchmark_dir / "benchmark.json"
        manifest_path = self.layout.benchmark_dir / "artifact-manifest.json"
        data_path = self.layout.prepared_dir / "czech" / "dev" / "selection.jsonl"
        inputs = {
            "adapt_handoff": self.layout.adapt_dir / "handoff.json",
            "data": data_path,
            **{f"checkpoint-{index}": path for index, path in enumerate(checkpoints)},
        }
        parameters = {
            "winning_arm": handoff["winning_arm"],
            "members": 3,
            "timing_scope": "three-member checkpoint load, tokenization, and sequential inference",
            "tiny": self.tiny,
        }
        if strict_resume(
            manifest_path,
            stage="benchmark",
            run_fingerprint=self.fingerprint,
            inputs=inputs,
            outputs={"benchmark": benchmark_path},
            parameters=parameters,
            relative_to=self.layout.run_dir,
        ):
            value = _read_json(benchmark_path)
        else:
            value = {
                "schema_version": 1,
                "run_fingerprint": self.fingerprint,
                "selection_influence": False,
                **_jsonable(
                    self.backend.benchmark_ensemble(
                        checkpoint_manifests=checkpoints,
                        data_path=data_path,
                        config=self.config,
                        tiny=self.tiny,
                    )
                ),
            }
            atomic_write_json(benchmark_path, value)
            write_manifest(
                manifest_path,
                stage="benchmark",
                run_fingerprint=self.fingerprint,
                inputs=inputs,
                outputs={"benchmark": benchmark_path},
                parameters=parameters,
                relative_to=self.layout.run_dir,
            )
        self._emit("benchmark", result=value)
        return value

    def report_draft(self) -> Mapping[str, Any]:
        if self.dry_run:
            self._emit(
                "report-draft",
                dry_run=True,
                protected_test_labels="not accepted by report-draft",
            )
            return {"dry_run": True}
        handoff = self._require_adapt_handoff()
        checkpoints = self._handoff_paths(handoff, verify=False)
        from .reporting import create_draft_report

        benchmark_path = self.layout.benchmark_dir / "benchmark.json"
        result = create_draft_report(
            run_dir=self.layout.run_dir,
            run_fingerprint=self.fingerprint,
            config=self.config,
            prepared_manifest_path=self.layout.prepared_dir / "artifact-manifest.json",
            dataset_manifest_path=self.layout.prepared_dir / "dataset_manifest.json",
            data_audit_path=self.layout.prepared_dir / "data_audit.json",
            budget_selection_manifest_path=self.layout.budget_dir / "selection.manifest.json",
            budget_selection_path=self.layout.budget_dir / "selection.json",
            adaptation_selection_manifest_path=self.layout.adapt_dir / "selection.manifest.json",
            adaptation_selection_path=self.layout.adapt_dir / "selection.json",
            calibration_manifest_path=self.layout.adapt_dir / "calibration" / "artifact-manifest.json",
            ensemble_path=self.layout.adapt_dir / "calibration" / "ensemble.json",
            calibration_path=self.layout.adapt_dir / "calibration" / "calibration.json",
            checkpoint_manifest_paths=checkpoints,
            output_dir=self.layout.report_dir,
            benchmark_path=benchmark_path if benchmark_path.is_file() else None,
        )
        self._emit("report-draft", result=result, test_labels_accessed=False)
        return result

    def finalize_test(
        self,
        *,
        selection_lock: str | Path,
        official_test_path: str | Path,
    ) -> Mapping[str, Any]:
        if self.dry_run:
            self._emit("finalize-test", dry_run=True, explicit_one_shot=True)
            return {"dry_run": True}
        from .reporting import finalize_official_test

        expected_test_sha256 = str(self.config["data"]["czech"]["test"][0]["sha256"])
        result = finalize_official_test(
            run_dir=self.layout.run_dir,
            run_fingerprint=self.fingerprint,
            selection_lock_path=Path(selection_lock).resolve(),
            prepared_unlabeled_test_path=self.layout.prepared_dir / "czech" / "test.jsonl",
            official_gold_test_path=Path(official_test_path).resolve(),
            expected_gold_sha256=expected_test_sha256,
            output_dir=self.layout.final_dir,
            evaluator=lambda checkpoint_by_seed, official_test, calibration: self.backend.evaluate_official_test(
                checkpoint_by_seed=checkpoint_by_seed,
                official_test=official_test,
                calibration=calibration,
                config=self.config,
            ),
        )
        self._emit("final-test", result=result)
        return result


class V2ExecutionBackend:
    """Production backend implemented solely with package-qualified v2 modules."""

    @staticmethod
    def _training_config(config: Mapping[str, Any], tiny: bool) -> Any:
        from .modeling import TrainingConfig

        values = dict(config["training"])
        selected = {item.name: values[item.name] for item in fields(TrainingConfig)}
        if tiny:
            selected.update(min_epochs=1, max_epochs=1, patience=1)
        return TrainingConfig(**selected)

    @staticmethod
    def _lapt_config(config: Mapping[str, Any], tiny: bool) -> Any:
        from .adaptation import LAPTConfig

        values = dict(config["lapt"])
        values["max_length"] = int(config["training"]["max_length"])
        if tiny:
            values.update(max_updates=1, eval_every=1, holdout_sentences=1)
        return LAPTConfig(**values)

    @staticmethod
    def _transfer_config(config: Mapping[str, Any], tiny: bool) -> Any:
        from .adaptation import TransferConfig

        values = dict(config["transfer"])
        if tiny:
            values.update(updates=1, eval_every=1)
        return TransferConfig(**values)

    @staticmethod
    def _sentences(
        path: Path, *, mode: Literal["labeled", "forms-only"]
    ) -> tuple[Any, ...]:
        labeled = mode == "labeled"
        from .data import (
            LabeledToken,
            Sentence,
            SentenceProvenance,
            Token,
            TokenProvenance,
            ordered_form_hash,
        )

        sentences = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ArtifactIntegrityError(
                        f"invalid prepared JSONL at {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise ArtifactIntegrityError(f"prepared row must be an object: {path}")
                provenance_value = record.get("provenance")
                token_values = record.get("tokens")
                if not isinstance(provenance_value, dict) or not isinstance(token_values, list) or not token_values:
                    raise ArtifactIntegrityError(f"malformed prepared row: {path}:{line_number}")
                tokens = []
                for token_value in token_values:
                    if not isinstance(token_value, dict):
                        raise ArtifactIntegrityError(f"malformed prepared token: {path}:{line_number}")
                    has_label = "upos" in token_value
                    if labeled and not has_label:
                        raise ArtifactIntegrityError(f"labeled prepared row lacks UPOS: {path}:{line_number}")
                    if not labeled and has_label:
                        raise ArtifactIntegrityError(
                            f"FORM-only prepared row contains UPOS: {path}:{line_number}"
                        )
                    token_provenance = TokenProvenance(
                        conllu_id=int(token_value["id"]),
                        line_number=int(token_value["line_number"]),
                    )
                    if labeled:
                        token = LabeledToken(
                            form=str(token_value["form"]),
                            provenance=token_provenance,
                            upos=str(token_value["upos"]),
                        )
                    else:
                        token = Token(
                            form=str(token_value["form"]), provenance=token_provenance
                        )
                    tokens.append(token)
                sentence = Sentence(
                    tokens=tuple(tokens),
                    provenance=SentenceProvenance(
                        source_file=str(provenance_value["source_file"]),
                        split=str(provenance_value["split"]),
                        sentence_index=int(provenance_value["sentence_index"]),
                        sent_id=(
                            None
                            if provenance_value.get("sent_id") is None
                            else str(provenance_value["sent_id"])
                        ),
                    ),
                    comments=tuple(str(item) for item in record.get("comments", [])),
                )
                if record.get("form_hash") != ordered_form_hash(sentence.forms):
                    raise ArtifactIntegrityError(f"prepared FORM hash mismatch: {path}:{line_number}")
                sentences.append(sentence)
        if not sentences:
            raise ArtifactIntegrityError(f"prepared dataset is empty: {path}")
        return tuple(sentences)

    @staticmethod
    def _model_tokenizer(
        *,
        model_family: str,
        initialization_manifest: Path | None,
        config: Mapping[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        from .modeling import initialize_token_classifier
        from .tokenization import create_fast_tokenizer

        model_config = config["models"][model_family]
        if initialization_manifest is None:
            model = initialize_token_classifier(
                model_config["id"], revision=model_config["revision"]
            )
            tokenizer = create_fast_tokenizer(
                model_config["id"],
                revision=model_config["revision"],
                add_prefix_space=bool(model_config["add_prefix_space"]),
            )
            source = {
                "model_family": model_family,
                "model_id": model_config["id"],
                "model_revision": model_config["revision"],
                "add_prefix_space": bool(model_config["add_prefix_space"]),
            }
            return model, tokenizer, source
        checkpoint = checkpoint_load_dir(initialization_manifest)
        model = initialize_token_classifier(checkpoint)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
        if not bool(getattr(tokenizer, "is_fast", False)):
            raise RunnerError("checkpoint tokenizer is not fast")
        source = {
            "model_family": model_family,
            "initialization_checkpoint_sha256": sha256_file(initialization_manifest),
        }
        return model, tokenizer, source

    @staticmethod
    def _save_model(model: Any, tokenizer: Any, output_dir: Path) -> None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ResumeMismatchError(f"checkpoint output already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

    def train_supervised(
        self,
        *,
        job: Mapping[str, Any],
        train_path: Path,
        dev_path: Path,
        initialization_manifest: Path | None,
        output_dir: Path,
        config: Mapping[str, Any],
        tiny: bool,
    ) -> CheckpointExecution:
        from .modeling import set_seed, train_token_classifier
        from .tokenization import tokenize_sentences

        train_sentences = self._sentences(train_path, mode="labeled")
        dev_sentences = self._sentences(dev_path, mode="labeled")
        if tiny:
            train_sentences = train_sentences[:8]
            dev_sentences = dev_sentences[:8]
        # Seed before loading the classifier so newly initialized head weights are
        # deterministic, not only the subsequent optimizer and data ordering.
        set_seed(int(job["seed"]))
        model, tokenizer, source = self._model_tokenizer(
            model_family=str(job["model_family"]),
            initialization_manifest=initialization_manifest,
            config=config,
        )
        tokenization = config["training"]
        from .modeling import LABEL2ID

        train_data = tokenize_sentences(
            train_sentences,
            tokenizer,
            max_length=int(tokenization["max_length"]),
            stride=int(tokenization["stride"]),
            label2id=LABEL2ID,
            labeled=True,
        )
        dev_data = tokenize_sentences(
            dev_sentences,
            tokenizer,
            max_length=int(tokenization["max_length"]),
            stride=int(tokenization["stride"]),
            label2id=LABEL2ID,
            labeled=True,
        )
        training = train_token_classifier(
            model,
            train_data,
            dev_data,
            config=self._training_config(config, tiny),
            seed=int(job["seed"]),
        )
        self._save_model(training.model, tokenizer, output_dir)
        return CheckpointExecution(
            checkpoint_dir=output_dir,
            metrics=training.best_metrics,
            metadata={
                **source,
                "load_subdir": ".",
                "best_epoch": training.best_epoch,
                "best_step": training.best_step,
                "optimizer_steps": training.optimizer_steps,
                "stopped_early": training.stopped_early,
                "history": _jsonable(training.history),
                "train_sentences": len(train_sentences),
                "dev_sentences": len(dev_sentences),
                "canonical_upos": list(UPOS_TAGS),
            },
        )

    def train_lapt(
        self,
        *,
        train_path: Path,
        output_dir: Path,
        config: Mapping[str, Any],
        tiny: bool,
    ) -> CheckpointExecution:
        from .adaptation import initialize_masked_lm, project_czech_training_forms, train_lapt
        from .modeling import set_seed
        from .tokenization import create_fast_tokenizer

        sentences = self._sentences(train_path, mode="forms-only")
        if tiny:
            sentences = sentences[:8]
        projected = project_czech_training_forms(sentences)
        lapt_config = self._lapt_config(config, tiny)
        set_seed(lapt_config.seed)
        model_config = config["models"]["slovakbert"]
        tokenizer = create_fast_tokenizer(
            model_config["id"],
            revision=model_config["revision"],
            add_prefix_space=bool(model_config["add_prefix_space"]),
        )
        model = initialize_masked_lm(
            model_config["id"], revision=model_config["revision"]
        )
        result = train_lapt(
            model,
            tokenizer,
            projected,
            output_dir,
            config=lapt_config,
            model_source=model_config["id"],
            model_revision=model_config["revision"],
        )
        # Downstream transfer loads only the encoder. Remove the duplicate full
        # masked-LM serialization before the runner seals and hashes this checkpoint.
        if result.checkpoint_dir.resolve() != result.encoder_dir.resolve():
            shutil.rmtree(result.checkpoint_dir)
        return CheckpointExecution(
            checkpoint_dir=output_dir,
            metrics={
                "best_holdout_loss": result.best_holdout_loss,
                "best_step": result.best_step,
            },
            metadata={
                "model_family": "slovakbert",
                "model_id": model_config["id"],
                "model_revision": model_config["revision"],
                "add_prefix_space": bool(model_config["add_prefix_space"]),
                "load_subdir": "encoder",
                "optimizer_steps": result.optimizer_steps,
                "labels_accessed": False,
                "input_artifact": "czech/lapt_train_forms.jsonl",
                "history": _jsonable(result.history),
            },
        )

    def train_transfer(
        self,
        *,
        job: Mapping[str, Any],
        slovak_train_path: Path,
        slovak_dev_path: Path,
        initialization_manifest: Path | None,
        output_dir: Path,
        config: Mapping[str, Any],
        tiny: bool,
    ) -> CheckpointExecution:
        from .adaptation import initialize_transfer_classifier, train_slovak_transfer
        from .modeling import LABEL2ID, set_seed
        from .tokenization import create_fast_tokenizer, tokenize_sentences

        train_sentences = self._sentences(slovak_train_path, mode="labeled")
        dev_sentences = self._sentences(slovak_dev_path, mode="labeled")
        if tiny:
            train_sentences = train_sentences[:8]
            dev_sentences = dev_sentences[:8]
        transfer_config = self._transfer_config(config, tiny)
        set_seed(transfer_config.seed)
        model_config = config["models"]["slovakbert"]
        initialization = str(job["initialization"])
        if initialization_manifest is None:
            model = initialize_transfer_classifier(
                model_config["id"],
                initialization="base",
                revision=model_config["revision"],
            )
            tokenizer = create_fast_tokenizer(
                model_config["id"],
                revision=model_config["revision"],
                add_prefix_space=bool(model_config["add_prefix_space"]),
            )
            source_hash = None
        else:
            checkpoint = checkpoint_load_dir(initialization_manifest)
            model = initialize_transfer_classifier(
                checkpoint, initialization="lapt"
            )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
            source_hash = sha256_file(initialization_manifest)
        training = config["training"]
        train_data = tokenize_sentences(
            train_sentences,
            tokenizer,
            max_length=int(training["max_length"]),
            stride=int(training["stride"]),
            label2id=LABEL2ID,
            labeled=True,
        )
        dev_data = tokenize_sentences(
            dev_sentences,
            tokenizer,
            max_length=int(training["max_length"]),
            stride=int(training["stride"]),
            label2id=LABEL2ID,
            labeled=True,
        )
        result = train_slovak_transfer(
            model,
            train_data,
            dev_data,
            initialization=initialization,
            config=transfer_config,
        )
        self._save_model(result.model, tokenizer, output_dir)
        return CheckpointExecution(
            checkpoint_dir=output_dir,
            metrics=result.best_metrics,
            metadata={
                "model_family": "slovakbert",
                "model_id": model_config["id"],
                "model_revision": model_config["revision"],
                "add_prefix_space": bool(model_config["add_prefix_space"]),
                "load_subdir": ".",
                "initialization": initialization,
                "initialization_checkpoint_sha256": source_hash,
                "best_step": result.best_step,
                "optimizer_steps": result.optimizer_steps,
                "optimizer_state_preserved": False,
                "history": _jsonable(result.history),
                "canonical_upos": list(UPOS_TAGS),
            },
        )

    @staticmethod
    def _predict_members(
        checkpoint_manifests: Sequence[Path],
        sentences: Sequence[Any],
        config: Mapping[str, Any],
        *,
        labeled: bool,
    ) -> tuple[list[Any], Any | None]:
        from .modeling import LABEL2ID, initialize_token_classifier, predict_probabilities
        from .tokenization import tokenize_sentences
        from transformers import AutoTokenizer

        members = []
        gold = None
        for manifest_path in checkpoint_manifests:
            checkpoint = checkpoint_load_dir(manifest_path)
            model = initialize_token_classifier(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
            tokenized = tokenize_sentences(
                sentences,
                tokenizer,
                max_length=int(config["training"]["max_length"]),
                stride=int(config["training"]["stride"]),
                label2id=LABEL2ID if labeled else None,
                labeled=labeled,
            )
            if labeled:
                if gold is None:
                    gold = tokenized.gold_label_ids
                elif gold != tokenized.gold_label_ids:
                    raise RunnerError("ensemble tokenizers disagree on gold alignment")
            members.append(
                predict_probabilities(
                    model,
                    tokenized,
                    batch_size=int(config["training"]["batch_size"]),
                    labels=UPOS_TAGS,
                )
            )
        return members, gold

    def calibrate_ensemble(
        self,
        *,
        checkpoint_manifests: Sequence[Path],
        selection_path: Path,
        calibration_fit_path: Path,
        calibration_assessment_path: Path,
        config: Mapping[str, Any],
        tiny: bool,
    ) -> CalibrationExecution:
        from .calibration import assess_calibration, fit_calibration
        from .metrics import compute_metrics
        from .modeling import ensemble_probabilities, predictions_from_probabilities

        if len(checkpoint_manifests) != 3:
            raise RunnerError("the winning ensemble requires exactly three seed checkpoints")
        selection = self._sentences(selection_path, mode="labeled")
        fit = self._sentences(calibration_fit_path, mode="labeled")
        assessment = self._sentences(calibration_assessment_path, mode="labeled")
        if tiny:
            selection = selection[:8]
            fit = fit[:8]
            assessment = assessment[:8]
        selection_members, selection_gold = self._predict_members(
            checkpoint_manifests, selection, config, labeled=True
        )
        fit_members, fit_gold = self._predict_members(
            checkpoint_manifests, fit, config, labeled=True
        )
        assessment_members, assessment_gold = self._predict_members(
            checkpoint_manifests, assessment, config, labeled=True
        )
        assert selection_gold is not None and fit_gold is not None and assessment_gold is not None
        selection_ensemble = ensemble_probabilities(selection_members)
        selection_predictions = predictions_from_probabilities(selection_ensemble)
        gold_tags = tuple(
            tuple(UPOS_TAGS[label] for label in row) for row in selection_gold
        )
        ensemble_metrics = compute_metrics(gold_tags, selection_predictions.tags)
        fit_ensemble = ensemble_probabilities(fit_members)
        bundle = fit_calibration(
            fit_ensemble,
            fit_gold,
            ensemble_members=fit_members,
            regularization_c=float(config["calibration"]["logistic_c"]),
        )
        assessment_ensemble = ensemble_probabilities(assessment_members)
        assessment_result = assess_calibration(
            bundle,
            assessment_ensemble,
            assessment_gold,
            ensemble_members=assessment_members,
            fixed_failure_risks=config["calibration"]["fixed_failure_risks"],
        )
        return CalibrationExecution(
            calibration={
                "fit_partition": "czech/dev/calibration_fit",
                "assessment_partition": "czech/dev/calibration_assessment",
                "bundle": _jsonable(bundle),
                "assessment": _jsonable(assessment_result),
                "selection_influence": False,
            },
            ensemble={
                "method": "unweighted-arithmetic-mean-probabilities",
                "member_count": 3,
                "selection_partition": "czech/dev/selection",
                "selection_metrics": _jsonable(ensemble_metrics),
            },
        )

    def benchmark_ensemble(
        self,
        *,
        checkpoint_manifests: Sequence[Path],
        data_path: Path,
        config: Mapping[str, Any],
        tiny: bool,
    ) -> Mapping[str, Any]:
        sentences = self._sentences(data_path, mode="forms-only")
        if tiny:
            sentences = sentences[:8]
        repeats = 1 if tiny else 3
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            self._predict_members(checkpoint_manifests, sentences, config, labeled=False)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
            durations.append(time.perf_counter() - started)
        tokens = sum(len(sentence.tokens) for sentence in sentences)
        median = sorted(durations)[len(durations) // 2]
        return {
            "repeats": repeats,
            "durations_seconds": durations,
            "median_seconds": median,
            "sentences": len(sentences),
            "tokens": tokens,
            "sentences_per_second": len(sentences) / median,
            "tokens_per_second": tokens / median,
            "selection_influence": False,
        }

    @staticmethod
    def _calibration_bundle(value: Mapping[str, Any]) -> Any:
        from .calibration import CalibrationBundle, SentenceSuccessModel, TokenTemperature

        bundle = value.get("bundle")
        if not isinstance(bundle, Mapping):
            raise ArtifactIntegrityError("locked calibration lacks a bundle")
        token = bundle.get("token_temperature")
        sentence = bundle.get("sentence_model")
        if not isinstance(token, Mapping) or not isinstance(sentence, Mapping):
            raise ArtifactIntegrityError("locked calibration bundle is malformed")
        return CalibrationBundle(
            token_temperature=TokenTemperature(**dict(token)),
            sentence_model=SentenceSuccessModel(
                feature_names=tuple(sentence["feature_names"]),
                feature_means=tuple(sentence["feature_means"]),
                feature_scales=tuple(sentence["feature_scales"]),
                coefficients=tuple(sentence["coefficients"]),
                intercept=float(sentence["intercept"]),
                regularization_c=float(sentence["regularization_c"]),
            ),
            low_confidence_threshold=float(bundle["low_confidence_threshold"]),
        )

    def evaluate_official_test(
        self,
        *,
        checkpoint_by_seed: Mapping[int, Path],
        official_test: Any,
        calibration: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from .calibration import assess_calibration
        from .metrics import compute_metrics
        from .modeling import (
            LABEL2ID,
            ensemble_probabilities,
            initialize_token_classifier,
            predict_probabilities,
            predictions_from_probabilities,
        )
        from .tokenization import tokenize_pretokenized
        from transformers import AutoTokenizer

        forms = official_test.forms
        gold_tags = official_test.gold_tags
        gold_ids = tuple(tuple(LABEL2ID[tag] for tag in row) for row in gold_tags)
        members = []
        seed_metrics = []
        for seed, manifest_path in checkpoint_by_seed.items():
            checkpoint = checkpoint_load_dir(manifest_path)
            model = initialize_token_classifier(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
            tokenized = tokenize_pretokenized(
                forms,
                tokenizer,
                max_length=int(config["training"]["max_length"]),
                stride=int(config["training"]["stride"]),
                sentence_ids=tuple(range(len(forms))),
            )
            probabilities = predict_probabilities(
                model,
                tokenized,
                batch_size=int(config["training"]["batch_size"]),
                labels=UPOS_TAGS,
            )
            members.append(probabilities)
            seed_prediction = predictions_from_probabilities(probabilities)
            seed_metrics.append(
                {
                    "seed": int(seed),
                    "tagging": _jsonable(
                        compute_metrics(gold_tags, seed_prediction.tags)
                    ),
                }
            )
        ensemble = ensemble_probabilities(members)
        prediction = predictions_from_probabilities(ensemble)
        tagging = compute_metrics(gold_tags, prediction.tags)
        bundle = self._calibration_bundle(calibration)
        calibrated = assess_calibration(
            bundle,
            ensemble,
            gold_ids,
            ensemble_members=members,
            fixed_failure_risks=config["calibration"]["fixed_failure_risks"],
        )
        return {
            "seed_metrics": seed_metrics,
            "ensemble": {
                "tagging": _jsonable(tagging),
                "calibration": _jsonable(calibrated),
            },
        }
