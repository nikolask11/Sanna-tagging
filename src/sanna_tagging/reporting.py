"""Draft reporting, immutable selection locking, and one-shot test finalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import (
    ArtifactIntegrityError,
    ResumeMismatchError,
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    file_record,
    load_manifest,
    sha256_file,
    sha256_json,
    strict_resume,
    verify_file_record,
    write_manifest,
)
from .config import UPOS_TAGS
from .data import ordered_form_hash

REPORTING_SCHEMA_VERSION = 1


class ReportingError(RuntimeError):
    """A report input violates the frozen-selection reporting contract."""


class FinalizationAlreadyAttempted(ReportingError):
    """Official test evaluation is one-shot and an attempt marker already exists."""


class FinalizationMismatch(ReportingError):
    """The official test, lock, fingerprint, or checkpoint set does not match."""


_FINAL_TEST_SEAL = object()


@dataclass(frozen=True, slots=True)
class _OfficialTestGold:
    """Gold labels that can only be constructed inside finalization."""

    forms: tuple[tuple[str, ...], ...]
    gold_tags: tuple[tuple[str, ...], ...]
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _FINAL_TEST_SEAL:
            raise TypeError("official test gold is restricted to finalization")
        if len(self.forms) != len(self.gold_tags) or not self.forms:
            raise ValueError("official test FORM and UPOS rows must align")
        if any(len(forms) != len(tags) or not forms for forms, tags in zip(self.forms, self.gold_tags)):
            raise ValueError("official test tokens must align within every sentence")


FinalEvaluator = Callable[
    [Mapping[int, Path], _OfficialTestGold, Mapping[str, Any]], Mapping[str, Any]
]


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
    except ValueError as error:
        raise ReportingError(f"report input must live inside the portable run: {path}") from error


def _resolve_portable(value: str, run_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise FinalizationMismatch(f"selection lock contains an absolute path: {value}")
    resolved = (run_dir / path).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as error:
        raise FinalizationMismatch(f"selection lock path escapes the run: {value}") from error
    return resolved


def _checkpoint_lock_entries(
    checkpoint_manifest_paths: Sequence[Path],
    *,
    run_dir: Path,
    run_fingerprint: str,
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    from .runner import verify_checkpoint_manifest

    if len(expected_seeds) != 3:
        raise ReportingError("the v2 protocol requires exactly three configured seeds")
    if len(checkpoint_manifest_paths) != len(expected_seeds):
        raise ReportingError("selection lock requires one checkpoint per configured seed")
    entries = []
    seen_seeds = set()
    for path in checkpoint_manifest_paths:
        path = Path(path).resolve()
        checkpoint = verify_checkpoint_manifest(
            path, expected_fingerprint=run_fingerprint
        )
        metadata = checkpoint.get("metadata")
        job = metadata.get("job") if isinstance(metadata, dict) else None
        seed = job.get("seed") if isinstance(job, dict) else None
        job_stage = job.get("stage") if isinstance(job, dict) else None
        artifact_manifest_path = path.parent / "artifact-manifest.json"
        if not artifact_manifest_path.is_file() or not isinstance(job_stage, str):
            raise ReportingError("winning checkpoint lacks its committed job manifest")
        artifact_manifest = load_manifest(artifact_manifest_path)
        if (
            artifact_manifest.get("stage") != job_stage
            or artifact_manifest.get("run_fingerprint") != run_fingerprint
        ):
            raise ReportingError("winning checkpoint job manifest does not match the run")
        job_outputs = artifact_manifest.get("outputs")
        checkpoint_record = (
            job_outputs.get("checkpoint_manifest")
            if isinstance(job_outputs, dict)
            else None
        )
        if not isinstance(checkpoint_record, dict):
            raise ReportingError("job manifest does not commit its checkpoint manifest")
        committed_checkpoint = verify_file_record(
            checkpoint_record, relative_to=run_dir
        )
        if committed_checkpoint.resolve() != path:
            raise ReportingError("winning checkpoint differs from its job commit")
        if not isinstance(seed, int) or seed in seen_seeds:
            raise ReportingError("winning checkpoints must identify three unique integer seeds")
        seen_seeds.add(seed)
        entries.append(
            {
                "seed": seed,
                "checkpoint_manifest": file_record(path, relative_to=run_dir),
                "checkpoint_sha256": sha256_file(path),
                "checkpoint_files": checkpoint["files"],
                "metadata": metadata,
            }
        )
    entries.sort(key=lambda item: item["seed"])
    if [item["seed"] for item in entries] != sorted(int(seed) for seed in expected_seeds):
        raise ReportingError("checkpoint seeds do not match the configured three seeds")
    return entries


def _draft_markdown(report: Mapping[str, Any]) -> str:
    budget = report["budget_selection"]["selected_budget"]
    arm = report["adaptation_selection"]["winning_arm"]
    ensemble_metrics = report["ensemble"].get("selection_metrics", {})
    token_accuracy = ensemble_metrics.get("token_accuracy", "n/a")
    sentence_pass = ensemble_metrics.get("sentence_at_least_98_rate", "n/a")
    return "\n".join(
        [
            "# Czech UPOS v2 draft report",
            "",
            f"- Run fingerprint: `{report['run_fingerprint']}`",
            f"- Selected Czech gold budget: {budget}",
            f"- Winning arm: `{arm}`",
            "- Ensemble: unweighted arithmetic mean of three frozen seed checkpoints",
            f"- Selection token accuracy: {token_accuracy}",
            f"- Selection sentence >=98% rate: {sentence_pass}",
            "- Official test status: not evaluated",
            "",
            "This draft is frozen before official test labels are available to reporting. "
            "The immutable `selection.lock.json` fixes the winning arm, all three "
            "checkpoint hashes, ensemble rule, and calibration artifact.",
            "",
        ]
    )


def create_draft_report(
    *,
    run_dir: str | Path,
    run_fingerprint: str,
    config: Mapping[str, Any],
    prepared_manifest_path: str | Path,
    dataset_manifest_path: str | Path,
    data_audit_path: str | Path,
    budget_selection_manifest_path: str | Path,
    budget_selection_path: str | Path,
    adaptation_selection_manifest_path: str | Path,
    adaptation_selection_path: str | Path,
    calibration_manifest_path: str | Path,
    ensemble_path: str | Path,
    calibration_path: str | Path,
    checkpoint_manifest_paths: Sequence[str | Path],
    output_dir: str | Path,
    benchmark_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the draft and freeze selection without accepting any test-gold input.

    The intentionally narrow signature is the leakage boundary: callers can pass
    prepared metadata, development-set results, checkpoints, and calibration only.
    There is no raw corpus path, parser callback, or labeled test object here.
    """
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    named_paths = {
        "prepared_manifest": Path(prepared_manifest_path).resolve(),
        "dataset_manifest": Path(dataset_manifest_path).resolve(),
        "data_audit": Path(data_audit_path).resolve(),
        "budget_selection_manifest": Path(budget_selection_manifest_path).resolve(),
        "budget_selection": Path(budget_selection_path).resolve(),
        "adaptation_selection_manifest": Path(adaptation_selection_manifest_path).resolve(),
        "adaptation_selection": Path(adaptation_selection_path).resolve(),
        "calibration_manifest": Path(calibration_manifest_path).resolve(),
        "ensemble": Path(ensemble_path).resolve(),
        "calibration": Path(calibration_path).resolve(),
    }
    for name, path in named_paths.items():
        if not path.is_file():
            raise ReportingError(f"missing {name} artifact: {path}")
        _portable(path, run_dir)
    if benchmark_path is not None:
        benchmark = Path(benchmark_path).resolve()
        if not benchmark.is_file():
            raise ReportingError(f"benchmark artifact is missing: {benchmark}")
        _portable(benchmark, run_dir)
        named_paths["benchmark"] = benchmark

    manifest_names = {
        "prepared_manifest",
        "budget_selection_manifest",
        "adaptation_selection_manifest",
        "calibration_manifest",
    }
    values = {
        name: (load_manifest(path) if name in manifest_names else _read_json(path))
        for name, path in named_paths.items()
    }
    prepared_manifest = values["prepared_manifest"]
    if (
        prepared_manifest.get("stage") != "data-preparation"
        or prepared_manifest.get("run_fingerprint") != run_fingerprint
    ):
        raise ReportingError("prepared manifest does not match this run")
    prepared_outputs = prepared_manifest.get("outputs")
    if not isinstance(prepared_outputs, dict):
        raise ArtifactIntegrityError("prepared manifest has malformed outputs")
    expected_prepared = {
        "dataset_manifest": named_paths["dataset_manifest"],
        "data_audit": named_paths["data_audit"],
    }
    for output_name, expected_path in expected_prepared.items():
        record = prepared_outputs.get(output_name)
        if not isinstance(record, dict):
            raise ReportingError(
                f"prepared manifest does not commit {output_name}"
            )
        committed_path = verify_file_record(
            record, relative_to=named_paths["prepared_manifest"].parent
        )
        if committed_path.resolve() != expected_path:
            raise ReportingError(
                f"{output_name} is not the artifact committed by preparation"
            )
    stage_commitments = (
        (
            "budget_selection_manifest",
            "budget-selection",
            {"selection": named_paths["budget_selection"]},
        ),
        (
            "adaptation_selection_manifest",
            "adaptation-selection",
            {"selection": named_paths["adaptation_selection"]},
        ),
        (
            "calibration_manifest",
            "ensemble-calibration",
            {
                "ensemble": named_paths["ensemble"],
                "calibration": named_paths["calibration"],
            },
        ),
    )
    for manifest_name, expected_stage, expected_outputs in stage_commitments:
        manifest = values[manifest_name]
        if (
            manifest.get("stage") != expected_stage
            or manifest.get("run_fingerprint") != run_fingerprint
        ):
            raise ReportingError(f"{manifest_name} does not match this run")
        records = manifest.get("outputs")
        if not isinstance(records, dict):
            raise ArtifactIntegrityError(f"{manifest_name} has malformed outputs")
        for output_name, expected_path in expected_outputs.items():
            record = records.get(output_name)
            if not isinstance(record, dict):
                raise ReportingError(
                    f"{manifest_name} does not commit {output_name}"
                )
            committed_path = verify_file_record(record, relative_to=run_dir)
            if committed_path.resolve() != expected_path:
                raise ReportingError(
                    f"{output_name} is not committed by {manifest_name}"
                )
    for name, value in values.items():
        fingerprint = value.get("run_fingerprint")
        if fingerprint is not None and fingerprint != run_fingerprint:
            raise ResumeMismatchError(f"{name} fingerprint does not match the run")
    dataset = values["dataset_manifest"]
    audit = values["data_audit"]
    if dataset.get("test_labels_exposed") is not False or audit.get("test_labels_exposed") is not False:
        raise ReportingError("prepared data does not attest that test labels stayed hidden")

    budget_selection = values["budget_selection"]
    adaptation_selection = values["adaptation_selection"]
    ensemble = values["ensemble"]
    calibration = values["calibration"]
    winning_arm = adaptation_selection.get("winning_arm")
    if winning_arm != ensemble.get("winning_arm") or winning_arm != calibration.get("winning_arm"):
        raise ReportingError("selection, ensemble, and calibration disagree on the winning arm")
    selected_budget = adaptation_selection.get("selected_budget")
    if selected_budget != budget_selection.get("selected_budget"):
        raise ReportingError("budget and adaptation artifacts disagree on selected budget")

    expected_seeds = [int(seed) for seed in config["random_seeds"]]
    checkpoint_entries = _checkpoint_lock_entries(
        [Path(path) for path in checkpoint_manifest_paths],
        run_dir=run_dir,
        run_fingerprint=run_fingerprint,
        expected_seeds=expected_seeds,
    )
    checkpoint_hashes = [entry["checkpoint_sha256"] for entry in checkpoint_entries]
    ensemble_record = file_record(named_paths["ensemble"], relative_to=run_dir)
    calibration_record = file_record(named_paths["calibration"], relative_to=run_dir)
    selection_lock = {
        "schema_version": REPORTING_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "selection_frozen": True,
        "selected_budget": selected_budget,
        "winning_arm": winning_arm,
        "seeds": expected_seeds,
        "canonical_upos": list(UPOS_TAGS),
        "checkpoints": checkpoint_entries,
        "checkpoint_hashes": checkpoint_hashes,
        "ensemble": {
            "method": ensemble.get("method"),
            "member_count": ensemble.get("member_count"),
            "artifact": ensemble_record,
            "stage_manifest": file_record(
                named_paths["calibration_manifest"], relative_to=run_dir
            ),
            "value": ensemble,
        },
        "calibration": {
            "fit_partition": calibration.get("fit_partition"),
            "assessment_partition": calibration.get("assessment_partition"),
            "artifact": calibration_record,
            "stage_manifest": file_record(
                named_paths["calibration_manifest"], relative_to=run_dir
            ),
            "value": calibration,
        },
        "data": {
            "prepared_manifest": file_record(
                named_paths["prepared_manifest"], relative_to=run_dir
            ),
            "dataset_manifest": file_record(
                named_paths["dataset_manifest"], relative_to=run_dir
            ),
            "data_audit": file_record(named_paths["data_audit"], relative_to=run_dir),
            "test_labels_exposed": False,
        },
        "selection_evidence": {
            "budget": file_record(
                named_paths["budget_selection"], relative_to=run_dir
            ),
            "budget_manifest": file_record(
                named_paths["budget_selection_manifest"], relative_to=run_dir
            ),
            "adaptation": file_record(
                named_paths["adaptation_selection"], relative_to=run_dir
            ),
            "adaptation_manifest": file_record(
                named_paths["adaptation_selection_manifest"], relative_to=run_dir
            ),
        },
        "final_test_evaluated": False,
    }
    if selection_lock["ensemble"]["method"] != "unweighted-arithmetic-mean-probabilities":
        raise ReportingError("only the predeclared unweighted ensemble can be locked")
    if selection_lock["ensemble"]["member_count"] != len(expected_seeds):
        raise ReportingError("the locked ensemble must contain one member per seed")

    report = {
        "schema_version": REPORTING_SCHEMA_VERSION,
        "status": "draft-selection-frozen-test-not-evaluated",
        "run_fingerprint": run_fingerprint,
        "test_labels_accessed": False,
        "budget_selection": budget_selection,
        "adaptation_selection": adaptation_selection,
        "ensemble": ensemble,
        "calibration": calibration,
        "data_audit": audit,
        "benchmark": values.get("benchmark"),
        "selection_lock_sha256": sha256_json(selection_lock),
    }
    report_json_path = output_dir / "report.draft.json"
    report_markdown_path = output_dir / "report.draft.md"
    selection_lock_path = output_dir / "selection.lock.json"
    manifest_path = output_dir / "artifact-manifest.json"
    outputs = {
        "draft_json": report_json_path,
        "draft_markdown": report_markdown_path,
        "selection_lock": selection_lock_path,
    }
    parameters = {
        "schema_version": REPORTING_SCHEMA_VERSION,
        "winning_arm": winning_arm,
        "selected_budget": selected_budget,
        "checkpoint_hashes": checkpoint_hashes,
        "test_labels_accessed": False,
    }
    if strict_resume(
        manifest_path,
        stage="report-draft",
        run_fingerprint=run_fingerprint,
        inputs=named_paths,
        outputs=outputs,
        parameters=parameters,
        relative_to=run_dir,
    ):
        existing_lock = _read_json(selection_lock_path)
        if canonical_json_bytes(existing_lock) != canonical_json_bytes(selection_lock):
            raise ResumeMismatchError("existing selection lock differs from recomputed lock")
        return _read_json(report_json_path)

    atomic_write_json(selection_lock_path, selection_lock)
    atomic_write_json(report_json_path, report)
    atomic_write_text(report_markdown_path, _draft_markdown(report))
    write_manifest(
        manifest_path,
        stage="report-draft",
        run_fingerprint=run_fingerprint,
        inputs=named_paths,
        outputs=outputs,
        parameters=parameters,
        relative_to=run_dir,
    )
    return report


def _read_prepared_unlabeled_test(path: Path) -> tuple[tuple[str, ...], ...]:
    forms = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise FinalizationMismatch(
                    f"invalid prepared test JSONL at line {line_number}"
                ) from error
            tokens = record.get("tokens") if isinstance(record, dict) else None
            if not isinstance(tokens, list) or not tokens:
                raise FinalizationMismatch("prepared test contains a malformed sentence")
            if any(not isinstance(token, dict) or "upos" in token for token in tokens):
                raise FinalizationMismatch("prepared test must be strictly unlabeled")
            row = tuple(str(token["form"]) for token in tokens)
            if record.get("form_hash") != ordered_form_hash(row):
                raise FinalizationMismatch("prepared test FORM hash mismatch")
            forms.append(row)
    if not forms:
        raise FinalizationMismatch("prepared test is empty")
    return tuple(forms)


def _parse_official_test_gold(path: Path) -> _OfficialTestGold:
    """Narrow gold parser called only after the one-shot attempt marker is sealed."""
    sentence_forms: list[tuple[str, ...]] = []
    sentence_tags: list[tuple[str, ...]] = []
    forms: list[str] = []
    tags: list[str] = []

    def finish() -> None:
        nonlocal forms, tags
        if forms:
            sentence_forms.append(tuple(forms))
            sentence_tags.append(tuple(tags))
        forms = []
        tags = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                finish()
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                raise FinalizationMismatch(
                    f"official test line {line_number} does not have ten columns"
                )
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue
            try:
                numeric_id = int(token_id)
            except ValueError as error:
                raise FinalizationMismatch(
                    f"official test line {line_number} has an invalid token ID"
                ) from error
            if numeric_id <= 0:
                raise FinalizationMismatch("official test token IDs must be positive")
            tag = fields[3]
            if tag not in UPOS_TAGS:
                raise FinalizationMismatch(
                    f"official test line {line_number} has invalid UPOS {tag!r}"
                )
            forms.append(fields[1])
            tags.append(tag)
        finish()
    return _OfficialTestGold(tuple(sentence_forms), tuple(sentence_tags), _FINAL_TEST_SEAL)


def _verify_locked_checkpoint_entries(
    lock: Mapping[str, Any], *, run_dir: Path, run_fingerprint: str
) -> dict[int, Path]:
    from .runner import verify_checkpoint_manifest

    entries = lock.get("checkpoints")
    hashes = lock.get("checkpoint_hashes")
    locked_seeds = lock.get("seeds")
    if not isinstance(locked_seeds, list) or len(locked_seeds) != 3:
        raise FinalizationMismatch("selection lock must contain the three configured seeds")
    expected_count = len(locked_seeds)
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise FinalizationMismatch("selection lock must contain one checkpoint per seed")
    if not isinstance(hashes, list) or len(hashes) != expected_count:
        raise FinalizationMismatch("selection lock must contain one hash per checkpoint")
    paths = []
    actual_hashes = []
    seeds = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise FinalizationMismatch("selection lock checkpoint entry is malformed")
        record = entry.get("checkpoint_manifest")
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise FinalizationMismatch("selection lock checkpoint record is malformed")
        path = _resolve_portable(record["path"], run_dir)
        verify_file_record(record, relative_to=run_dir)
        digest = sha256_file(path)
        if digest != entry.get("checkpoint_sha256"):
            raise FinalizationMismatch("locked checkpoint manifest hash changed")
        verify_checkpoint_manifest(path, expected_fingerprint=run_fingerprint)
        paths.append(path)
        actual_hashes.append(digest)
        seeds.append(entry.get("seed"))
    if actual_hashes != hashes:
        raise FinalizationMismatch("checkpoint hashes do not match the frozen hash list")
    if seeds != locked_seeds or len(set(seeds)) != expected_count:
        raise FinalizationMismatch("checkpoint seeds do not match the frozen seed list")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise FinalizationMismatch("checkpoint seeds must be integers")
    return dict(zip(seeds, paths, strict=True))


def _verify_locked_artifact(section: Any, *, run_dir: Path, name: str) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise FinalizationMismatch(f"selection lock lacks frozen {name}")
    record = section.get("artifact")
    frozen = section.get("value")
    if not isinstance(record, dict) or not isinstance(frozen, dict):
        raise FinalizationMismatch(f"selection lock has malformed frozen {name}")
    path = verify_file_record(record, relative_to=run_dir)
    current = _read_json(path)
    if canonical_json_bytes(current) != canonical_json_bytes(frozen):
        raise FinalizationMismatch(f"frozen {name} content changed")
    return frozen


def _verify_report_lock_provenance(
    selection_lock_path: Path, *, run_dir: Path, run_fingerprint: str
) -> None:
    expected_lock = (run_dir / "report" / "selection.lock.json").resolve()
    if selection_lock_path != expected_lock:
        raise FinalizationMismatch(
            "finalization requires the exact lock committed by this run's report-draft"
        )
    manifest_path = run_dir / "report" / "artifact-manifest.json"
    if not manifest_path.is_file():
        raise FinalizationMismatch("report-draft commit manifest is missing")
    manifest = load_manifest(manifest_path)
    if (
        manifest.get("stage") != "report-draft"
        or manifest.get("run_fingerprint") != run_fingerprint
    ):
        raise FinalizationMismatch("report-draft manifest does not match this run")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise FinalizationMismatch("report-draft manifest has malformed outputs")
    for output_name, record in outputs.items():
        if not isinstance(record, dict):
            raise FinalizationMismatch(
                f"report-draft output record is malformed: {output_name}"
            )
        verify_file_record(record, relative_to=run_dir)
    lock_record = outputs.get("selection_lock")
    if not isinstance(lock_record, dict):
        raise FinalizationMismatch("report-draft manifest does not commit a selection lock")
    committed_lock = verify_file_record(lock_record, relative_to=run_dir)
    if committed_lock.resolve() != selection_lock_path:
        raise FinalizationMismatch("supplied lock differs from the report-draft lock")
    draft_record = outputs.get("draft_json")
    if not isinstance(draft_record, dict):
        raise FinalizationMismatch("report-draft manifest does not commit its JSON report")
    draft = _read_json(verify_file_record(draft_record, relative_to=run_dir))
    if draft.get("selection_lock_sha256") != sha256_file(selection_lock_path):
        raise FinalizationMismatch("draft report and selection lock hashes disagree")


def _validate_test_metrics(
    value: Mapping[str, Any], *, locked_seeds: Sequence[int]
) -> dict[str, Any]:
    if set(value) != {"seed_metrics", "ensemble"}:
        raise ReportingError(
            "final evaluator must return only seed_metrics and ensemble metrics"
        )
    seed_values = value.get("seed_metrics")
    ensemble = value.get("ensemble")
    if not isinstance(seed_values, list) or not isinstance(ensemble, Mapping):
        raise ReportingError("final evaluator metrics schema is malformed")
    if set(ensemble) != {"tagging", "calibration"}:
        raise ReportingError("ensemble metrics must contain tagging and calibration only")
    if not isinstance(ensemble["tagging"], Mapping) or not isinstance(
        ensemble["calibration"], Mapping
    ):
        raise ReportingError("ensemble metric sections must be mappings")
    normalized_seeds = []
    result_seeds = []
    required_rates = (
        "token_accuracy",
        "sentence_at_least_98_rate",
        "exact_match_rate",
    )
    for item in seed_values:
        if not isinstance(item, Mapping) or set(item) != {"seed", "tagging"}:
            raise ReportingError("each seed metric must contain seed and tagging only")
        seed = item["seed"]
        tagging = item["tagging"]
        if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(tagging, Mapping):
            raise ReportingError("seed metric identity or tagging metrics are malformed")
        result_seeds.append(seed)
        normalized_seeds.append({"seed": seed, "tagging": dict(tagging)})
    if result_seeds != list(locked_seeds) or len(set(result_seeds)) != len(locked_seeds):
        raise ReportingError("final evaluator seed results must exactly match frozen seeds")
    for name, tagging in [
        *[(f"seed {item['seed']}", item["tagging"]) for item in normalized_seeds],
        ("ensemble", ensemble["tagging"]),
    ]:
        for metric in required_rates:
            metric_value = tagging.get(metric)
            if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool):
                raise ReportingError(f"{name} tagging metrics lack numeric {metric}")
    return {
        "seed_metrics": normalized_seeds,
        "ensemble": {
            "tagging": dict(ensemble["tagging"]),
            "calibration": dict(ensemble["calibration"]),
        },
    }


def _final_markdown(lock: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Final Official-Test Report",
        "",
        f"- Selected budget: {lock['selected_budget']}",
        f"- Winning arm: {lock['winning_arm']}",
        f"- Ensemble: {lock['ensemble']['method']}",
        "",
        "## Frozen checkpoints",
        "",
    ]
    for entry in lock["checkpoints"]:
        lines.append(
            f"- Seed {entry['seed']}: `{entry['checkpoint_sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Test metrics",
            "",
            "| System | Token accuracy | Sentence ≥98% | Exact match |",
            "|---|---:|---:|---:|",
        ]
    )
    rows = [
        *[
            (f"Seed {item['seed']}", item["tagging"])
            for item in metrics["seed_metrics"]
        ],
        ("Frozen 3-seed ensemble", metrics["ensemble"]["tagging"]),
    ]
    for name, tagging in rows:
        lines.append(
            f"| {name} | {float(tagging['token_accuracy']):.6f} | "
            f"{float(tagging['sentence_at_least_98_rate']):.6f} | "
            f"{float(tagging['exact_match_rate']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Ensemble calibration",
            "",
            "```json",
            json.dumps(
                metrics["ensemble"]["calibration"],
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            ),
            "```",
            "",
            "Selection, checkpoints, ensemble membership and weights, and calibration "
            "were not changed after selection.lock.json was frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_official_test(
    *,
    run_dir: str | Path,
    run_fingerprint: str,
    selection_lock_path: str | Path,
    prepared_unlabeled_test_path: str | Path,
    official_gold_test_path: str | Path,
    expected_gold_sha256: str,
    output_dir: str | Path,
    evaluator: FinalEvaluator,
) -> dict[str, Any]:
    """Evaluate the official test exactly once without reopening selection.

    Any existing finalization artifact rejects the call. The attempt marker is
    atomically written before the raw gold file is opened; a failed attempt is
    deliberately not resumable, preventing accidental repeated test evaluation.
    """
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    selection_lock_path = Path(selection_lock_path).resolve()
    prepared_path = Path(prepared_unlabeled_test_path).resolve()
    official_path = Path(official_gold_test_path).resolve()
    if output_dir.exists():
        raise FinalizationAlreadyAttempted(
            f"official test finalization was already attempted: {output_dir}"
        )
    if not selection_lock_path.is_file():
        raise FinalizationMismatch(f"selection lock is missing: {selection_lock_path}")
    if not prepared_path.is_file():
        raise FinalizationMismatch(f"prepared unlabeled test is missing: {prepared_path}")
    if not official_path.is_file():
        raise FinalizationMismatch(f"official gold test is missing: {official_path}")
    _verify_report_lock_provenance(
        selection_lock_path,
        run_dir=run_dir,
        run_fingerprint=run_fingerprint,
    )
    lock = _read_json(selection_lock_path)
    if lock.get("run_fingerprint") != run_fingerprint:
        raise FinalizationMismatch("selection lock fingerprint does not match the run")
    if lock.get("selection_frozen") is not True or lock.get("final_test_evaluated") is not False:
        raise FinalizationMismatch("selection lock is not an unused frozen lock")
    if lock.get("canonical_upos") != list(UPOS_TAGS):
        raise FinalizationMismatch("selection lock UPOS map is not canonical")
    checkpoint_by_seed = _verify_locked_checkpoint_entries(
        lock, run_dir=run_dir, run_fingerprint=run_fingerprint
    )
    ensemble = _verify_locked_artifact(lock.get("ensemble"), run_dir=run_dir, name="ensemble")
    calibration = _verify_locked_artifact(
        lock.get("calibration"), run_dir=run_dir, name="calibration"
    )
    if ensemble.get("winning_arm") != lock.get("winning_arm"):
        raise FinalizationMismatch("locked ensemble does not match the winning arm")
    prepared_forms = _read_prepared_unlabeled_test(prepared_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise FinalizationAlreadyAttempted(
            f"official test finalization was already claimed: {output_dir}"
        ) from error
    attempt_path = output_dir / "finalization.started.json"
    lock_sha256 = sha256_file(selection_lock_path)
    atomic_write_json(
        attempt_path,
        {
            "schema_version": REPORTING_SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "selection_lock_sha256": lock_sha256,
            "state": "official-test-access-committed-one-shot",
        },
    )

    actual_gold_sha256 = sha256_file(official_path)
    if actual_gold_sha256 != expected_gold_sha256:
        raise FinalizationMismatch(
            "official gold test SHA-256 does not match the pinned configuration"
        )
    official_test = _parse_official_test_gold(official_path)
    if official_test.forms != prepared_forms:
        raise FinalizationMismatch(
            "official gold test FORM values/order differ from prepared unlabeled test"
        )

    raw_metrics = evaluator(checkpoint_by_seed, official_test, calibration)
    if not isinstance(raw_metrics, Mapping):
        raise ReportingError("final evaluator must return a metrics mapping")
    metrics_value = _validate_test_metrics(
        raw_metrics, locked_seeds=list(checkpoint_by_seed)
    )
    metrics_path = output_dir / "test_metrics.json"
    report_json_path = output_dir / "report.final.json"
    report_markdown_path = output_dir / "REPORT_FINAL.md"
    manifest_path = output_dir / "artifact-manifest.json"
    final_report = {
        "schema_version": REPORTING_SCHEMA_VERSION,
        "status": "final-official-test-evaluated-once",
        "run_fingerprint": run_fingerprint,
        "selection_lock_sha256": lock_sha256,
        "selection": {
            "selected_budget": lock["selected_budget"],
            "winning_arm": lock["winning_arm"],
            "seeds": lock["seeds"],
            "checkpoint_hashes": lock["checkpoint_hashes"],
            "ensemble_method": lock["ensemble"]["method"],
        },
        "official_test": metrics_value,
        "selection_changed_after_lock": False,
    }
    markdown = _final_markdown(lock, metrics_value)
    atomic_write_json(metrics_path, metrics_value)
    atomic_write_json(report_json_path, final_report)
    atomic_write_text(report_markdown_path, markdown)
    write_manifest(
        manifest_path,
        stage="finalize-test-one-shot",
        run_fingerprint=run_fingerprint,
        inputs={
            "selection_lock": selection_lock_path,
            "prepared_unlabeled_test": prepared_path,
            "official_gold_test": official_path,
        },
        outputs={
            "attempt_marker": attempt_path,
            "test_metrics": metrics_path,
            "final_report_json": report_json_path,
            "final_report_markdown": report_markdown_path,
        },
        parameters={
            "selection_lock_sha256": lock_sha256,
            "winning_arm": lock["winning_arm"],
            "checkpoint_hashes": lock["checkpoint_hashes"],
            "ensemble_method": lock["ensemble"]["method"],
            "selection_changed_after_lock": False,
            "one_shot": True,
        },
        relative_to=run_dir,
    )
    return final_report
