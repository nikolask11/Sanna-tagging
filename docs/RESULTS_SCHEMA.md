# Czech v2 results schema

All v2 JSON is UTF-8, canonicalized with sorted keys and compact separators, and terminated by
a newline. JSON artifacts reject non-finite floats. Unless stated otherwise, schema versions
are currently `1`.

The run root is:

```text
<runs-root>/<run-name>/<run-fingerprint>/
```

For the accepted configuration, `<run-name>` is `cs-rerun-v2` and `<run-fingerprint>` is the
first 20 hexadecimal characters of SHA-256 over:

```json
{
  "config": "canonical YAML mapping without private loader keys",
  "code_commit": "exact Git commit",
  "dependency_lock_sha256": "SHA-256 of requirements-kaggle.lock"
}
```

Tiny/non-production execution adds an execution-mode value to the fingerprint input and marks
job results `experimental_valid: false`.

## Common file record

Manifests and locks identify files with:

```json
{
  "path": "portable/path/relative/to/the-record-base",
  "bytes": 123,
  "sha256": "64 lowercase hexadecimal characters"
}
```

A record is valid only when the resolved path is a regular file and both byte length and
SHA-256 match. Report locks require paths inside the run and reject absolute/escaping paths.

## Artifact commit manifest

`artifact-manifest.json`, `*.manifest.json`, and handoff manifests use:

```json
{
  "schema_version": 1,
  "stage": "stage identity",
  "run_fingerprint": "20-character fingerprint",
  "created_at_utc": "ISO-8601 timestamp",
  "parameters": {},
  "inputs": {"logical-name": {"path": "...", "bytes": 0, "sha256": "..."}},
  "outputs": {"logical-name": {"path": "...", "bytes": 0, "sha256": "..."}}
}
```

The manifest is the final commit marker. Outputs without it are partial and cannot resume.
Resume recomputes the stage, fingerprint, parameters, input records, and expected output path
set, then verifies every recorded input/output byte. Prepared-data raw acquisition inputs may
live outside the portable run; downstream verification intentionally requires the committed
prepared outputs rather than the old raw cache path.

## Checkpoint manifest

Every compute job seals its checkpoint as `checkpoint.manifest.json`:

```json
{
  "schema_version": 1,
  "run_fingerprint": "...",
  "checkpoint_root": "checkpoint",
  "metadata": {
    "job": {"stage": "budget", "variant": "slovakbert", "size": 50, "seed": 0},
    "execution_mode": "production",
    "load_subdir": "."
  },
  "files": {
    "config.json": {"path": "checkpoint/config.json", "bytes": 0, "sha256": "..."}
  }
}
```

`files` recursively records regular checkpoint files. Symlinks are forbidden when a
checkpoint is sealed. `load_subdir` must be relative, cannot contain `..`, and must exist.
Transfer job metadata records `optimizer_state_preserved: false`.

## Plan and identity

`plan/plan.json` contains:

- `schema_version`, `run_name`, `run_fingerprint`, `run_path`, `path_under_runs_root`;
- `execution_mode` (`production` or `tiny-nonproduction`);
- `counts` with the exact 15/1/2/3/9 groups, 29 supervised jobs, and 30 total jobs;
- ordered `stages`, their dependencies, and finalization protection flags; and
- ordered `jobs`, each with a deterministic `id`.

`plan/run.identity.json` contains the run name, fingerprint, execution mode, and complete
canonical configuration. `plan/config.snapshot.json` and
`plan/requirements-kaggle.lock` preserve the exact fingerprint inputs available from the
checkout.

## Prepared data

Prepared sentence JSONL records have:

```json
{
  "provenance": {
    "source_file": "czech/train/file.conllu",
    "split": "train",
    "sentence_index": 0,
    "sent_id": "optional source sent_id"
  },
  "comments": ["# sent_id = ..."],
  "form_hash": "SHA-256 of canonical JSON ordered FORM values",
  "tokens": [
    {"id": 1, "form": "Slovo", "line_number": 10, "upos": "NOUN"}
  ]
}
```

Train/dev/Slovak records include canonical UPOS. `prepared/czech/test.jsonl` uses the same
shape but **must not contain `upos`**.

`prepared/dataset_manifest.json` contains:

- `schema_version`, `run_fingerprint`, and `form_hash_algorithm`;
- `sources`: raw source byte counts and SHA-256 values;
- `splits`: sentence count, token count, and ordered FORM hashes for every retained split,
  dev role, seed, and budget; and
- `test_labels_exposed: false`.

`prepared/data_audit.json` records duplicate policy, raw/retained split counts, cross-split
overlap, removals, and `test_labels_exposed: false`.

## Compute job result

Each job `result.json` contains:

```json
{
  "schema_version": 1,
  "run_fingerprint": "...",
  "job": {},
  "metrics": {},
  "checkpoint_manifest": "portable/path/checkpoint.manifest.json",
  "experimental_valid": true,
  "metadata": {}
}
```

`metrics` is stage-specific. Supervised metrics include complete-word token accuracy,
sentence ≥98% rate/count, exact match, missing predictions, sentence error buckets, macro and
per-UPOS precision/recall/F1, and length bins. Aggregated metrics store the three source
values plus mean, population variance/std, minimum, and maximum at every numeric leaf.

## Selection and handoff artifacts

`budget/selection.json` contains the selected/reference budget, eligible budgets, fixed
policy tolerances, candidate summaries, and all three-seed aggregates.

`adapt/selection.json` contains the selected budget, winning arm, three winning checkpoint
manifest paths, non-inferiority rejections, deterministic eligible ranking, fixed policy, and
candidate aggregates.

`<stage>/handoff.json` contains:

```json
{
  "schema_version": 1,
  "stage": "budget-or-adapt",
  "run_name": "cs-rerun-v2",
  "run_fingerprint": "...",
  "run_path": "runs/cs-rerun-v2/<fingerprint>",
  "path_under_runs_root": "cs-rerun-v2/<fingerprint>"
}
```

Budget adds selected budget/selection/checkpoints. Adapt adds winning arm, selected budget,
three checkpoint manifests, ensemble, and calibration. All artifact paths are portable
relative paths under the run.

## Ensemble and calibration

`adapt/calibration/ensemble.json` identifies the fingerprint, winning arm, three checkpoint
manifests, method `unweighted-arithmetic-mean-probabilities`, member count `3`, and selection
metrics.

`adapt/calibration/calibration.json` identifies the fingerprint/winning arm and records:

- token temperature, fit-token count, NLL before/after;
- logistic sentence-success feature schema, standardization values, coefficients, intercept,
  and regularization;
- calibration-fit and assessment partition names;
- Brier score, log loss, complete risk–coverage points; and
- fixed-risk assessment at configured predicted failure risks.

Calibration fit and assessment evidence is development-only. It never contains official test
metrics.

## Draft report and selection lock

`report/report.draft.json` contains:

- `status: draft-selection-frozen-test-not-evaluated`;
- `run_fingerprint` and `selection_lock_sha256`;
- budget/adaptation selection, ensemble, calibration, audit, and optional benchmark; and
- `test_labels_accessed: false`.

`report/selection.lock.json` is the immutable finalization authorization object:

```json
{
  "schema_version": 1,
  "run_fingerprint": "...",
  "selection_frozen": true,
  "selected_budget": 50,
  "winning_arm": "...",
  "seeds": [0, 1, 2],
  "canonical_upos": ["ADJ", "...", "X"],
  "checkpoints": [],
  "checkpoint_hashes": [],
  "ensemble": {},
  "calibration": {},
  "data": {"test_labels_exposed": false},
  "selection_evidence": {},
  "final_test_evaluated": false
}
```

Each checkpoint entry fixes its seed, checkpoint-manifest file record/hash, recursive
checkpoint files, and job metadata. Ensemble/calibration sections include both a file record
and the exact frozen JSON value. Selection evidence fixes budget/adaptation artifacts and
their commit manifests.

## Optional benchmark

`benchmark/benchmark.json` contains `selection_influence: false`, fingerprint, timing scope,
and backend throughput measurements. No benchmark value is a selection input.

## One-shot final artifacts

Routine runs and CI do not create `final/`. If separately authorized, finalization first
writes `final/finalization.started.json` with state
`official-test-access-committed-one-shot`. It then writes:

- `test_metrics.json`: immutable per-seed metrics plus the fixed probability-ensemble metrics;
- `REPORT_FINAL.md`: human-readable final report using the frozen selection;
- `report.final.json`: frozen selection summary, official metrics, and
  `selection_changed_after_lock: false`; and
- `artifact-manifest.json`: one-shot stage commitment including the attempt marker.

The final status is `final-official-test-evaluated-once`. Any pre-existing `final/` directory
rejects another attempt.

## Standard output records

Every CLI stage mirrors important results as one line:

```text
SANNA_RESULT {"record_type":"...","run_fingerprint":"...",...}
```

The suffix is valid JSON. `record_type` includes `plan`, `prepare`, `job`,
`budget-selection`, `adaptation-selection`, `benchmark`, `report-draft`, and `final-test`.
Consumers must filter by the exact `SANNA_RESULT ` prefix and retain unknown additional keys
for forward compatibility. Files plus their manifests remain authoritative; stdout is a
portable recovery/monitoring channel, not the resume database.
