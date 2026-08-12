# Czech v2 runbook

This runbook operates the manifest-driven Czech v2 experiment. The legacy Maltese/Czech v1
notebooks use a different CSV resume format and are not inputs to this procedure.

## Safety rules

1. Run only committed code from an exact detached Git commit. The commit is part of the run
   fingerprint; do not start a production run from a dirty or untracked checkout.
2. Keep the configured corpus/model revisions, file SHA-256 values, canonical UPOS order,
   seed sizes, and seeds unchanged during a run.
3. Run stages in order: `plan`, `prepare`, `budget`, `adapt`, optional `benchmark`, then
   `report-draft`.
4. Do not provide official test gold to preparation, selection, adaptation, calibration,
   benchmarking, or draft reporting.
5. Do not invoke `finalize-test` until the draft and `selection.lock.json` have been reviewed
   and the one-shot access is explicitly authorized. A failed attempt cannot be retried in the
   same fingerprint directory.

## Prerequisites

- Python 3.11 for the resolved Kaggle environment (the package supports 3.10–3.12).
- A GPU runtime for production training.
- Internet access for the initial pinned corpus/model acquisition.
- Enough persistent space for 30 compute jobs and their sealed checkpoints.
- A clean checkout at the intended commit.

```bash
git status --short
git rev-parse HEAD
python -m pip install -r requirements-kaggle.lock
python -m pip install --no-deps -e .
```

`requirements-kaggle.lock` is a fully resolved set of exact Kaggle Linux/CPython 3.11
runtime versions. Its exact bytes are part of the run fingerprint and are snapshotted under
`plan/`; package hashes are not embedded, so retain the committed lock and pip index provenance.

Choose one artifact root and use it for every local stage:

```bash
export RUNS_ROOT="$PWD/runs"
export CONFIG="$PWD/configs/cs_kaggle_v2.yaml"
```

Global CLI options must appear before the subcommand.

## 1. Inspect and commit the plan

A dry run writes nothing and prints a `SANNA_RESULT` JSON record:

```bash
sanna-tagging --config "$CONFIG" --runs-root "$RUNS_ROOT" --dry-run plan
```

The plan must report exactly:

| Job group | Count |
|---|---:|
| Budget supervised | 15 |
| Czech LAPT | 1 |
| Slovak transfer | 2 |
| XLM-R reference supervised | 3 |
| Adapted supervised | 9 |
| Total supervised | 29 |
| Total compute | 30 |

Materialize it:

```bash
sanna-tagging --config "$CONFIG" --runs-root "$RUNS_ROOT" plan
```

Record the printed 20-character fingerprint. The run directory is
`$RUNS_ROOT/cs-rerun-v2/<fingerprint>/`. Inspect:

- `plan/plan.json`
- `plan/run.identity.json`
- `plan/config.snapshot.json`
- `plan/requirements-kaggle.lock`
- `plan/artifact-manifest.json`

## 2. Prepare pinned data

```bash
sanna-tagging --config "$CONFIG" --runs-root "$RUNS_ROOT" prepare
```

An optional immutable acquisition cache may be supplied:

```bash
sanna-tagging --config "$CONFIG" --runs-root "$RUNS_ROOT" prepare \
  --raw-root /persistent/pinned-ud-cache
```

Before proceeding, verify that:

- `prepared/data_audit.json` says `test_labels_exposed: false`;
- `prepared/dataset_manifest.json` has the expected fingerprint;
- `prepared/czech/test.jsonl` contains FORM/provenance but no `upos` keys;
- `prepared/artifact-manifest.json` exists; and
- selection, calibration-fit, and calibration-assessment hashes are disjoint.

Preparation applies duplicate precedence `test > dev > train`, retains official test order,
creates deterministic nested gold samples for every seed, and commits all outputs atomically.

## 3. Run the budget curve

```bash
sanna-tagging --config "$CONFIG" --runs-root "$RUNS_ROOT" budget
```

This executes or resumes 15 jobs under `budget/jobs/n<size>-seed<seed>/`. Each complete job
contains `result.json`, `checkpoint.manifest.json`, `artifact-manifest.json`, and a checkpoint
directory. The final selection is in:

- `budget/selection.json`
- `budget/selection.manifest.json`
- `budget/handoff.json`
- `budget/handoff.manifest.json`

Do not delete a failed partial job directory and continue blindly. Preserve it for diagnosis;
a fresh run identity or an explicitly cleaned failed job directory is required.

## 4. Run adaptation, selection, ensemble, and calibration

```bash
sanna-tagging --config "$CONFIG" --runs-root "$RUNS_ROOT" adapt
```

This executes 15 additional compute jobs: one Czech LAPT, two Slovak transfers, three XLM-R
references, and nine adapted Czech supervised jobs. The stage then applies the fixed
non-inferiority/ranking policy, freezes the winning three-seed checkpoint set, creates the
unweighted probability ensemble, and fits/assesses calibration.

Inspect:

- `adapt/selection.json` and `adapt/selection.manifest.json`
- `adapt/calibration/ensemble.json`
- `adapt/calibration/calibration.json`
- `adapt/calibration/artifact-manifest.json`
- `adapt/handoff.json` and `adapt/handoff.manifest.json`

All paths in handoffs are relative to the run directory. Transfer metadata must say
`optimizer_state_preserved: false`.

## 5. Optional non-selective benchmark

```bash
sanna-tagging --config "$CONFIG" --runs-root "$RUNS_ROOT" benchmark
```

The benchmark measures the already frozen ensemble. It is optional and must not alter budget,
arm, checkpoint, ensemble, or calibration selection. Its output is
`benchmark/benchmark.json` with a commit manifest.

## 6. Freeze the no-test draft

```bash
sanna-tagging --config "$CONFIG" --runs-root "$RUNS_ROOT" report-draft
```

Review:

- `report/report.draft.json`
- `report/report.draft.md`
- `report/selection.lock.json`
- `report/artifact-manifest.json`

Required lock properties include `selection_frozen: true`, `final_test_evaluated: false`, the
matching fingerprint, exactly three unique seeds/checkpoint hashes, the arithmetic-mean
ensemble, calibration artifacts, and `data.test_labels_exposed: false`. Archive the complete
run before considering official test access.

## Kaggle stage handoff

Use `notebooks/cs_kaggle_all.ipynb` as the frontend. It writes only beneath
`/kaggle/working/sanna-v2`, which Kaggle attaches to a committed notebook version.

For every stage:

1. Set Accelerator = GPU and Internet = On.
2. Set `GIT_COMMIT` to the exact committed 40-character revision.
3. Set one `STAGE`: `prepare`, `budget`, `adapt`, or `report-draft` (the first `prepare` run
   materializes `plan` automatically).
4. Use **Save Version → Save & Run All (Commit)**, not an interactive overnight run.
5. After completion, inspect `SANNA_RESULT` lines and the run handoff.
6. Add the previous committed version's output as a Kaggle read-only input.
7. Set `PREVIOUS_RUN_DIR` to its exact
   `/kaggle/input/.../runs/cs-rerun-v2/<fingerprint>` directory.
8. Commit the next stage.

Before copying prior state, the notebook:

- computes the expected fingerprint from the detached checkout and lock;
- requires the source directory name and `run.identity.json` fingerprint to match;
- rejects symlinks and absolute/escaping manifest paths;
- verifies every committed artifact output and every checkpoint file by size and SHA-256; and
- copies only after verification into a new writable run directory.

The source under `/kaggle/input` remains read-only. Never point `--runs-root` at it.

### Kaggle image quirks

Both are handled inside the notebook, not by the lock or the configuration. Notebook content
is deliberately outside the run fingerprint, so these fixes are fingerprint-neutral; editing
`requirements-kaggle.lock` or the YAML instead would produce a different run.

- **Preinstalled torchvision/torchaudio.** Kaggle ships them built against a newer torch than
  the lock pins, so `import transformers` fails with
  `operator torchvision::nms does not exist`. Neither is used here. The notebook uninstalls
  both and then asserts they are unimportable. Do not widen the lock to accommodate them.
- **Dataset mount paths.** Plain datasets mount at
  `/kaggle/input/datasets/<user>/<slug>/<file>`, not the documented
  `/kaggle/input/<slug>/<file>`; notebook-output inputs mount at
  `/kaggle/input/notebooks/<user>/<notebook-slug>/<subdir>`. A `PREVIOUS_RUN_DIR` or
  `OFFICIAL_TEST_PATH` can therefore be right on paper and still miss. For gold, the notebook
  recovers by filename under `/kaggle/input`, excluding the `notebooks/` subtree — a `.conllu`
  under there is resumed run state, not gold — and requires exactly one match.

### Reading the output of a committed version

The Kaggle log page **renumbers its retained window from 1**, so a log showing lines 1..N with
no gaps can still be a tail-only view of a much longer run. Do not treat log line numbering as
evidence of completeness; confirm by content. The authoritative full stdout of a committed
version is the executed notebook JSON, including outputs, at
`/kernels/scriptcontent/<scriptVersionId>/download`. For run `cs-rerun-v2` this was 141,382
lines against the log page's retained 30,822. The public output API, the internal session-log
endpoints, and the Output file browser all failed for a 1.86 GB output.

## Resume and recovery

A normal rerun of a completed stage is safe: `strict_resume` verifies identity and all recorded
bytes before returning the existing result. It never silently overwrites mismatched state.

| Failure | Meaning | Action |
|---|---|---|
| fingerprint mismatch | config, commit, lock, or execution mode changed | use the matching checkout/lock or start a new run |
| output exists without manifest | interrupted partial publication | diagnose and remove only that failed partial output, or start a new run |
| input/parameter/output-set mismatch | stage identity changed | do not merge states; start a new fingerprint directory |
| SHA-256/size mismatch | artifact corruption or mutation | restore the exact committed output or rerun from a clean upstream handoff |
| checkpoint manifest failure | missing, altered, or unsafe checkpoint | do not load it; recompute from verified upstream state |
| missing stage dependency | prior handoff absent or invalid | restore/complete the prior stage first |
| existing `final/` directory | finalization already claimed | never retry in that run; preserve evidence and follow the one-shot policy |

Do not edit result JSON, manifests, checkpoints, or the selection lock in place. Any legitimate
method/configuration change requires a new fingerprint.

## Official test finalization checklist

This section documents the gate; routine operation and CI stop before it.

- Draft and selection lock independently reviewed.
- Complete run archived and hashes recorded.
- Exact official test source and configured SHA-256 confirmed without opening labels in the
  ordinary pipeline.
- `selection.lock.json` path is the exact file committed under this run's `report/` directory.
- One-shot access explicitly authorized and scheduled once.
- No `final/` directory exists.
- **The resolved gold path has been hashed and compared to
  `config.data.czech.test[0].sha256` before dispatch.** The attempt marker is sealed before
  `finalize_official_test` checksums gold, so an unverified dispatch spends the single attempt
  on a file that was never eligible. Verify while aborting is still free.

Only then construct the explicit command shown by `sanna-tagging finalize-test --help`. The
attempt marker is written before the official gold checksum and parser run. There is no retry
or selection change after that marker.

### Consumed attempts

| Run | Fingerprint | Status |
|---|---|---|
| `cs-rerun-v2` | `30a9fbaa373488ecfeec` | finalized 2026-08-12 — attempt consumed, see [results/cs_v2/](../results/cs_v2/) |

Any correction to a finalized run — including recalibration — requires a new fingerprint.
