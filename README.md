# Sanna POS tagging

Building a Universal Dependencies POS tagger for **Sanna** (Cypriot Maronite Arabic), an
endangered Arabic variety written in a mixed Roman/Greek orthography with no treebank and
no pretrained model of its own.

Because Sanna has no gold data to measure against yet, the work starts on **proxy languages
that do** have gold treebanks. The proxy experiments validate the method where ground truth
exists before it is transferred to Sanna.

- **Maltese** — Arabic-base, non-Arabic superstrate, written in Latin script, with the MUDT
  gold treebank. It tests whether selective transliteration into Arabic script improves UPOS.
- **Czech** — an unrelated, data-rich proxy for annotation-budget scaling, related-language
  adaptation, confidence calibration, and a leakage-safe evaluation protocol.

The full rationale and locked Sanna/Maltese decisions remain in [DECISIONS.md](DECISIONS.md).

## Status

| Track | State |
|---|---|
| Maltese Step 1a selective-transliteration preprocessing | complete — [STEP_1A_RESULTS.md](STEP_1A_RESULTS.md) |
| Legacy v1 Maltese/Czech budget, calibration, and self-training notebooks | retained for history |
| Czech v2 reproducible rerun | implementation and offline test suite complete; production run pending |
| Sanna etymology heuristic and scaled annotation | parked until proxy evidence is complete |

## v2 Czech rerun

V2 is the current Czech experiment. It is a package under `src/sanna_tagging/`, driven by
`configs/cs_kaggle_v2.yaml`, and writes integrity-checked artifacts rather than append-only
CSV state. Its fixed compute plan has 30 jobs:

- 15 direct SlovakBERT budget jobs: 5 gold sizes × 3 seeds;
- 1 Czech language-adaptive pretraining (LAPT) job;
- 2 Slovak UPOS transfer jobs;
- 3 XLM-R reference jobs; and
- 9 adapted Czech jobs: 3 adaptation arms × 3 seeds.

The run fingerprint binds the canonical configuration, exact Git commit, and SHA-256 of
`requirements-kaggle.lock`. Every stage commits checksummed manifests. Resume succeeds only
when the fingerprint, inputs, parameters, outputs, sizes, and hashes match exactly.

### Local CLI

Use a source checkout; the configuration and dependency lock are repository-level runtime
inputs. Install the pinned environment and package without resolving alternate versions:

```bash
python -m pip install -r requirements-kaggle.lock
python -m pip install --no-deps -e .
```

Inspect the exact plan, then run stages in order:

```bash
sanna-tagging --runs-root runs plan
sanna-tagging --runs-root runs prepare
sanna-tagging --runs-root runs budget
sanna-tagging --runs-root runs adapt
# Optional, never used for selection:
sanna-tagging --runs-root runs benchmark
sanna-tagging --runs-root runs report-draft
```

Artifacts are written under
`runs/cs-rerun-v2/<20-character-fingerprint>/`. Machine-readable result records are also
mirrored to stdout with the `SANNA_RESULT ` prefix. See [docs/RUNBOOK.md](docs/RUNBOOK.md)
for stage handoffs, resume, Kaggle operation, and failure recovery, and
[docs/RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md) for artifact schemas.

### Protected official test

`report-draft` cannot accept or parse official test gold. It freezes the budget, winning arm,
three checkpoint manifests and hashes, unweighted ensemble, calibration, and selection
evidence in `report/selection.lock.json`.

`finalize-test` is a separate explicit one-shot operation. It requires that exact committed
lock and the pinned official CoNLL-U path. The attempt marker is sealed before gold bytes are
hashed or parsed; success or failure consumes that run's finalization attempt. Do not invoke
it during development, CI, model selection, or draft reporting.

### Kaggle

Open `notebooks/cs_kaggle_all.ipynb`, set Accelerator to GPU and Internet to On, replace the
exact detached-commit placeholder, choose one `STAGE`, and use
**Save Version → Save & Run All (Commit)**. Do not rely on an interactive session: Kaggle
removes `/kaggle/working` when it ends. Each committed stage leaves a portable run under
`/kaggle/working/sanna-v2/runs`; add that committed output as a read-only input to the next
version and set `PREVIOUS_RUN_DIR` to the prior fingerprint directory. The notebook verifies
the fingerprint and all committed output hashes before copying the run into the new writable
working directory.

## Legacy v1 and Maltese history

The flat modules in `src/*.py` and the numbered notebooks are the preserved v1 workflow. They
remain useful for reproducing the Maltese transliteration ladder and the original Czech
budget/calibration/self-training study; they are not imported by the v2 package.

- **Maltese Colab:** run `notebooks/01_data_prep.ipynb` through `05_report.ipynb` in order.
- **Legacy Czech Colab:** use `notebooks/cs_01_data_prep.ipynb` through `cs_05_report.ipynb`.
- Set `STUDY=mt` (default) or `STUDY=cs` for the legacy flat-module workflow.
- Legacy outputs use `results{_study}/` and append-only CSVs. V2 uses fingerprints and JSON
  manifests; the two resume formats are intentionally not interchangeable.

Step 1a requires `pynini` and OpenFst 1.8.2. Build notes and patches are recorded in
[STEP_1A_RESULTS.md](STEP_1A_RESULTS.md). Run it locally with:

```bash
NLTK_DISABLE_IMPORT_SECURITY=1 .venv/bin/python scripts/run_step1a.py
```

## Repository layout

```text
configs/cs_kaggle_v2.yaml    frozen v2 experiment contract
src/sanna_tagging/           v2 package: data, training, adaptation, selection, reporting, CLI
requirements-kaggle.lock     exact top-level Kaggle dependency pins
tests/                       offline unit and mocked end-to-end v2 tests
docs/RUNBOOK.md              operational stage and resume procedure
docs/RESULTS_SCHEMA.md       manifests, handoffs, locks, and result record schema
notebooks/cs_kaggle_all.ipynb thin stage-oriented v2 Kaggle CLI frontend
src/*.py                     preserved legacy v1 study pipeline
notebooks/01–05, cs_01–cs_05 preserved legacy Colab runners
scripts/run_step1a.py        Maltese Step 1a driver
data/processed/              Step 1a Latin/full/selective transliteration outputs
experiments/malti/           MLRS/malti at the 2024.eacl release
```

## Data and models

- **MUDT:** Maltese UD treebank used by the historical transliteration study.
- **UD Czech PDT-C and UD Slovak SNK:** v2 pins repository commits and source-file SHA-256
  values in the YAML configuration.
- **SlovakBERT and XLM-R:** v2 pins exact model revisions. Runtime code does not request
  mutable latest revisions.
- All models are open-weight; commercial models remain outside project scope.

## Development checks

```bash
python -m pip install -e '.[kaggle,test]'
python -m pytest
python -m compileall -q src tests
ruff check src/sanna_tagging tests
python -m json.tool notebooks/cs_kaggle_all.ipynb >/dev/null
```

CI sets Hugging Face offline variables and uses only synthetic/mocked model boundaries; it
does not download corpora or models and never invokes `finalize-test`.

## License

MIT — see [LICENSE](LICENSE). `experiments/malti/` is a clone of
[MLRS/malti](https://github.com/MLRS/malti) and carries its own license.
