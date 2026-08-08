# Sanna POS tagging

Building a Universal Dependencies POS tagger for **Sanna** (Cypriot Maronite Arabic), an
endangered Arabic variety written in a mixed Roman/Greek orthography with no treebank and
no pretrained model of its own.

Because Sanna has no gold data to measure against yet, the work starts on **proxy languages
that do** have gold treebanks. Everything in this repo right now is that groundwork: the
method is validated where ground truth exists, then carried over to Sanna.

- **Maltese** — Arabic-base, non-Arabic superstrate, written in Latin script, with a gold UD
  treebank (MUDT). The closest measurable structural analogue to Sanna, used to test whether
  *selective transliteration into Arabic script* buys accuracy.
- **Czech** — unrelated language, used purely as a large-treebank scale study: how POS accuracy
  behaves as annotation budget grows, how well confidence calibrates, and how much self-training
  recovers. These curves inform how much human annotation Sanna will actually need.

The full decision log, method rationale and reading list live in [DECISIONS.md](DECISIONS.md).

## Status

| Step | What | State |
|---|---|---|
| 1a | Run the MLRS selective-transliteration pipeline over MUDT | done — [STEP_1A_RESULTS.md](STEP_1A_RESULTS.md) |
| 1b | Budget curve / calibration / self-training study | code complete, runs on Colab + Kaggle |
| 2 | Sanna-side etymology heuristic | parked until the mechanism is proven |

## Repo layout

```
src/                  study pipeline, one module per stage
  runpaths.py         resolves the results/cache root (Drive, Kaggle, or local)
  data_utils.py       CoNLL-U reading, splits, model + seed-size constants
  prep.py             stage 1: fetch/cache treebank splits
  budget.py           stage 2: accuracy vs. annotation budget, 3 seeds per point
  calibrate.py        stage 3: confidence calibration + coverage/precision curve
  selftrain.py        stage 4: silver-data self-training rounds
  report_gen.py       stage 5: frozen test eval + REPORT.md
  modeling.py         encode / train / evaluate / predict
  results_io.py       append-only CSV rows, runtime and problem logs
notebooks/            thin runners: 01–05 Maltese, cs_01–cs_05 Czech, cs_kaggle_all (single-cell Kaggle run)
scripts/run_step1a.py driver for the step-1a transliteration run
data/raw/             MUDT as downloaded
data/processed/       latin/ · arabic_full/ · arabic_partial/ — step-1a output
experiments/malti/    plain clone of MLRS/malti @ 2024.eacl (transliteration + etymology classifier)
```

## Running it

### Step 1a — transliteration (local, macOS/Linux)

Needs `pynini`, which needs OpenFst 1.8.2 built from source; the build notes and the
patches that were required are in [STEP_1A_RESULTS.md](STEP_1A_RESULTS.md).

```bash
NLTK_DISABLE_IMPORT_SECURITY=1 .venv/bin/python scripts/run_step1a.py
```

Writes the three script variants under `data/processed/` plus `scripts/step1a_stats.json`.

### The study — Colab or Kaggle (GPU)

The notebooks are deliberately thin: they clone this repo, put `src/` on the path, and call
one `main()` per stage. All the logic is in `src/`, so a fix is a commit, not a notebook edit.

- **Colab:** open `notebooks/01_data_prep.ipynb` … `05_report.ipynb` in order. Results are
  written to `MyDrive/sanna_m1/`.
- **Kaggle:** open `notebooks/cs_kaggle_all.ipynb`, set Accelerator = GPU and Internet = On,
  then use **Save Version → Save & Run All (Commit)**. Do *not* run it interactively — Kaggle
  wipes `/kaggle/working` when an interactive session ends, so an overnight interactive run
  loses everything. A committed run executes headless and attaches the output permanently.

Set `STUDY=cs` for the Czech study, `STUDY=mt` (the default) for Maltese.

Every stage is **resumable and append-only**: each finished configuration is recorded as a CSV
row, and re-running skips whatever is already there. Result rows are also echoed to stdout, so
the numbers survive even if the filesystem is wiped. Outputs land in `results{_study}/`:
`budget.csv`, `coverage.csv`, `selftrain.csv`, `derived_params.json`, `test_result.json`,
the PNG curves, and a generated `REPORT.md`.

The test split is guarded — `report_gen.py` refuses to re-run it once `test_result.json` exists,
so the frozen test set is consulted exactly once.

## Data and models

- **MUDT** — Maltese UD treebank, vendored under `data/raw/`.
- **UD Czech PDT-C** — downloaded at runtime from the UD GitHub repo (5 train parts, ~82k sentences).
- **Models** — XLM-R base, CAMeL-BERT mix, SlovakBERT. All open-weights; no commercial models,
  which is a hard project constraint.

## License

MIT — see [LICENSE](LICENSE). `experiments/malti/` is a clone of
[MLRS/malti](https://github.com/MLRS/malti) and carries its own license.
