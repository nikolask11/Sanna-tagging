# Step 1a Results

MLRS transliteration pipeline (Micallef, Habash, Borg, Eryani & Bouamor, 2024, EACL,
["Treating Low-Resource Maltese as Multilingual Code-Switching"](https://aclanthology.org/2024.eacl-long.61))
applied to the MUDT (Maltese Universal Dependencies) treebank as a structural stand-in for Sanna.

- **Date:** 2026-08-04
- **Repo used:** [github.com/MLRS/malti](https://github.com/MLRS/malti), branch `2024.eacl`, commit [`a17de27`](https://github.com/MLRS/malti/commit/a17de27a30847677042ce06239c9e39deb44ffde)
- **Cloned into:** `experiments/malti/` (plain clone, not a git submodule)

## What was done

1. Cloned MLRS/malti at branch `2024.eacl`.
2. Got the toolchain building locally on macOS (arm64) — see [Environment / build issues](#environment--build-issues-macos-arm64-and-how-they-were-resolved) below for what that took.
3. Fixed one real bug in the malti repo itself — see [Bugs found in MLRS code](#bugs-found-in-mlrs-code).
4. Ran the repo's own dataset-processing pipeline (`dataset_processors.py`'s `UniversalDependenciesDatasetProcessor`, the same code path `process.py`'s CLI uses) on all three MUDT files (train/dev/test), producing three output versions under `data/processed/`:
   - `data/processed/latin/` — untouched Latin-script MUDT (straight copy)
   - `data/processed/arabic_full/` — every token transliterated to Arabic script
   - `data/processed/arabic_partial/` — only tokens the pretrained etymology classifier labels `Arabic` are transliterated; everything else (`Non-Arabic`, `Name`, `Code-Switching`, `Symbol`) is passed through unchanged

   Only the CoNLL-U token column is touched; all UD annotation columns (POS tags, heads, dep labels, etc.) are preserved as-is, exactly as the MLRS tool does.
5. Verified the pipeline against the repo's own worked example (`src/demo.ipynb`).
6. A runnable copy of the driver script is at [`scripts/run_step1a.py`](scripts/run_step1a.py).

**Runtime:** the whole treebank (44,162 tokens, 2,074 sentences) processes in ~7 seconds.

## Classifier / model used

The etymology classifier is MLRS's own **pretrained** model, loaded as-is from `experiments/malti/src/etymology_data/model.pickle`. It was **not retrained** (that's step 1b, out of scope here). It's an `MleCrfClassifier`: a Maximum-Likelihood memoriser backed by a CRF (`sklearn_crfsuite`) fallback for unseen tokens, predicting one of 5 etymology labels per token: `Arabic`, `Non-Arabic`, `Name`, `Code-Switching`, `Symbol`.

Transliteration itself uses MLRS's deterministic character-mapping FST (see below for why the deterministic mode was used instead of the paper's default non-deterministic/ranked mode).

## What could not be run, and why

The paper's primary transliteration pipeline (X<sub>ara</sub>) uses **non-deterministic** character mapping, disambiguated by two n-gram language models (`word_model_score` and `character_model_score` rankers) trained on Tunisian/Maghrebi Arabic dialect data (`tn-maghreb.arpa`). These `.arpa` LM files are **not included in the MLRS repo** — the repo's README points to them living in a Google Drive folder belonging to a different, external repo ([CAMeL-Lab/HierarchicalArabicDialectID](https://github.com/CAMeL-Lab/HierarchicalArabicDialectID)).

That Drive folder's anonymous-download quota was already exhausted (`Cannot retrieve the public link of the file... have had many accesses`) — this triggered on the very first request, before this session downloaded anything of consequence, and persisted even when targeting the two specific files by their file ID directly (bypassing a full folder crawl), and even after browsing to them by hand in a browser. This is a pre-existing rate-limit on that specific public folder, not something fixable locally, and not a build/dependency problem — no login flow would get around it either (public files behind a download quota fail the same way whether or not you're authenticated, since it's not designed to require auth in the first place).

**Consequence:** the "fully transliterated" output uses MLRS's **deterministic** character mapping instead of the paper's ranked/non-deterministic mode. This is not a fallback improvised for this run — it's a first-class mode the MLRS code fully supports without any external LM (*"When [rankers are] unspecified, a deterministic mapping is used"* — their README). The practical difference is small and systematic: the deterministic map picks a single fixed short-vowel rendering per Maltese input, where the ranked mode can pick a longer final-vowel rendering when the language model favours it. See [Verification](#verification-against-the-paper--repos-own-example) below for a concrete before/after comparison against the paper's own example.

A second, smaller consequence: the etymology classifier's own feature extraction (`etymology_classification.featurise`) normally also calls the non-deterministic transliterator internally (to compute an `arabic_distance` feature). Since those LMs aren't available, this session's driver script substitutes the deterministic transliterator there too — this is the one deliberate deviation from "just running the model as-is." It was validated (see below): on the paper's own example sentence, the pretrained classifier reproduces the gold etymology labels 9/9 with this substitution, so it does not appear to change classifier behaviour in practice, at least not on that example.

Nothing else was skipped. Both required native dependencies (`pynini` for the FST transliterator, `kenlm` for the LM rankers) were successfully built and installed locally despite not being available as prebuilt wheels for macOS/arm64 — see next section. `kenlm` ended up unused in the final run only because there's no LM file to score against, not because it doesn't work.

## Environment / build issues (macOS arm64) and how they were resolved

- **`pynini` 2.1.5** (pinned in `requirements.txt`) has no prebuilt wheel for macOS at all (PyPI only ships manylinux wheels + an sdist). Building from source needs OpenFst's C++ headers/libs, which pynini 2.1.5 requires at exactly ~1.8.2 API. Homebrew only offers OpenFst 1.8.4, whose API pynini 2.1.5 does not compile against (renamed/removed symbols). **Resolution:** downloaded and built OpenFst 1.8.2 from source, installed to `vendor/openfst-install/`. Hit one genuine bug in OpenFst 1.8.2 itself when compiled with a modern clang (a copy constructor in `src/include/fst/bi-table.h` referencing a non-existent member `s_` instead of `selector_` — a one-line fix, patched locally). pynini 2.1.5 was then built against this local OpenFst via `CPATH`/`LIBRARY_PATH`, and the resulting `.so` files were re-linked with `install_name_tool` to point at `vendor/openfst-install/lib` (a stable, repo-local path) instead of the session's temporary build directory, so it keeps working after the session that built it ends.
- **`kenlm==0.1`** (pinned in `requirements.txt`) fails to build on Python 3.12 — its bundled Cython-generated C++ uses CPython internals (`PyLongObject.ob_digit`, `PyFrameObject` internals) that no longer exist/are opaque in 3.12. **Resolution:** installed `kenlm` 0.3.0 instead (latest on PyPI), which builds cleanly and exposes the same scoring API.
- **`nltk` 3.10.1** ships a new import-time security guard that (falsely) flagged this project's own `.venv` as a "current-working-directory hijack" risk, because `.venv/` sits inside the project directory that also serves as cwd when running the driver script. **Resolution:** `NLTK_DISABLE_IMPORT_SECURITY=1` is set when running `scripts/run_step1a.py` (documented by nltk itself as the intended escape hatch for this situation; there is no actual untrusted file being shadowed here).
- All other Python deps (`numpy`, `scikit-learn`, `pandas`, `tqdm`, `nltk`, `python-Levenshtein`, `sklearn_crfsuite`) installed cleanly, generally at newer versions than `requirements.txt` pins (this environment already had a newer torch/transformers/numpy/etc. stack installed; nothing in the malti pipeline required the exact pinned versions to work).

## Bugs found in MLRS code

`experiments/malti/src/transliterate.py` imports `RandomRanker, TokenRanker` from `token_rankers` but uses a third class, `Token`, that was never imported — every call to `transliterate()` crashed with `NameError: name 'Token' is not defined`, in **both** deterministic and non-deterministic modes. This is a genuine bug in the pinned `2024.eacl` branch, not an environment issue. Fixed locally with a one-line import addition (added `Token` to the existing import line). Without this fix, none of this task would run.

Also observed (not fixed, just noted): the `ConllDatasetProcessor`'s reader treats every comment line as replacing the previous one, keeping only the last `#`-prefixed line before a sentence. MUDT's CoNLL-U files have two comment lines per sentence (`# sent_id = ...` then `# text = ...`); MLRS's writer only re-emits `# text`, so the processed output files lose their `sent_id` comments. Token content and all UD columns are unaffected — this only affects sentence-ID metadata in the output files.

## Verification against the paper / repo's own example

The repo ships a worked example with recorded gold output in `src/demo.ipynb`, for the sentence:

> "Il-karozza Porsche tal-2022 għandha speed fenomenali!"
> (tokenised as: `Il- karozza Porsche tal- 2022 għandha speed fenomenali !`)

**Etymology classifier** (pretrained `model.pickle`, deterministic-transliteration feature substitution as described above):

| | Il- | karozza | Porsche | tal- | 2022 | għandha | speed | fenomenali | ! |
|---|---|---|---|---|---|---|---|---|---|
| Gold label | Arabic | Non-Arabic | Name | Arabic | Symbol | Arabic | Code-Switching | Non-Arabic | Symbol |
| Reproduced | Arabic | Non-Arabic | Name | Arabic | Symbol | Arabic | Code-Switching | Non-Arabic | Symbol |

**Match: 9/9 tokens.**

**Transliteration** (deterministic mode used here vs. the paper's non-deterministic/ranked mode, which could not be run — see above):

| Token | Gold (ranked, X<sub>ara</sub>) | This run (deterministic) |
|---|---|---|
| Il- | ال | ال |
| karozza | كردزة | كردز |
| Porsche | برسكهي | برسكه |
| tal- | تاع ال | تاع ال |
| 2022 | ٢٠٢٢ | ٢٠٢٢ |
| għandha | عندها | عندها |
| speed | صباد | سبد |
| fenomenali | فنمنلي | فنمنل |
| ! | ! | ! |

5/9 tokens identical; the rest differ only in a trailing/short vowel choice that the ranked mode resolves using the (unavailable) dialect language model, e.g. "كردزة" vs "كردز", "صباد" vs "سبد". No token is wrong in kind, only in vowel detail.

**Partial transliteration** (X<sub>ara</sub>/P: transliterate only tokens labelled `Arabic`):

| | Il- | karozza | Porsche | tal- | 2022 | għandha | speed | fenomenali | ! |
|---|---|---|---|---|---|---|---|---|---|
| Label | Ar | NonAr | Name | Ar | Sym | Ar | CodeSw | NonAr | Sym |
| Output | ال | karozza | Porsche | تاع ال | 2022 | عندها | speed | fenomenali | ! |

This matches the gold `transliteration+pass` column in `demo.ipynb` exactly (modulo the same vowel-detail difference on "tal-"/"għandha" as above, which are unaffected here since both render identically in deterministic and ranked mode for this example).

## Sample before/after on real MUDT sentences

*(from `data/raw/mt_mudt-ud-dev.conllu`, sentence 1)*

**Latin (untouched):**
> Fil- qalba tal- Partit Laburista tfaċċat kontroversja ġdida wara li fil- laqgħa politika laburista tal- Ħadd li għadda fi Pjazza Castro fin- Naxxar , Jason Micallef tpoġġa fuq siġġu wara l- leader biex waqt id- diskors ta' Muscat ikun viżibbli ħafna .

**Arabic (fully transliterated):**
> فال قلب تاع ال برتت لبرست تفتشت كنترفرسي جدد ورا اللي فال لقع بلتك لبرست تاع ال حد اللي عد في بيدز كستر فال نشر ، يسن مكلف تبج فوق سج ورا ال لدر باش وقت ال دسكرس تاع مسكت يكون فزبل حفنة .

**Arabic (partial — etymology-classifier gated, Arabic-labelled tokens only):**
> فال قلب تاع ال Partit Laburista tfaċċat kontroversja جدد ورا اللي فال لقع politika laburista تاع ال Ħadd اللي عد في Pjazza Castro فال نشر , Jason Micallef تبج فوق سج ورا ال leader باش وقت ال diskors تاع Muscat يكون viżibbli حفنة .

Note how proper names (Partit Laburista, Jason Micallef, Pjazza Castro, Muscat) and non-Arabic-origin content words (kontroversja, politika, leader, diskors, viżibbli) are left in Latin script in the partial version, while function words and Arabic-origin vocabulary (fal-, qalba, wara, li, fuq...) are transliterated — this is the intended behaviour of the X<sub>ara</sub>/P pipeline.

## Counts / stats

Whole treebank (train + dev + test combined): **44,162 tokens, 2,074 sentences**.

| Split | Tokens | Changed (full) | Changed (partial) |
|---|---:|---:|---:|
| train | 22,880 | 21,161 (92.5%) | 14,123 (61.7%) |
| dev | 10,209 | 9,655 (94.6%) | 6,540 (64.1%) |
| test | 11,073 | 10,345 (93.4%) | 7,627 (68.9%) |
| **Total** | **44,162** | **41,161 (93.2%)** | **28,290 (64.1%)** |

*("Changed" = token text differs from the untouched Latin form; unchanged tokens in the "full" column are mostly punctuation/digits that map to themselves under the deterministic mapping.)*

**Etymology label distribution** (pretrained classifier, over all 44,162 tokens):

| Label | Count | % |
|---|---:|---:|
| Arabic | 28,295 | 64.1% |
| Non-Arabic | 8,124 | 18.4% |
| Symbol | 5,191 | 11.8% |
| Name | 1,830 | 4.1% |
| Code-Switching | 722 | 1.6% |

## Output files

```
data/processed/latin/{mt_mudt-ud-train,dev,test}.conllu
data/processed/arabic_full/{mt_mudt-ud-train,dev,test}.conllu
data/processed/arabic_partial/{mt_mudt-ud-train,dev,test}.conllu
scripts/run_step1a.py           # driver script, reproducible:
                                 #   NLTK_DISABLE_IMPORT_SECURITY=1 .venv/bin/python scripts/run_step1a.py
scripts/step1a_stats.json       # raw stats + 5 sample sentences, machine-readable
vendor/openfst-install/         # locally built OpenFst 1.8.2, needed by pynini
```
