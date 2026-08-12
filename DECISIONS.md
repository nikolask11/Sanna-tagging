# Sanna POS Project — Decisions & Current Plan

_Running decision log. Append a dated line whenever a choice is made or changed, with who made it and one sentence of why. This doubles as the backbone of the paper's methodology section._

_Last updated: 2026-08-12 · Maintained by Nikolas_

---

## 1. Locked decisions

### Scope & task
- **D1.** Assign Universal Dependencies UPOS tags to every word of the 100,000-sentence Sanna corpus.
- **D2.** All output delivered in the modern standardized Sanna orthography (mixed Roman/Greek). Any internal transliteration is reversed before output; the original is never discarded.
- **D3.** POS first. Dependencies (HEAD/DEPREL) are a possible later phase, not in current scope.

### Method & architecture
- **D4.** Tagging is **contextual** (whole-sentence), never word-in-isolation — the same form takes different tags by context (e.g. Sanna *I* = definite article vs 1sg possessive).
- **D5.** Core model = a pretrained **encoder + token-classification head, fine-tuned**. Not zero/few-shot prompting — the shortlisted models are encoders, which learn by fine-tuning, not prompts. (Open generative models could later *pre-draft* annotations for human correction, but never produce gold.)
- **D6.** **Adaptive pretraining (TAPT):** let the chosen encoder read the full raw corpus (unlabelled) before any tagging, to learn Sanna's spelling, vocabulary, and word order.
- **D7.** **Human-in-the-loop:** model tags with a calibrated confidence score; high-confidence tags are accepted automatically, low-confidence tags are routed to expert review.
- **D8.** **Self-training loop:** high-confidence predictions become provisional "silver" training data and are recycled into training; gold data is upweighted; accuracy is checked on the dev set after each round; stop when dev accuracy plateaus (typically 2–3 rounds).
- **D9.** **Active learning** selects review batches by uncertainty, diversified by clustering so the reviewer isn't shown many variants of the same ambiguity. The expert *corrects drafts* rather than annotating blank text (≈3–5× faster).

### Gold / silver distinction
- **D10.** **Gold = human-verified only.** Model output is *silver* and can never serve as the gold standard or as the evaluation reference. The ~400-sentence human-annotated dev+test core is irreducible under any timeline.

### Data & preprocessing
- **D11.** Phase-0 preprocessing is a fixed, reproducible script sequence kept in the repo, each step file-in → file-out:
  1. Intake & UTF-8 encoding (+ source ID per line)
  2. Character canonicalization (Unicode normalization + homoglyph table, e.g. Greek χ / Latin x → one codepoint)
  3. Deduplication (exact + near-duplicate)
  4. Language filtering (keep Sanna; drop Cypriot Greek, Standard Greek, Turkish, English)
  5. Sentence segmentation
  6. Spelling normalization (older/ad-hoc → standard orthography)
  7. Tokenization (under the agreed clitic policy)
- **D12.** **Clean the full 100k first, then sample gold/dev/test from the cleaned pool.** Normalization (steps 2, 6, 7) is identical across all sets; destructive filters (steps 3, 4) run only on the bulk corpus, never on the protected evaluation sentences.
- **D13.** Orthographic normalization + clitic/article **segmentation policy must be locked with David before scaling annotation** (it changes every downstream count). Includes: clitics split as separate tokens (per UD Arabic treebanks), definite-article allomorphs (*li / l / I* / assimilated-geminated forms) normalized to a single lemma.
- **D14.** Store all annotation as valid CoNLL-U from day one (FEATS pipe-separated, alphabetical, no spaces — validator-clean).

### Evaluation
- **D15.** **Test set (~300 sentences)** is frozen and consulted exactly once, at the end. **Dev set (~100)** drives every decision (model choice, thresholds, stopping). ~100 sentences double-annotated to report inter-annotator agreement (the real accuracy ceiling).
- **D16.** Calibrate confidence per-tag on dev (temperature scaling or precision-target threshold sweep); report overall accuracy, per-POS P/R/F1, accuracy split by Greek-origin vs Arabic-origin tokens, a learning curve, and the coverage–precision curve.
- **D17.** Small labelled sets are noisy → run every configuration with **3 random seeds** and report averages.

### Models & hyperparameters
- **D18.** Encoder choice is an **empirical comparison**, decided on dev: CAMeL-BERT (with transliteration), XLM-R (native Latin/Greek), BERTu (Maltese). All open-weights, offline — **no commercial models**, satisfying the project constraint.
- **D19.** Token-classification fine-tuning defaults: lr 2e-5 or 5e-5, batch 16–32, ≤~15 epochs with early stopping on dev (patience 3), 10% warmup, weight decay 0.01, max length 128. Label alignment: tag on first subword piece, mask the rest with −100.

### Transliteration approach (resolved)
- **D20.** "Transliteration" here means mapping **Arabic-origin words into Arabic script** to reach an Arabic model's knowledge — **not** romanization. The method is **selective / etymology-conditioned**: transliterate only Arabic-origin tokens, leave non-Arabic tokens (Greek-origin in Sanna; Italian/English in Maltese) in their original script, producing mixed-script text. This is the method of **Micallef et al. 2024 (EACL)**, which shows selective transliteration beats both full transliteration and raw text. ("Partial transliteration" = this selective approach; the paper being reproduced is Micallef et al. 2024, **not** Muller et al. 2021 — Muller is background, and romanizes toward Latin, the opposite direction.)
- **D21.** The engine of the method is a **word-level etymology decision** (which tokens are Arabic-origin). For the Maltese leg we **reuse the authors' released etymology data + classifier** rather than rebuild. For Sanna (no annotated etymology data) this will likely become a **lexicon heuristic** — Arabic-root word lists + the fact that Greek-script characters are already a near-perfect origin signal — deferred until after Milestone 1.

---

## 2. Current plan — Milestone 1: reproduce selective transliteration on Maltese POS

**Why Maltese, and why this first.** Maltese is the closest fully-measurable proxy for Sanna: Arabic-base, heavy non-Arabic superstrate (Italian/English, vs Greek for Sanna), written in non-Arabic script, **but** with a gold UD treebank (**MUDT**) and native model (**BERTu**). Milestone 1 is the smallest step that is both (a) measurable against ground truth and (b) load-bearing for Sanna — it exercises the exact transliteration mechanism the Sanna plan depends on, on the language structurally closest to Sanna. Everything Sanna-specific (etymology heuristic, Greek-script handling, self-training loop) comes *after* the mechanism is proven, so nothing is blocked waiting on it.

**Reference numbers from the paper (for orientation).** Maltese vocabulary is ~32% Arabic / ~53% Italian-Sicilian / ~6% English by dictionary count; Arabic-origin words skew to high frequency and function words, so they cover a large share of *running* text even at 32% of the lexicon. Prior selective pipelines (e.g. Hinglish → transliterate only Hindi-tagged tokens to Devanagari, skip English) are the same shape as ours.

### The baseline ladder (each rung independently testable, reuses the previous rung's code)
- **Rung 0 — raw baseline (the floor).** Fine-tune CAMeL-BERT and XLM-R on raw Maltese POS, no transliteration. Establishes the number every later rung must beat.
- **Rung 1 — full transliteration.** Transliterate *all* tokens to Arabic script, fine-tune, measure. This is the non-selective version the paper reports as suboptimal; we expect it to underperform Rung 2.
- **Rung 2 — selective transliteration (the target).** Only Arabic-origin tokens → Arabic script (using the authors' released etymology classifier + transliterator), non-Arabic tokens stay Latin; fine-tune on the mixed-script result, measure.

**Success criterion:** Rung 2 > Rung 1 > Rung 0 on MUDT gold POS, roughly matching the paper's reported POS results. When this ordering reproduces, the mechanism is proven and you have a measured accuracy figure — the first preliminary result to show David.

---

## 3. First steps (Milestone 1)

1. **Clone resources:** `github.com/MLRS/malti` at the **`2024.eacl`** tag (etymology data, transliteration code, classifier); dataset also under `github.com/mbzuai-nlp/M4`. Get MUDT (`universaldependencies.org/treebanks/mt_mudt`). Get CAMeL-BERT, XLM-R, BERTu weights.
2. **Measurement harness:** hold out the MUDT test split as frozen ground truth; wire up UPOS accuracy + per-tag P/R/F1 + an origin split (Arabic vs non-Arabic).
3. **Rung 0:** fine-tune CAMeL-BERT and XLM-R on raw Maltese; record the floor (3 seeds, averaged).
4. **Rung 1:** apply full transliteration via the repo's tools; fine-tune; measure.
5. **Rung 2:** apply the repo's selective etymology-conditioned transliteration; fine-tune; measure.
6. **Report:** ladder comparison table (+ 3-seed variance). This is the first preliminary result.

---

## 4. Sanna-side gap (parked until Milestone 1 is green)

The method's engine is the etymology decision, which for Maltese rides on annotated etymology data that **will not exist for Sanna**. When we cross to Sanna, this becomes a lexicon/heuristic problem, not a trained classifier — Arabic-root lists plus Greek-script-as-origin-signal. Flagged as the real research gap the project inherits; do not act on it until the mechanism is proven on Maltese.

**Standing item for David:** sign-off on the clitic/article segmentation policy and the homoglyph/spelling normalization table (D13) — blocks scaled Sanna annotation, independent of Milestone 1.

---

## 5. Cypriot Greek pilot (separate, complementary)

Still in play as a second validation angle: rehearses close dialect→standard transfer and the endangered-variety annotation workflow, using the Zenodo CG corpus + Standard Greek treebank + GreekBERT. Distinct from the Maltese leg (which tests the transliteration mechanism with ground-truth measurement). Sequencing: can run in parallel; neither blocks the other.

---

## 6. Key references

- **Micallef, Habash, Borg, Eryani & Bouamor (2024). _Cross-Lingual Transfer from Related Languages: Treating Low-Resource Maltese as Multilingual Code-Switching._ EACL.** — the paper being reproduced. aclanthology.org/2024.eacl-long.61 · code+data: github.com/MLRS/malti (2024.eacl)
- Micallef et al. (2023). _Exploring the Impact of Transliteration on NLP Performance: Treating Maltese as an Arabic Dialect._ CAWL. aclanthology.org/2023.cawl-1.4
- Muller et al. (2021). _When Being Unseen from mBERT is Just the Beginning._ NAACL (background: transliteration-to-model-script). arxiv.org/abs/2010.12858
- Khalifa, Abdul-Mageed & Shaalan (2021). _Self-Training Pre-trained LMs for Zero- and Few-Shot Multi-Dialectal Arabic Sequence Labeling._ EACL. arxiv.org/abs/2101.04758
- Gururangan et al. (2020). _Don't Stop Pretraining._ ACL. arxiv.org/abs/2004.10964
- Guo et al. (2017). _On Calibration of Modern Neural Networks._ ICML. arxiv.org/abs/1706.04599
- Inoue et al. (2021). _The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models_ (CAMeLBERT). WANLP. arxiv.org/abs/2103.06678
- Micallef et al. (2022). _Pre-training Data Quality and Quantity for a Low-Resource Language_ (BERTu/Maltese). DeepLo. arxiv.org/abs/2205.10517
- Conneau et al. (2020). _Unsupervised Cross-lingual Representation Learning at Scale_ (XLM-R). ACL. arxiv.org/abs/1911.02116
- Garrette & Baldridge (2013). _Learning a Part-of-Speech Tagger from Two Hours of Annotation._ NAACL. aclanthology.org/N13-1014
- Maltese UD treebank (MUDT): universaldependencies.org/treebanks/mt_mudt · UPOS guidelines: universaldependencies.org/u/pos

---

## 7. Czech v2 rerun protocol (added 2026-08-08)

The Czech v2 study is a reproducibility and evaluation-protocol redesign of the Czech proxy
experiment. It does **not** supersede D1–D21, the Maltese selective-transliteration ladder,
`STEP_1A_RESULTS.md`, or the parked Sanna-side etymology problem. The legacy flat-module v1
workflow remains available as historical evidence; v2 is isolated in the `sanna_tagging`
package and uses a different artifact format.

- **D22. Immutable inputs.** Pin UD Czech PDT-C and UD Slovak SNK to full repository commits,
  pin every configured CoNLL-U file by SHA-256, pin SlovakBERT and XLM-R to full model
  revisions, and bind the canonical YAML, exact runtime Git commit, and dependency-lock hash
  into one run fingerprint. A real run must use committed code; uncommitted v2 code cannot be
  represented by the recorded Git identity.
- **D23. Leakage-safe data precedence.** Define sentence identity as the SHA-256 of the exact
  ordered FORM sequence. Preserve official Czech test order, remove test duplicates from dev
  and train, then remove retained dev duplicates from train. Ordinary preparation may parse
  test IDs and FORM only and may never serialize UPOS.
- **D24. Fixed development roles.** Deterministically cap Czech dev at 3,000 sentences and
  partition it once as 80% selection, 10% calibration fit, and 10% calibration assessment.
  Selection and early stopping use only `selection`; token temperature and the sentence
  success model fit only on `calibration_fit`; `calibration_assessment` reports held-out
  calibration and cannot change the model or threshold policy.
- **D25. Fixed budget design.** Evaluate 50, 100, 200, 400, and 800 nested Czech gold
  sentences with seeds 0, 1, and 2. Every configuration uses complete overflow-safe word
  alignment, sentence-balanced training loss, and best-checkpoint restoration. Select the
  smallest budget within 0.01 mean sentence-≥98 rate and 0.005 mean token accuracy of the
  best three-seed budget.
- **D26. Predeclared model arms.** Compare direct SlovakBERT, an XLM-R reference, Czech LAPT,
  Slovak UPOS transfer, and Czech-LAPT-plus-Slovak-transfer. Czech LAPT receives cleaned Czech
  training FORM only and uses an internal training-only holdout. Slovak transfer uses Slovak
  train/dev only. Each transition to Czech supervised refinement constructs a fresh optimizer
  and scheduler; optimizer state never crosses adaptation boundaries.
- **D27. Adaptation non-inferiority and ranking.** Reject an adapted arm when its mean Czech
  selection token accuracy is more than 0.002 below direct SlovakBERT. Rank remaining arms by
  mean sentence-≥98 rate, token accuracy, exact match, lower adaptation complexity, then
  stable arm name. This deterministic policy is fixed before test access.
- **D28. Three-seed ensemble.** Freeze all three checkpoints for the winning arm and ensemble
  by an unweighted arithmetic mean of aligned word-probability vectors. Do not select a
  single lucky seed and do not learn ensemble weights on test data.
- **D29. Sentence-level calibration target.** Fit one token temperature and an L2 logistic
  sentence-success predictor using length, expected errors, low-confidence fraction, minimum
  confidence, and ensemble disagreement. Define sentence success literally as
  `100 * correct >= 98 * token_count`; report risk–coverage and accepted high-quality yield at
  the fixed 1%, 2%, and 5% predicted failure-risk thresholds.
- **D30. Manifest-only resume.** A stage is resumable only from a final commit manifest whose
  fingerprint, inputs, parameters, output paths, sizes, and SHA-256 values match. Partial or
  altered state fails closed. Stage handoff paths are relative to the fingerprinted run so a
  committed Kaggle output can be verified read-only and copied to a new writable session.
- **D31. Frozen draft before official test.** `report-draft` has no test-gold parameter and
  freezes the selected budget, winning arm, three checkpoint manifests and hashes, ensemble,
  calibration, and selection evidence in `selection.lock.json`. Optional throughput
  benchmarking is non-selective and cannot affect that lock.
- **D32. Explicit one-shot finalization.** Official test evaluation requires the exact lock
  committed by the matching draft and the configured pinned test bytes. Seal an attempt
  marker before opening gold; any success or failure consumes the run. CI, development,
  notebook default execution, and draft reporting must never invoke this stage.

---

## 8. Czech v2 execution record (added 2026-08-12)

The protocol in section 7 has now been executed end to end on Kaggle. **Run `cs-rerun-v2`,
fingerprint `30a9fbaa373488ecfeec`, code commit `abb04f9`.** The one-shot finalization of
D32 is therefore **consumed for this fingerprint**; nothing in this run may be re-finalized.
Artefacts: [results/cs_v2/](results/cs_v2/).

**Outcome.** Selected budget 800; winning arm `slovakbert-czech-lapt-slovak-upos`; frozen
three-seed unweighted ensemble on the 20,187-sentence official PDT-C test set:

| System | Token accuracy | Sentence ≥98% | Exact match |
|---|---:|---:|---:|
| Seeds 0/1/2 | 0.977175 / 0.977447 / 0.976937 | 0.735771 / 0.738396 / 0.735077 | 0.733145 / 0.736365 / 0.732600 |
| Ensemble | 0.978944 | 0.754644 | 0.751771 |

The predeclared ranking policy of D27 selected a *smaller* base model (SlovakBERT) over the
XLM-R reference arm, and it beats Czech v1's 0.9759 token accuracy — roughly 12.5% of v1's
remaining token error removed. Every individual seed already clears v1, so the gain is not
ensemble luck. `adapt` cost 5 h 01 m of the 12 h Kaggle commit ceiling; `report-draft` 238.5 s;
`finalize-test` 1114.8 s on GPU T4 x2 (no training — inference, ensembling, and the frozen
calibration bundle only).

- **D33. Cross-version comparisons are made on token accuracy, never on coverage.** v1
  reported *token-level* coverage; v2 defines success *per sentence* as
  `100 * correct >= 98 * token_count` (D29). A v2 coverage figure that looks worse than v1's
  is a change of yardstick, not a regression. Only token accuracy is comparable across the
  two protocols, and any write-up must say which yardstick it is quoting.
- **D34. Verify the gold hash before the attempt marker, not after.** `finalize_official_test`
  seals the marker before it checksums gold (D32), so dispatching the wrong file spends the
  single attempt on a file that was never eligible. The Kaggle frontend therefore verifies
  the official CoNLL-U against `config.data.czech.test[0].sha256` *before* dispatch, while
  aborting is still free. Confirmed on this run: 39,742,361 bytes, SHA-256 `f5c1a7ee…`,
  byte-identical to the pinned value.
- **D35. Recover the gold path by filename when the platform mounts it elsewhere, but never
  from resumed run state.** Kaggle mounts plain datasets at
  `/kaggle/input/datasets/<user>/<slug>/<file>`, not the documented `/kaggle/input/<slug>/<file>`,
  so a correctly declared `OFFICIAL_TEST_PATH` can still miss (it did — v6 died on
  `FileNotFoundError`). Recovery searches `/kaggle/input` by filename, excludes the
  `notebooks/` subtree because a `.conllu` under there is resumed run state rather than gold,
  requires exactly one match, and then still passes through the D34 hash check.
- **D36. Trim the runtime image; never widen the lock to accommodate it.** Kaggle preinstalls
  torchvision/torchaudio built against a newer torch than `requirements-kaggle.lock` pins,
  which breaks `import transformers`. Neither package is used, so the frontend uninstalls
  them and asserts they are gone. Relaxing the lock would change its SHA-256 and therefore
  the run fingerprint (D22) — the lock stays authoritative and the environment is corrected
  to match it. Notebook content is deliberately outside the fingerprint, so frontend fixes
  like this and D35 are fingerprint-neutral and do not invalidate a run.
- **D37. Known limitation, deferred to v3: the low-risk tail of the calibration is
  over-conservative.** At a 1% risk budget the mean predicted failure risk is 4.95% against
  an empirical 0.94%, so the D29 fixed thresholds accept **0 sentences at 1% and 2%** and only
  266 (1.3% coverage) at 5%. The risk–coverage curve itself is well-behaved and the ranking is
  unaffected — the sentence-success model is simply miscalibrated where it matters most for
  the human-in-the-loop routing of D7. This must be fixed by a **new fingerprint**, not by
  re-finalizing `cs-rerun-v2`.
