# Sanna POS Project — Decisions & Current Plan

_Running decision log. Append a dated line whenever a choice is made or changed, with who made it and one sentence of why. This doubles as the backbone of the paper's methodology section._

_Last updated: 2026-08-01 · Maintained by Nikolas_

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
