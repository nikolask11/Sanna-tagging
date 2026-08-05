import gc
import json
import time

import torch

from data_utils import XLMR, load_split, label_list_from, seed_pool_split
from modeling import encode, train_model, evaluate
from results_io import read_rows, log_runtime
from runpaths import results_dir
from selftrain import merge_silver, silverize


def best_config(budget_rows, st_rows, chosen_size):
    cands = []
    for r in budget_rows:
        cands.append({"model": r["model"], "seed_size": int(r["seed_size"]),
                      "random_seed": int(r["random_seed"]), "round": None,
                      "dev_acc": float(r["upos_acc"])})
    for r in st_rows:
        cands.append({"model": XLMR, "seed_size": chosen_size,
                      "random_seed": int(r["random_seed"]), "round": int(r["round"]),
                      "dev_acc": float(r["upos_acc"])})
    return max(cands, key=lambda c: c["dev_acc"])


def eval_on_test(cfg):
    guard = results_dir() / "test_result.json"
    if guard.exists():
        return json.loads(guard.read_text())
    from transformers import AutoTokenizer
    train = load_split("train")
    test = load_split("test")
    labels = label_list_from(train)
    label2id = {t: i for i, t in enumerate(labels)}
    id2label = {i: t for t, i in label2id.items()}
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    pad_id = tok.pad_token_id
    gold, pool, _, _ = seed_pool_split(train, cfg["seed_size"], cfg["random_seed"])
    feats = encode(gold, tok, label2id)
    if cfg["round"]:
        merged = merge_silver(cfg["random_seed"], cfg["round"])
        silver_sents, _ = silverize(pool, merged)
        feats = feats + encode(silver_sents, tok, label2id)
    model = train_model(cfg["model"], feats, len(labels), cfg["random_seed"], pad_id)
    acc, per_tag = evaluate(model, encode(test, tok, label2id), test, id2label, pad_id)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {"config": cfg, "test_upos_acc": round(acc, 4), "per_tag_f1": per_tag}
    guard.write_text(json.dumps(result, indent=2))
    return result


def budget_table(budget_rows):
    by_key = {}
    for r in budget_rows:
        by_key.setdefault((r["model"], int(r["seed_size"])), []).append(float(r["upos_acc"]))
    lines = ["| Model | Gold sentences | Mean accuracy | Min | Max |",
             "|---|---|---|---|---|"]
    for (m, s), v in sorted(by_key.items()):
        lines.append(f"| {m} | {s} | {sum(v)/len(v):.4f} | {min(v):.4f} | {max(v):.4f} |")
    return "\n".join(lines), by_key


def selftrain_table(st_rows):
    by_round = {}
    for r in st_rows:
        by_round.setdefault(int(r["round"]), []).append(
            (float(r["upos_acc"]), int(r["n_silver_tokens"])))
    lines = ["| Round | Runs | Mean silver tokens | Mean dev accuracy |",
             "|---|---|---|---|"]
    for rnd, v in sorted(by_round.items()):
        lines.append(f"| {rnd} | {len(v)} | {sum(x[1] for x in v)/len(v):.0f} "
                     f"| {sum(x[0] for x in v)/len(v):.4f} |")
    return "\n".join(lines), by_round


def make_plots(budget_rows, cov_rows, params):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, by_key = budget_table(budget_rows)
    xl = sorted((s, v) for (m, s), v in by_key.items() if m == XLMR)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [s for s, _ in xl]
    means = [sum(v) / len(v) for _, v in xl]
    ax.errorbar(xs, means,
                yerr=[[m - min(v) for m, (_, v) in zip(means, xl)],
                      [max(v) - m for m, (_, v) in zip(means, xl)]],
                fmt="o-", capsize=3, label="XLM-R base")
    for (m, s), v in by_key.items():
        if m != XLMR:
            ax.errorbar([s], [sum(v) / len(v)],
                        yerr=[[sum(v) / len(v) - min(v)], [max(v) - sum(v) / len(v)]],
                        fmt="s", capsize=3, label="CAMeLBERT-mix")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.set_xlabel("gold sentences (log scale)")
    ax.set_ylabel("UPOS accuracy on dev")
    ax.set_title("Annotation budget curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir() / "budget_curve.png", dpi=150)
    plt.close(fig)

    cov = [(float(r["token_coverage"]), float(r["accuracy_of_accepted"]), float(r["threshold"]))
           for r in cov_rows if r["accuracy_of_accepted"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([c for c, _, _ in cov], [a for _, a, _ in cov], "-")
    if params.get("tau_98") is not None:
        ax.plot([params["coverage_at_tau_98"]], [0.98], "r*", markersize=12,
                label=f"tau_98 = {params['tau_98']}")
        ax.legend()
    ax.set_xlabel("fraction of tokens auto-accepted (coverage)")
    ax.set_ylabel("accuracy of accepted tokens")
    ax.set_title("Coverage vs accuracy of accepted tokens")
    fig.tight_layout()
    fig.savefig(results_dir() / "coverage_curve.png", dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    rd = results_dir()
    budget_rows = read_rows(rd / "budget.csv")
    cov_rows = read_rows(rd / "coverage.csv")
    st_rows = read_rows(rd / "selftrain.csv")
    if not (budget_rows and cov_rows and st_rows):
        raise RuntimeError("missing results: run notebooks 02, 03 and 04 first")
    params = json.loads((rd / "derived_params.json").read_text())
    runtimes = read_rows(rd / "runtimes.csv")
    problems = (rd / "problems.log").read_text().splitlines() if (rd / "problems.log").exists() else []

    cfg = best_config(budget_rows, st_rows, params["chosen_seed_size"])
    test_result = eval_on_test(cfg)
    make_plots(budget_rows, cov_rows, params)

    b_table, by_key = budget_table(budget_rows)
    s_table, by_round = selftrain_table(st_rows)
    total_h = sum(float(r["seconds"]) for r in runtimes) / 3600
    n_runs = len(budget_rows) + len(st_rows) + 2
    tau = params.get("tau_98")
    cov98 = params.get("coverage_at_tau_98")
    chosen = params["chosen_seed_size"]
    r0 = by_round.get(0, [])
    best_round = max(by_round, key=lambda k: sum(x[0] for x in by_round[k]) / len(by_round[k])) if by_round else 0
    r0_mean = sum(x[0] for x in r0) / len(r0) if r0 else float("nan")
    best_mean = sum(x[0] for x in by_round[best_round]) / len(by_round[best_round]) if by_round else float("nan")
    st_delta = best_mean - r0_mean
    cfg_desc = (f"{cfg['model']}, {cfg['seed_size']} gold sentences, random seed "
                f"{cfg['random_seed']}" + (f", self-training round {cfg['round']}" if cfg["round"] else ""))

    def size_mean(size):
        v = by_key.get((XLMR, size))
        return f"{sum(v) / len(v):.3f}" if v else "n/a"

    mean_at_50, mean_at_800 = size_mean(50), size_mean(800)

    if tau is not None:
        headline = (f"With {chosen} annotated sentences, the model can tag "
                    f"**{cov98:.0%} of all tokens at 98% accuracy or better** "
                    f"(confidence threshold {tau}). A human would only need to review the remaining "
                    f"{1 - cov98:.0%}.")
        tau_line = (f"the lowest confidence threshold whose accepted tokens are at least 98% correct is "
                    f"**tau_98 = {tau}**, and **{cov98:.1%}** of dev tokens clear it.")
    else:
        headline = (f"No confidence threshold reached 98% accuracy on accepted tokens; the best achieved was "
                    f"{params['best_accepted_accuracy']:.4f} at threshold {params['best_accuracy_threshold']} "
                    f"(coverage {params['coverage_at_best']:.1%}).")
        tau_line = ("no threshold up to 0.99 reached 98% accuracy on accepted tokens. The best accepted-token "
                    f"accuracy was {params['best_accepted_accuracy']:.4f} at threshold "
                    f"{params['best_accuracy_threshold']}. Self-training therefore used that threshold instead.")

    problems_md = "\n".join(f"- `{p}`" for p in problems) if problems else "- (no runtime problems were logged)"

    report = f"""# Milestone 1 report: how much annotation does the pipeline need?

*Generated {time.strftime('%Y-%m-%d')} from the result files in `results/`. Written for a reader
without a machine-learning background; every technical term is explained where it first appears.*

## Headline

{headline}

## 1. What was run

We simulated the Sanna situation on Maltese: the MUDT treebank (a corpus of Maltese sentences where
every word already has a human-verified part-of-speech tag) was transliterated in full into Arabic
script, and we then *hid* most of its tags and pretended only a small "seed" of sentences was
annotated. Because the hidden tags still exist, we can measure exactly how well the system recovers
them — something impossible on a genuinely unannotated corpus.

The tags are **UPOS** tags: the Universal Part-of-Speech inventory (NOUN, VERB, ADJ, and 14 others)
used by the Universal Dependencies project. The model is **XLM-RoBERTa base**, a neural network
pre-trained on text in 100 languages, which we **fine-tune**: continue training it briefly on our
small annotated seed so it learns to output UPOS tags. As a comparison point closer to the Sanna
case, **CAMeLBERT-mix** (a model pre-trained on Arabic, including dialects) was run at seed size 400.

Configurations run, each repeated with 3 different random initialisations ("random seeds") because
results vary run-to-run at small data sizes:

- Budget curve: seed sizes 50, 100, 200, 400, 800 sentences ({len(budget_rows)} training runs total,
  including CAMeLBERT at size 400).
- Calibration: one further run at the chosen seed size ({chosen}).
- Self-training: {len(st_rows)} runs (up to 4 rounds x 3 seeds).
- One final run for the test-set evaluation.

Total: {n_runs} training runs, about **{total_h:.1f} hours** of measured compute on a free
Colab T4 GPU. Active learning (planned step 5) was skipped by instruction.

## 2. Results

### 2.1 Annotation budget curve

Accuracy is the fraction of tokens (words) whose predicted tag matches the human tag, measured on
the *dev set* — a held-out portion of the treebank never used for training.

{b_table}

![budget curve](budget_curve.png)

The chosen seed size for all later stages was **{chosen}** — the smallest size whose mean accuracy
comes within 1 point of the best observed, i.e. where the curve begins to plateau.

### 2.2 Calibration and auto-accept coverage

For each tagged token the model also outputs a **confidence** — its own probability that the tag is
right. If confidences are trustworthy ("calibrated"), we can auto-accept every token above a
threshold and only send the rest to a human. Sweeping thresholds from 0.50 to 0.99:
{tau_line}

**Coverage** means: of all tokens in the dev set, the fraction whose confidence clears the
threshold and is therefore accepted without review.

![coverage curve](coverage_curve.png)

The reliability diagram (`calibration.png`) shows, for 10 bins of confidence, whether tokens the
model calls e.g. 90% certain are actually right 90% of the time; points below the diagonal mean
over-confidence.

### 2.3 Self-training

**Self-training** uses the model's own high-confidence predictions on unannotated text as extra
("silver") training data: tag the unlabelled pool, keep tokens above the threshold, retrain from
the original pre-trained checkpoint on gold + silver, repeat. Round 0 is the gold-only baseline.

{s_table}

Mean dev accuracy moved from {r0_mean:.4f} (round 0) to {best_mean:.4f} at its best round
(round {best_round}), a change of {st_delta:+.4f}.

### 2.4 Final test-set result

The single best configuration by dev accuracy — {cfg_desc} — was retrained once and evaluated once
on the frozen test set (untouched until this step, so this number is an honest estimate):

**Test UPOS accuracy: {test_result['test_upos_acc']:.4f}**

## 3. What worked

- Fine-tuning on tiny seeds works at all: {mean_at_50} mean accuracy from just 50 sentences, rising to {mean_at_800} at 800.
- The budget curve flattens by {chosen} sentences: beyond that, each extra annotated sentence buys
  little dev accuracy. That is the practically-sized annotation ask for David.
- {"Auto-accepting at tau_98 = " + str(tau) + f" covers {cov98:.1%} of tokens at >=98% accuracy — the headline pipeline property." if tau is not None else "The confidence signal was informative but never reached the 98% bar; see section 4."}
- Self-training {"helped" if st_delta > 0.002 else "did not meaningfully help" if st_delta > -0.002 else "hurt"}: {st_delta:+.4f} mean accuracy versus the gold-only baseline. {"It stays in the Sanna pipeline." if st_delta > 0.002 else "Per the ablation-ladder rule, it is a candidate for cutting from the Sanna pipeline."}

## 4. What went wrong

Everything logged automatically during the runs, verbatim:

{problems_md}

Deviations from the original plan and other honesty items:

- **Transliteration is the deterministic variant, not the paper's full pipeline.** The MLRS
  (2024 EACL) non-deterministic transliteration needs ranking language models hosted in an external
  Google Drive folder whose download quota was exhausted; the deterministic character mapping was
  used instead for the whole corpus (decided and documented before this study, see
  `STEP_1A_RESULTS.md`). All numbers here are for that variant.
- **Threshold and seed size were chosen on the dev set**, the same set used to report dev accuracy.
  Only the final test number is free of this mild optimism.
- **Trained models are deliberately discarded** after each run (Drive space); reproducing a row
  means retraining, which on a GPU is not bit-for-bit deterministic — expect small differences.
- **Active learning (step 5 of the original plan) was skipped entirely** by instruction, so this
  study says nothing about whether an uncertainty-ordered review queue beats corpus-order
  annotation.
- Silver labels from later self-training rounds overwrite earlier ones for the same token when the
  rounds disagree; the accumulation rule in the spec did not define this case.

## 5. Threats to validity

- **MUDT is a clean, curated treebank** with standardised spelling and consistent annotation. The
  Sanna corpus is none of those things. Every number above is an optimistic bound, not a prediction.
- **Maltese is not Sanna.** The transliteration quality, the tag distribution, and the match to
  CAMeLBERT's pre-training data will all differ.
- **One test evaluation of one configuration** — the test number has no error bar.
- **Dev-set reuse**: seed size and threshold were tuned on the same dev set that produced the curves,
  so dev numbers are slightly flattered; coverage at tau_98 on truly new text may be a little lower.
- **Sentence-length truncation**: sentences longer than 256 subword pieces are cut off (a handful of
  tokens at most); their tail tokens are neither trained on nor evaluated.

## 6. Open questions for the next session

- Does an uncertainty-based review queue (active learning) beat random annotation order? Skipped
  here; it is the remaining unanswered question from the original plan.
- Would continued pre-training on raw transliterated Maltese (plan stage S1) lift the small-seed end
  of the budget curve?
- The error-asymmetry probe (transliterate-vs-leave for unclassifiable tokens) remains unrun.
- Is CAMeLBERT's gap to XLM-R at seed 400 stable across seed sizes? Only 400 was tested.
- How much does coverage at tau_98 drop on out-of-domain text? MUDT's genre mix is narrow.
"""
    (rd / "REPORT.md").write_text(report)
    log_runtime("05", "report", time.time() - t0)
    print(f"REPORT.md written to {rd} (test acc {test_result['test_upos_acc']:.4f})")
