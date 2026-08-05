import gc
import json
import time

import torch

from runpaths import STUDY, results_dir
from data_utils import (PRIMARY_MODEL, SELFTRAIN_POOL_CAP, load_split,
                        label_list_from, seed_pool_split, pool_subsample)
from modeling import encode, train_model, evaluate
from results_io import read_rows, log_runtime
from selftrain import merge_silver, silverize


def best_config(budget_rows, st_rows, chosen_size):
    cands = []
    for r in budget_rows:
        cands.append({"model": r["model"], "seed_size": int(r["seed_size"]),
                      "random_seed": int(r["random_seed"]), "round": None,
                      "dev_acc": float(r["upos_acc"])})
    for r in st_rows:
        size = int(r["seed_size"]) if r.get("seed_size") else chosen_size
        cands.append({"model": PRIMARY_MODEL[STUDY], "seed_size": size,
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
        pool = pool_subsample(pool, cfg["random_seed"])
        size_arg = cfg["seed_size"] if STUDY == "cs" else None
        merged = merge_silver(cfg["random_seed"], cfg["round"], size_arg)
        silver_sents, _ = silverize(pool, merged)
        feats = feats + encode(silver_sents, tok, label2id)
    target = 500 if len(feats) < 2000 else 1500
    model = train_model(cfg["model"], feats, len(labels), cfg["random_seed"], pad_id,
                        target_steps=target)
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


def selftrain_tables(st_rows, chosen_size):
    groups = {}
    for r in st_rows:
        size = int(r["seed_size"]) if r.get("seed_size") else chosen_size
        groups.setdefault(size, {}).setdefault(int(r["round"]), []).append(
            (float(r["upos_acc"]), int(r["n_silver_tokens"])))
    out = []
    for size in sorted(groups):
        lines = [f"**Seed size {size}:**", "",
                 "| Round | Runs | Mean silver tokens | Mean dev accuracy |",
                 "|---|---|---|---|"]
        for rnd, v in sorted(groups[size].items()):
            lines.append(f"| {rnd} | {len(v)} | {sum(x[1] for x in v)/len(v):.0f} "
                         f"| {sum(x[0] for x in v)/len(v):.4f} |")
        out.append("\n".join(lines))
    return "\n\n".join(out), groups


def st_summary(groups):
    parts = []
    for size in sorted(groups):
        by_round = groups[size]
        r0 = sum(x[0] for x in by_round.get(0, [(0, 0)])) / max(1, len(by_round.get(0, [])))
        best_rnd = max(by_round, key=lambda k: sum(x[0] for x in by_round[k]) / len(by_round[k]))
        best = sum(x[0] for x in by_round[best_rnd]) / len(by_round[best_rnd])
        parts.append((size, r0, best, best_rnd, best - r0))
    return parts


def make_plots(budget_rows, cov_rows, params, primary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, by_key = budget_table(budget_rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    models = sorted({m for m, _ in by_key})
    for m in models:
        pts = sorted((s, v) for (mm, s), v in by_key.items() if mm == m)
        xs = [s for s, _ in pts]
        means = [sum(v) / len(v) for _, v in pts]
        yerr = [[mn - min(v) for mn, (_, v) in zip(means, pts)],
                [max(v) - mn for mn, (_, v) in zip(means, pts)]]
        style = "o-" if m == primary else "s--"
        ax.errorbar(xs, means, yerr=yerr, fmt=style, capsize=3, label=m)
    ax.set_xscale("log")
    all_x = sorted({s for _, s in by_key})
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_x)
    ax.set_xlabel("gold sentences (log scale)")
    ax.set_ylabel("UPOS accuracy on dev")
    ax.set_title("Annotation budget curve")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(results_dir() / "budget_curve.png", dpi=150)
    plt.close(fig)

    cov = [(float(r["token_coverage"]), float(r["accuracy_of_accepted"]))
           for r in cov_rows if r["accuracy_of_accepted"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([c for c, _ in cov], [a for _, a in cov], "-")
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


INTRO = {
    "mt": """We simulated the Sanna situation on Maltese: the MUDT treebank (a corpus of Maltese sentences where
every word already has a human-verified part-of-speech tag) was transliterated in full into Arabic
script, and we then *hid* most of its tags and pretended only a small "seed" of sentences was
annotated. Because the hidden tags still exist, we can measure exactly how well the system recovers
them — something impossible on a genuinely unannotated corpus.""",
    "cs": """We simulated the Sanna situation on Czech, at real scale this time. Czech was chosen because it has
a very large gold-annotated treebank (UD Czech PDT-C; we use a ~100,000-sentence portion) and a
closely related higher-resourced sibling language, Slovak, with a published pretrained model
(SlovakBERT) — mirroring Sanna's relationship to Arabic and CAMeLBERT. We *hid* the tags of almost
the entire corpus and pretended only a small "seed" of sentences was annotated. Unlike the earlier
Maltese pilot (1,123 training sentences), the withheld pool here is ~100,000 sentences, so the
seed-to-pool ratio matches the real Sanna deployment. Czech and Slovak share the Latin script, so
this study isolates the *scale* question; the script/transliteration cost was measured separately
on Maltese.""",
}


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
    primary = PRIMARY_MODEL[STUDY]
    chosen = params["chosen_seed_size"]

    cfg = best_config(budget_rows, st_rows, chosen)
    test_result = eval_on_test(cfg)
    make_plots(budget_rows, cov_rows, params, primary)

    b_table, by_key = budget_table(budget_rows)
    s_tables, groups = selftrain_tables(st_rows, chosen)
    total_h = sum(float(r["seconds"]) for r in runtimes) / 3600
    tau = params.get("tau_98")
    cov98 = params.get("coverage_at_tau_98")
    cfg_desc = (f"{cfg['model']}, {cfg['seed_size']} gold sentences, random seed "
                f"{cfg['random_seed']}" + (f", self-training round {cfg['round']}" if cfg["round"] else ""))

    def size_mean(model, size):
        v = by_key.get((model, size))
        return f"{sum(v) / len(v):.3f}" if v else "n/a"

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

    st_lines = []
    for size, r0, best, best_rnd, delta in st_summary(groups):
        verdict = "helped" if delta > 0.002 else ("did not meaningfully help" if delta > -0.002 else "hurt")
        st_lines.append(f"- At seed size {size}: {verdict} — round-0 baseline {r0:.4f}, best round "
                        f"({best_rnd}) {best:.4f}, change {delta:+.4f}.")
    st_block = "\n".join(st_lines)

    problems_md = "\n".join(f"- `{p}`" for p in problems) if problems else "- (no runtime problems were logged)"

    if STUDY == "cs":
        models_note = ("The primary model is **SlovakBERT** (pretrained only on Slovak, the related language — "
                       "the analogue of using an Arabic model for Sanna). **XLM-RoBERTa base** (pretrained on "
                       "100 languages, Czech included) is the comparison baseline; note XLM-R has seen Czech, "
                       "so it is a *ceiling-ish* reference, like BERTu was for Maltese.")
        deviations = """- **The unlabelled pool for self-training was capped at 10,000 sentences** (a fixed random
  subsample of the ~100k withheld sentences, per random seed) to keep each round's tagging and
  retraining inside a free Colab session. Coverage numbers are computed on the dev set, not the
  capped pool.
- **Dev evaluations use a fixed 3,000-sentence subsample** of the (very large) PDT-C dev split;
  the final test evaluation uses the full frozen test split.
- **Only three PDT-C training sections (lt, la, ca) were used** (~100k sentences), not the full
  treebank; genre composition therefore differs from full PDT-C.
- **No transliteration step in this study** — Czech and Slovak share the Latin script. Scale and
  script were deliberately separated: script cost was measured on Maltese, scale is measured here.
- Silver labels from later self-training rounds overwrite earlier ones for the same token when
  rounds disagree; the accumulation rule in the spec did not define this case."""
        open_qs = """- Combine the two measured effects: does self-training still behave the same *after* lossy
  transliteration (the Maltese condition) at Czech scale? That is the full Sanna condition.
- Active learning (uncertainty-ordered review queue) remains untested.
- Does temperature scaling (a one-parameter calibration fix) raise coverage at tau_98?
- Would continued pretraining of SlovakBERT on raw Czech close the gap to XLM-R?
- Replicate on a second pair (e.g. Russian treebank + Ukrainian model) to check pair-specificity."""
    else:
        models_note = ""
        deviations = "(see original Maltese report)"
        open_qs = "(see original Maltese report)"

    report = f"""# Milestone 1{" (Czech scale study)" if STUDY == "cs" else ""} report: how much annotation does the pipeline need?

*Generated {time.strftime('%Y-%m-%d')} from the result files in `{rd.name}/`. Written for a reader
without a machine-learning background; every technical term is explained where it first appears.*

## Headline

{headline}

## 1. What was run

{INTRO[STUDY]}

The tags are **UPOS** tags: the Universal Part-of-Speech inventory (NOUN, VERB, ADJ, and 14 others)
used by the Universal Dependencies project. **Fine-tuning** means continuing the training of a
pretrained neural network briefly on our small annotated seed so it learns to output UPOS tags.
{models_note}

Configurations run, each repeated with 3 different random initialisations ("random seeds"):

- Budget curve: seed sizes 50, 100, 200, 400, 800 sentences ({len(budget_rows)} training runs).
- Calibration: threshold sweeps at seed size(s) {sorted(params.get('per_size', {'?': 0}).keys())}.
- Self-training: {len(st_rows)} runs.
- One final run for the test-set evaluation.

Total: about **{total_h:.1f} hours** of measured compute on a free Colab T4 GPU.
Active learning was skipped by instruction.

## 2. Results

### 2.1 Annotation budget curve

Accuracy is the fraction of tokens whose predicted tag matches the human tag, measured on the
dev set (held-out, never trained on).

{b_table}

![budget curve](budget_curve.png)

The chosen seed size for later stages was **{chosen}** — the smallest size whose mean accuracy comes
within 1 point of the best observed for the primary model.

### 2.2 Calibration and auto-accept coverage

For each tagged token the model outputs a **confidence** — its own probability that the tag is
right. If confidences are trustworthy, we can auto-accept every token above a threshold and only
send the rest to a human. Sweeping thresholds from 0.50 to 0.99 at seed size {chosen}:
{tau_line}

![coverage curve](coverage_curve.png)

The reliability diagram (`calibration.png`) shows whether tokens the model calls e.g. 90% certain
are actually right 90% of the time; points below the diagonal mean over-confidence.

### 2.3 Self-training

**Self-training** uses the model's own high-confidence predictions on unannotated text as extra
("silver") training data: tag the unlabelled pool, keep tokens above the per-size threshold,
retrain from the original pretrained checkpoint on gold + accumulated silver, repeat.
Round 0 is the gold-only baseline.

{s_tables}

{st_block}

### 2.4 Final test-set result

The single best configuration by dev accuracy — {cfg_desc} — was retrained once and evaluated once
on the frozen test set (untouched until this step):

**Test UPOS accuracy: {test_result['test_upos_acc']:.4f}**

## 3. What worked

- Fine-tuning on tiny seeds: {size_mean(primary, 50)} mean accuracy from 50 sentences,
  {size_mean(primary, 200)} at 200, {size_mean(primary, 800)} at 800 ({primary}).
- {"Auto-accepting at tau_98 = " + str(tau) + f" covers {cov98:.1%} of tokens at >=98% accuracy." if tau is not None else "The 98% auto-accept bar was not reached; see section 4."}
{st_block}

## 4. What went wrong / deviations

Logged automatically during the runs, verbatim:

{problems_md}

Deviations and honesty items:

{deviations}

## 5. Threats to validity

- Gold treebanks are clean and consistently annotated; the Sanna corpus will not be. All numbers
  are optimistic bounds, not predictions.
- Seed size and thresholds were chosen on the dev set; only the final test number is free of that
  mild optimism.
- One test evaluation of one configuration — the test number has no error bar.
- Trained models are discarded after each run; GPU retraining is not bit-for-bit reproducible.
{"- Czech/Slovak are closer relatives than Sanna/Arabic may be; transfer from SlovakBERT is likely easier than the Sanna case." if STUDY == "cs" else ""}

## 6. Open questions for the next session

{open_qs}
"""
    (rd / "REPORT.md").write_text(report)
    log_runtime("05", "report", time.time() - t0)
    print(f"REPORT.md written to {rd} (test acc {test_result['test_upos_acc']:.4f})")
