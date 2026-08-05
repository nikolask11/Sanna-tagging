import gc
import json
import time

import numpy as np
import torch

from data_utils import (SEED_SIZES, XLMR, load_split, label_list_from,
                        seed_pool_split)
from modeling import encode, train_model, predict
from results_io import append_row, read_rows, log_runtime, log_problem
from runpaths import results_dir, cache_dir

THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(50)]


def choose_seed_size(budget_rows):
    accs = {}
    for r in budget_rows:
        if r["model"] == XLMR:
            accs.setdefault(int(r["seed_size"]), []).append(float(r["upos_acc"]))
    means = {s: sum(v) / len(v) for s, v in accs.items()}
    best = max(means.values())
    return min(s for s in SEED_SIZES if s in means and means[s] >= best - 0.01)


def dev_confidences(chosen):
    npz_path = cache_dir() / f"calib_preds_size{chosen}.npz"
    if npz_path.exists():
        d = np.load(npz_path)
        return d["conf"], d["correct"]
    from transformers import AutoTokenizer
    train = load_split("train")
    dev = load_split("dev")
    labels = label_list_from(train)
    label2id = {t: i for i, t in enumerate(labels)}
    id2label = {i: t for t, i in label2id.items()}
    tok = AutoTokenizer.from_pretrained(XLMR)
    gold, _, _, _ = seed_pool_split(train, chosen, 0)
    model = train_model(XLMR, encode(gold, tok, label2id), len(labels), 0, tok.pad_token_id)
    preds = predict(model, encode(dev, tok, label2id), id2label, tok.pad_token_id)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    conf, correct = [], []
    for sent, pred in zip(dev, preds):
        for widx, ptag, c in pred:
            conf.append(c)
            correct.append(1 if ptag == sent["upos"][widx] else 0)
    conf, correct = np.array(conf), np.array(correct)
    np.savez(npz_path, conf=conf, correct=correct)
    return conf, correct


def reliability_png(conf, correct, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 1, 11)
    mids, accs, weights = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.sum() == 0:
            continue
        mids.append(conf[m].mean())
        accs.append(correct[m].mean())
        weights.append(m.sum())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(mids, accs, "o-", label="model")
    ax.set_xlabel("mean predicted confidence (bin)")
    ax.set_ylabel("actual accuracy (bin)")
    ax.set_title("Reliability diagram, dev set, 10 bins")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    budget_rows = read_rows(results_dir() / "budget.csv")
    if not budget_rows:
        raise RuntimeError("budget.csv missing or empty: run 02_budget_curve first")
    chosen = choose_seed_size(budget_rows)
    cov_path = results_dir() / "coverage.csv"
    png_path = results_dir() / "calibration.png"
    par_path = results_dir() / "derived_params.json"
    if cov_path.exists() and png_path.exists() and par_path.exists():
        print(f"coverage.csv: {len(read_rows(cov_path))} rows (already complete)")
        return
    t0 = time.time()
    conf, correct = dev_confidences(chosen)
    if cov_path.exists():
        cov_path.unlink()
    rows = []
    for t in THRESHOLDS:
        m = conf >= t
        coverage = float(m.mean())
        acc = float(correct[m].mean()) if m.sum() else None
        rows.append({"threshold": t, "token_coverage": round(coverage, 4),
                     "accuracy_of_accepted": round(acc, 4) if acc is not None else ""})
        append_row(cov_path, ["threshold", "token_coverage", "accuracy_of_accepted"], rows[-1])
    reliability_png(conf, correct, png_path)
    ok = [r for r in rows if r["accuracy_of_accepted"] != "" and r["accuracy_of_accepted"] >= 0.98]
    params = {"chosen_seed_size": chosen}
    if ok:
        best = min(ok, key=lambda r: r["threshold"])
        params["tau_98"] = best["threshold"]
        params["coverage_at_tau_98"] = best["token_coverage"]
    else:
        scored = [r for r in rows if r["accuracy_of_accepted"] != ""]
        top = max(scored, key=lambda r: r["accuracy_of_accepted"])
        params["tau_98"] = None
        params["coverage_at_tau_98"] = None
        params["best_accepted_accuracy"] = top["accuracy_of_accepted"]
        params["best_accuracy_threshold"] = top["threshold"]
        params["coverage_at_best"] = top["token_coverage"]
        log_problem(f"03: no threshold reached 0.98 accepted accuracy; "
                    f"best {top['accuracy_of_accepted']} at {top['threshold']}")
    par_path.write_text(json.dumps(params, indent=2))
    log_runtime("03", f"size{chosen}", time.time() - t0)
    print(f"coverage.csv: {len(rows)} rows; derived_params.json written (chosen_seed_size={chosen})")
