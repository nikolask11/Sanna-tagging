import gc
import json
import time

import numpy as np
import torch

from runpaths import STUDY, results_dir, cache_dir
from data_utils import (SEED_SIZES, PRIMARY_MODEL, load_split, label_list_from,
                        seed_pool_split)
from modeling import encode, train_model, predict
from results_io import append_row, read_rows, log_runtime, log_problem

THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(50)]
SMALL_SIZE = 200


def choose_seed_size(budget_rows, model_name):
    accs = {}
    for r in budget_rows:
        if r["model"] == model_name:
            accs.setdefault(int(r["seed_size"]), []).append(float(r["upos_acc"]))
    means = {s: sum(v) / len(v) for s, v in accs.items()}
    best = max(means.values())
    return min(s for s in SEED_SIZES if s in means and means[s] >= best - 0.01)


def dev_confidences(size, model_name):
    npz_path = cache_dir() / f"calib_preds_size{size}.npz"
    if npz_path.exists():
        d = np.load(npz_path)
        return d["conf"], d["correct"]
    from transformers import AutoTokenizer
    train = load_split("train")
    dev = load_split("dev")
    labels = label_list_from(train)
    label2id = {t: i for i, t in enumerate(labels)}
    id2label = {i: t for t, i in label2id.items()}
    tok = AutoTokenizer.from_pretrained(model_name)
    gold, _, _, _ = seed_pool_split(train, size, 0)
    model = train_model(model_name, encode(gold, tok, label2id), len(labels), 0, tok.pad_token_id)
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


def sweep(conf, correct):
    rows = []
    for t in THRESHOLDS:
        m = conf >= t
        coverage = float(m.mean())
        acc = float(correct[m].mean()) if m.sum() else None
        rows.append({"threshold": t, "token_coverage": round(coverage, 4),
                     "accuracy_of_accepted": round(acc, 4) if acc is not None else ""})
    return rows


def derive(rows, size):
    scored = [r for r in rows if r["accuracy_of_accepted"] != ""]
    ok = [r for r in scored if r["accuracy_of_accepted"] >= 0.98]
    if ok:
        best = min(ok, key=lambda r: r["threshold"])
        return {"tau_98": best["threshold"], "coverage_at_tau_98": best["token_coverage"]}
    top = max(scored, key=lambda r: r["accuracy_of_accepted"])
    log_problem(f"03: size {size}: no threshold reached 0.98 accepted accuracy; "
                f"best {top['accuracy_of_accepted']} at {top['threshold']}")
    return {"tau_98": None, "coverage_at_tau_98": None,
            "best_accepted_accuracy": top["accuracy_of_accepted"],
            "best_accuracy_threshold": top["threshold"],
            "coverage_at_best": top["token_coverage"]}


def reliability_png(conf, correct, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 1, 11)
    mids, accs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.sum() == 0:
            continue
        mids.append(conf[m].mean())
        accs.append(correct[m].mean())
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
    primary = PRIMARY_MODEL[STUDY]
    chosen = choose_seed_size(budget_rows, primary)
    sizes = sorted({chosen} | ({SMALL_SIZE} if STUDY == "cs" else set()))
    cov_path = results_dir() / "coverage.csv"
    png_path = results_dir() / "calibration.png"
    par_path = results_dir() / "derived_params.json"
    if cov_path.exists() and png_path.exists() and par_path.exists():
        params = json.loads(par_path.read_text())
        if all(str(s) in params.get("per_size", {}) for s in sizes):
            print(f"coverage.csv: {len(read_rows(cov_path))} rows (already complete)")
            return
    t0 = time.time()
    per_size = {}
    for size in sizes:
        conf, correct = dev_confidences(size, primary)
        per_size[str(size)] = derive(sweep(conf, correct), size)
        if size == chosen:
            if cov_path.exists():
                cov_path.unlink()
            for row in sweep(conf, correct):
                append_row(cov_path, ["threshold", "token_coverage", "accuracy_of_accepted"], row)
            reliability_png(conf, correct, png_path)
    params = {"chosen_seed_size": chosen, "model": primary, "per_size": per_size}
    params.update(per_size[str(chosen)])
    par_path.write_text(json.dumps(params, indent=2))
    log_runtime("03", f"sizes{sizes}", time.time() - t0)
    print(f"coverage.csv written; derived_params.json for sizes {sizes} (chosen={chosen})")
