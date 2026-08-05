import gc
import json
import time

import torch

from runpaths import STUDY, results_dir, cache_dir
from data_utils import (RANDOM_SEEDS, PRIMARY_MODEL, load_split, label_list_from,
                        seed_pool_split, pool_subsample)
from modeling import encode, train_model, evaluate, predict
from results_io import append_row, read_rows, log_runtime, log_problem

ROUNDS = 3
SMALL_SIZE = 200


def fields():
    base = ["round", "random_seed", "n_silver_tokens", "upos_acc"]
    return ["seed_size"] + base if STUDY == "cs" else base


def get_params():
    par_path = results_dir() / "derived_params.json"
    if not par_path.exists():
        raise RuntimeError("derived_params.json missing: run 03_calibration first")
    return json.loads(par_path.read_text())


def tau_for(params, size):
    entry = params.get("per_size", {}).get(str(size), params)
    tau = entry.get("tau_98")
    if tau is None:
        tau = entry["best_accuracy_threshold"]
        log_problem(f"04: tau_98 null for size {size}; using best-accuracy threshold {tau}")
    return tau


def silver_path(rs, rnd, size=None):
    if STUDY == "mt":
        return cache_dir() / f"silver_rs{rs}_r{rnd}.json"
    return cache_dir() / f"silver_s{size}_rs{rs}_r{rnd}.json"


def merge_silver(rs, upto_round, size=None):
    merged = {}
    for rnd in range(upto_round):
        data = json.loads(silver_path(rs, rnd, size).read_text())
        for sidx, tokmap in data.items():
            merged.setdefault(sidx, {}).update(tokmap)
    return merged


def silverize(pool, merged):
    sents, n_tokens = [], 0
    for sidx, tokmap in merged.items():
        sent = pool[int(sidx)]
        upos = [None] * len(sent["tokens"])
        for widx, tag in tokmap.items():
            upos[int(widx)] = tag
            n_tokens += 1
        sents.append({"tokens": sent["tokens"], "upos": upos})
    return sents, n_tokens


def stopped(accs):
    return len(accs) >= 3 and accs[-1] < accs[-2] < accs[-3]


def row_key(r):
    size = int(r["seed_size"]) if "seed_size" in r and r.get("seed_size") else None
    return (size, int(r["random_seed"]), int(r["round"]))


def main():
    params = get_params()
    chosen = params["chosen_seed_size"]
    sizes = sorted({SMALL_SIZE, chosen}) if STUDY == "cs" else [chosen]
    model_name = PRIMARY_MODEL[STUDY]
    csv_path = results_dir() / "selftrain.csv"
    train = load_split("train")
    dev = load_split("dev")
    labels = label_list_from(train)
    label2id = {t: i for i, t in enumerate(labels)}
    id2label = {i: t for t, i in label2id.items()}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    pad_id = tok.pad_token_id
    dev_feats = encode(dev, tok, label2id)

    for size in sizes:
        tau = tau_for(params, size)
        for rs in RANDOM_SEEDS:
            gold, pool, _, _ = seed_pool_split(train, size, rs)
            pool = pool_subsample(pool, rs)
            gold_feats = encode(gold, tok, label2id)
            pool_feats = None
            all_rows = read_rows(csv_path)
            rows = {int(r["round"]): float(r["upos_acc"]) for r in all_rows
                    if row_key(r)[:2] == ((size if STUDY == "cs" else None), rs)}
            accs = []
            rnd = 0
            while rnd <= ROUNDS:
                if rnd in rows and (silver_path(rs, rnd, size).exists() or rnd == ROUNDS):
                    accs.append(rows[rnd])
                    if stopped(accs):
                        break
                    rnd += 1
                    continue
                t0 = time.time()
                merged = merge_silver(rs, rnd, size)
                silver_sents, n_silver = silverize(pool, merged)
                train_feats = gold_feats + encode(silver_sents, tok, label2id)
                target = 500 if len(train_feats) < 2000 else 1500
                model = train_model(model_name, train_feats, len(labels), rs, pad_id,
                                    target_steps=target)
                acc, _ = evaluate(model, dev_feats, dev, id2label, pad_id)
                if rnd not in rows:
                    row = {"round": rnd, "random_seed": rs,
                           "n_silver_tokens": n_silver, "upos_acc": round(acc, 4)}
                    if STUDY == "cs":
                        row["seed_size"] = size
                    append_row(csv_path, fields(), row)
                    rows[rnd] = acc
                if rnd < ROUNDS:
                    if pool_feats is None:
                        pool_feats = encode(pool, tok, label2id)
                    preds = predict(model, pool_feats, id2label, pad_id)
                    accepted = {}
                    for sidx, pred in enumerate(preds):
                        tokmap = {str(w): t for w, t, c in pred if c >= tau}
                        if tokmap:
                            accepted[str(sidx)] = tokmap
                    silver_path(rs, rnd, size).write_text(json.dumps(accepted))
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log_runtime("04", f"s{size}|rs{rs}|round{rnd}", time.time() - t0)
                accs.append(rows[rnd])
                if stopped(accs):
                    log_problem(f"04: early stop for size={size} rs={rs} after round {rnd}")
                    break
                rnd += 1
    print(f"selftrain.csv: {len(read_rows(csv_path))} rows")
