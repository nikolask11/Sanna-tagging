import gc
import json
import time

import torch

from data_utils import RANDOM_SEEDS, XLMR, load_split, label_list_from, seed_pool_split
from modeling import encode, train_model, evaluate, predict
from results_io import append_row, read_rows, log_runtime, log_problem
from runpaths import results_dir, cache_dir

ROUNDS = 3
FIELDS = ["round", "random_seed", "n_silver_tokens", "upos_acc"]


def get_tau_and_size():
    par_path = results_dir() / "derived_params.json"
    if not par_path.exists():
        raise RuntimeError("derived_params.json missing: run 03_calibration first")
    params = json.loads(par_path.read_text())
    tau = params["tau_98"]
    if tau is None:
        tau = params["best_accuracy_threshold"]
        log_problem(f"04: tau_98 was null; using best-accuracy threshold {tau} instead")
    return tau, params["chosen_seed_size"]


def silver_path(rs, rnd):
    return cache_dir() / f"silver_rs{rs}_r{rnd}.json"


def merge_silver(rs, upto_round):
    merged = {}
    for rnd in range(upto_round):
        data = json.loads(silver_path(rs, rnd).read_text())
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


def main():
    tau, size = get_tau_and_size()
    csv_path = results_dir() / "selftrain.csv"
    train = load_split("train")
    dev = load_split("dev")
    labels = label_list_from(train)
    label2id = {t: i for i, t in enumerate(labels)}
    id2label = {i: t for t, i in label2id.items()}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(XLMR)
    pad_id = tok.pad_token_id
    dev_feats = encode(dev, tok, label2id)

    for rs in RANDOM_SEEDS:
        gold, pool, _, _ = seed_pool_split(train, size, rs)
        gold_feats = encode(gold, tok, label2id)
        pool_feats = None
        rows = {int(r["round"]): float(r["upos_acc"])
                for r in read_rows(csv_path) if int(r["random_seed"]) == rs}
        accs = []
        rnd = 0
        while rnd <= ROUNDS:
            if rnd in rows and (silver_path(rs, rnd).exists() or rnd == ROUNDS):
                accs.append(rows[rnd])
                if stopped(accs):
                    break
                rnd += 1
                continue
            t0 = time.time()
            merged = merge_silver(rs, rnd)
            silver_sents, n_silver = silverize(pool, merged)
            train_feats = gold_feats + encode(silver_sents, tok, label2id)
            model = train_model(XLMR, train_feats, len(labels), rs, pad_id)
            acc, _ = evaluate(model, dev_feats, dev, id2label, pad_id)
            if rnd not in rows:
                append_row(csv_path, FIELDS,
                           {"round": rnd, "random_seed": rs,
                            "n_silver_tokens": n_silver, "upos_acc": round(acc, 4)})
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
                silver_path(rs, rnd).write_text(json.dumps(accepted))
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_runtime("04", f"rs{rs}|round{rnd}", time.time() - t0)
            accs.append(rows[rnd])
            if stopped(accs):
                log_problem(f"04: early stop for rs={rs} after round {rnd} "
                            "(dev accuracy dropped two consecutive rounds)")
                break
            rnd += 1
    print(f"selftrain.csv: {len(read_rows(csv_path))} rows")
