import gc
import json
import time

import torch

from runpaths import STUDY, results_dir
from data_utils import (SEED_SIZES, RANDOM_SEEDS, XLMR, CAMELBERT, SLOVAKBERT,
                        load_split, label_list_from, seed_pool_split)
from modeling import encode, train_model, evaluate
from results_io import append_row, read_rows, log_runtime, log_problem

FIELDS = ["model", "seed_size", "random_seed", "upos_acc", "per_tag_f1_json"]


def run_list():
    runs = []
    if STUDY == "mt":
        runs += [(XLMR, s, rs) for s in SEED_SIZES for rs in RANDOM_SEEDS]
        runs += [(CAMELBERT, 400, rs) for rs in RANDOM_SEEDS]
    else:
        for s in SEED_SIZES:
            for model in (SLOVAKBERT, XLMR):
                runs += [(model, s, rs) for rs in RANDOM_SEEDS]
    return runs


def run_one(model_name, size, rs, train, dev, labels):
    from transformers import AutoTokenizer
    label2id = {t: i for i, t in enumerate(labels)}
    id2label = {i: t for t, i in label2id.items()}
    tok = AutoTokenizer.from_pretrained(model_name)
    pad_id = tok.pad_token_id
    gold, _, _, _ = seed_pool_split(train, size, rs)
    model = train_model(model_name, encode(gold, tok, label2id), len(labels), rs, pad_id)
    acc, per_tag = evaluate(model, encode(dev, tok, label2id), dev, id2label, pad_id)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acc, per_tag


def main():
    if not torch.cuda.is_available():
        print("WARNING: no GPU detected; runs will be very slow")
    csv_path = results_dir() / "budget.csv"
    done = {(r["model"], int(r["seed_size"]), int(r["random_seed"]))
            for r in read_rows(csv_path)}
    train = load_split("train")
    dev = load_split("dev")
    labels = label_list_from(train)
    for model_name, size, rs in run_list():
        if (model_name, size, rs) in done:
            continue
        t0 = time.time()
        first_400 = size == 400 and not any(k[1] == 400 for k in done)
        acc, per_tag = run_one(model_name, size, rs, train, dev, labels)
        append_row(csv_path, FIELDS,
                   {"model": model_name, "seed_size": size, "random_seed": rs,
                    "upos_acc": round(acc, 4),
                    "per_tag_f1_json": json.dumps(per_tag)})
        log_runtime("02", f"{model_name}|{size}|{rs}", time.time() - t0)
        done.add((model_name, size, rs))
        if first_400 and acc < 0.60:
            log_problem(f"F2: first seed-size-400 run ({model_name}, rs={rs}) scored {acc:.3f} < 0.60")
            raise RuntimeError(f"F2: first run at seed size 400 scored {acc:.3f} < 0.60 UPOS "
                               "-> likely label misalignment")
    print(f"budget.csv: {len(read_rows(csv_path))} rows")
