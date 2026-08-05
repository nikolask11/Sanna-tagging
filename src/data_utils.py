import random
from pathlib import Path

from runpaths import data_dir

SPLIT_FILES = {
    "train": "mt_mudt-ud-train.conllu",
    "dev": "mt_mudt-ud-dev.conllu",
    "test": "mt_mudt-ud-test.conllu",
}
SEED_SIZES = [50, 100, 200, 400, 800]
RANDOM_SEEDS = [0, 1, 2]
XLMR = "xlm-roberta-base"
CAMELBERT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


def read_conllu(path):
    sents, toks, tags = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if toks:
                sents.append({"tokens": toks, "upos": tags})
                toks, tags = [], []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        toks.append(cols[1])
        tags.append(cols[3])
    if toks:
        sents.append({"tokens": toks, "upos": tags})
    return sents


def load_split(name, base=None):
    return read_conllu((base or data_dir()) / SPLIT_FILES[name])


def label_list_from(sentences):
    return sorted({t for s in sentences for t in s["upos"]})


def shuffled_indices(n, random_seed):
    idx = list(range(n))
    random.Random(1000 + random_seed).shuffle(idx)
    return idx


def seed_pool_split(train, seed_size, random_seed):
    order = shuffled_indices(len(train), random_seed)
    gold_idx = sorted(order[:seed_size])
    pool_idx = sorted(order[seed_size:])
    gold = [train[i] for i in gold_idx]
    pool = [train[i] for i in pool_idx]
    return gold, pool, gold_idx, pool_idx
