import random
from pathlib import Path

from runpaths import STUDY, data_dir

SEED_SIZES = [50, 100, 200, 400, 800]
RANDOM_SEEDS = [0, 1, 2]
XLMR = "xlm-roberta-base"
CAMELBERT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
SLOVAKBERT = "gerulata/slovakbert"

CS_RAW = "https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-PDT/master/"
CS_TRAIN_PARTS = ["cs_pdtc-ud-train-lt.conllu", "cs_pdtc-ud-train-la.conllu",
                  "cs_pdtc-ud-train-ca.conllu", "cs_pdtc-ud-train-wt0.conllu",
                  "cs_pdtc-ud-train-wt1.conllu"]

SPLIT_FILES = {
    "mt": {"train": ["mt_mudt-ud-train.conllu"],
           "dev": ["mt_mudt-ud-dev.conllu"],
           "test": ["mt_mudt-ud-test.conllu"]},
    "cs": {"train": CS_TRAIN_PARTS,
           "dev": ["cs_pdtc-ud-dev.conllu"],
           "test": ["cs_pdtc-ud-test.conllu"]},
}
PRIMARY_MODEL = {"mt": XLMR, "cs": SLOVAKBERT}
DEV_SUBSAMPLE = {"mt": None, "cs": 3000}
SELFTRAIN_POOL_CAP = 10000

_QUOTES = {"„": '"', "“": '"', "”": '"',
           "‚": "'", "‘": "'", "’": "'"}


def _norm(token):
    return _QUOTES.get(token, token)


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
        toks.append(_norm(cols[1]))
        tags.append(cols[3])
    if toks:
        sents.append({"tokens": toks, "upos": tags})
    return sents


def load_split(name, base=None):
    base = base or data_dir()
    sents = []
    for fname in SPLIT_FILES[STUDY][name]:
        sents.extend(read_conllu(base / fname))
    cap = DEV_SUBSAMPLE[STUDY]
    if name == "dev" and cap and len(sents) > cap:
        idx = sorted(random.Random(77).sample(range(len(sents)), cap))
        sents = [sents[i] for i in idx]
    return sents


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


def pool_subsample(pool, random_seed, cap=SELFTRAIN_POOL_CAP):
    if len(pool) <= cap:
        return pool
    idx = sorted(random.Random(500 + random_seed).sample(range(len(pool)), cap))
    return [pool[i] for i in idx]
