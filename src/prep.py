import json
import shutil
import time
import urllib.request

from runpaths import STUDY, REPO_DATA, LATIN_DATA, cache_dir
from data_utils import (SPLIT_FILES, RANDOM_SEEDS, CS_RAW, read_conllu,
                        shuffled_indices, load_split)
from results_io import log_runtime


def _prep_mt(dest):
    for fname in [f for fl in SPLIT_FILES["mt"].values() for f in fl]:
        if not (dest / fname).exists():
            shutil.copy(REPO_DATA / fname, dest / fname)
    latin = read_conllu(LATIN_DATA / SPLIT_FILES["mt"]["train"][0])
    arabic = read_conllu(dest / SPLIT_FILES["mt"]["train"][0])
    total = changed = 0
    for ls, ar in zip(latin, arabic):
        for lt, at in zip(ls["tokens"], ar["tokens"]):
            total += 1
            if lt != at:
                changed += 1
    frac = changed / total
    if frac < 0.5:
        raise RuntimeError(
            f"F1: only {frac:.1%} of tokens changed by transliteration -> misconfigured")
    return f"{frac:.1%} tokens transliterated"


def _prep_cs(dest):
    for fname in [f for fl in SPLIT_FILES["cs"].values() for f in fl]:
        if not (dest / fname).exists():
            urllib.request.urlretrieve(CS_RAW + fname, dest / fname)
    n_train = len(load_split("train", base=dest))
    if n_train < 50000:
        raise RuntimeError(f"cs train unexpectedly small: {n_train} sentences")
    return f"{n_train} train sentences"


def main():
    t0 = time.time()
    dest = cache_dir() / "data"
    dest.mkdir(parents=True, exist_ok=True)
    note = _prep_mt(dest) if STUDY == "mt" else _prep_cs(dest)
    manifest_path = cache_dir() / "manifest.json"
    if not manifest_path.exists():
        n = len(load_split("train", base=dest))
        manifest = {str(rs): shuffled_indices(n, rs) for rs in RANDOM_SEEDS}
        manifest_path.write_text(json.dumps(manifest))
    log_runtime("01", f"prep_{STUDY}", time.time() - t0)
    print(f"[{STUDY}] cached splits to {dest}; {note}; manifest written")
