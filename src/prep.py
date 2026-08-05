import json
import shutil
import time

from runpaths import REPO_DATA, LATIN_DATA, cache_dir
from data_utils import SPLIT_FILES, RANDOM_SEEDS, read_conllu, shuffled_indices
from results_io import log_runtime


def main():
    t0 = time.time()
    dest = cache_dir() / "data"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in SPLIT_FILES.values():
        if not (dest / fname).exists():
            shutil.copy(REPO_DATA / fname, dest / fname)

    latin = read_conllu(LATIN_DATA / SPLIT_FILES["train"])
    arabic = read_conllu(dest / SPLIT_FILES["train"])
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

    manifest_path = cache_dir() / "manifest.json"
    if not manifest_path.exists():
        n = len(arabic)
        manifest = {str(rs): shuffled_indices(n, rs) for rs in RANDOM_SEEDS}
        manifest_path.write_text(json.dumps(manifest))
    log_runtime("01", "prep", time.time() - t0)
    print(f"cached 3 splits to {dest}; {frac:.1%} tokens transliterated; manifest written")
