"""
Step 1a: run MLRS's transliteration pipeline (Micallef et al. 2024, EACL) on the MUDT treebank.

Produces 3 versions of each split under data/processed/:
  - latin/           untouched Latin script (straight copy)
  - arabic_full/      fully transliterated to Arabic script (deterministic character mapping)
  - arabic_partial/   only tokens the etymology classifier labels "Arabic" are transliterated

NOTE on deviations from the paper's exact pipeline: see step_1a_results.txt for the full
explanation. In short, the word/character ranking LMs used for *non-deterministic*
transliteration (X_ara in the paper) live in a Google Drive folder external to the
malti repo, and that folder's anonymous-download quota was exhausted while trying to
fetch them. This script therefore uses the *deterministic* transliteration mode (fully
supported by the MLRS code, no LM required) everywhere, including as a substitute inside
the pretrained etymology classifier's own feature extraction (which normally also calls
the non-deterministic transliterator). This was verified against the paper repo's demo
example sentence: the pretrained classifier still reproduces the gold etymology labels
9/9 with this substitution (see step_1a_results.txt).
"""
import json
import os
import pickle
import sys
import shutil
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MALTI_SRC = REPO_ROOT / "experiments" / "malti" / "src"
sys.path.insert(0, str(MALTI_SRC))
os.chdir(MALTI_SRC)  # malti's own modules use paths relative to src/ (e.g. character mapping files)

import dataset_processors  # noqa: E402
import etymology_classification as eclf  # noqa: E402
from transliterate import transliterate as det_transliterate  # noqa: E402

TOKEN_MAPPINGS = ["token_mappings/small_closed_class.map", "token_mappings/additional_closed_class.map"]


def _transliterate_deterministic(token: str) -> str:
    return det_transliterate(token, eclf.CLOSED_CLASS_MAPPINGS_PATHS, None)


# Patch out the classifier's internal transliteration feature to use the deterministic
# pipeline, since the non-deterministic ranker LMs could not be downloaded (see module docstring).
eclf._transliterate = _transliterate_deterministic


def det_transliterate_token(token: str) -> str:
    return det_transliterate(token, TOKEN_MAPPINGS, None)


def main():
    data_raw = REPO_ROOT / "data" / "raw"
    out_root = REPO_ROOT / "data" / "processed"
    latin_dir = out_root / "latin"
    full_dir = out_root / "arabic_full"
    partial_dir = out_root / "arabic_partial"
    for d in (latin_dir, full_dir, partial_dir):
        d.mkdir(parents=True, exist_ok=True)

    # (1) untouched Latin script
    for f in data_raw.glob("*.conllu"):
        shutil.copy(f, latin_dir / f.name)

    processor = dataset_processors.UniversalDependenciesDatasetProcessor()

    # stats
    stats = {
        "total_tokens": 0,
        "full_changed": 0,
        "partial_changed": 0,
        "etymology_label_counts": Counter(),
        "samples": [],  # (split, before_tokens, full_after, partial_after, labels)
    }

    with open(MALTI_SRC / "etymology_data" / "model.pickle", "rb") as fh:
        model = pickle.load(fh)

    def make_full_processor():
        def _process(tokens):
            out = [det_transliterate_token(t) for t in tokens]
            stats["total_tokens"] += len(tokens)
            stats["full_changed"] += sum(1 for a, b in zip(tokens, out) if a != b)
            return out
        return _process

    def make_partial_processor():
        def _process(tokens):
            labels = model.predict([eclf.featurise(tokens)])[0]
            out = []
            for token, label in zip(tokens, labels):
                stats["etymology_label_counts"][label] += 1
                if label == "Arabic":
                    out.append(det_transliterate_token(token))
                else:
                    out.append(token)
            stats["partial_changed"] += sum(1 for a, b in zip(tokens, out) if a != b)
            return out
        return _process

    # collect a few sample sentences (from dev, first 5) before running the real pass,
    # since dataset_processor.process() doesn't expose per-sentence tokens directly.
    sample_split = "mt_mudt-ud-dev.conllu"
    sample_tokens = []
    with open(data_raw / sample_split, encoding="utf-8") as fh:
        cur = []
        for line in fh:
            line = line.strip()
            if line.startswith("#") or not line:
                if not line and cur:
                    sample_tokens.append(cur)
                    cur = []
                continue
            cur.append(line.split("\t")[1])
    sample_tokens = sample_tokens[:5]

    print("Running full transliteration (deterministic)...")
    processor.process(make_full_processor(), data_raw, full_dir)

    print("Running partial transliteration (pretrained etymology classifier + deterministic transliteration)...")
    processor.process(make_partial_processor(), data_raw, partial_dir)

    # build samples from the same 5 dev sentences, post-hoc, for the report
    for tokens in sample_tokens:
        full = [det_transliterate_token(t) for t in tokens]
        labels = list(model.predict([eclf.featurise(tokens)])[0])
        partial = [det_transliterate_token(t) if lbl == "Arabic" else t for t, lbl in zip(tokens, labels)]
        stats["samples"].append({
            "latin": tokens,
            "arabic_full": full,
            "arabic_partial": partial,
            "etymology_labels": labels,
        })

    stats["etymology_label_counts"] = dict(stats["etymology_label_counts"])
    with open(REPO_ROOT / "scripts" / "step1a_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps({k: v for k, v in stats.items() if k != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
