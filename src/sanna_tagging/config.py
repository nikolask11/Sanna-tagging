from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

UPOS_TAGS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cs_kaggle_v2.yaml"
LOCK_FILE = REPO_ROOT / "requirements-kaggle.lock"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    config = deepcopy(config)
    config["_config_path"] = str(path)
    _validate(config)
    return config


def _validate(config: dict[str, Any]) -> None:
    if tuple(config.get("upos_tags", ())) != UPOS_TAGS:
        raise ValueError("configuration must use the canonical 17-tag UPOS order")
    sizes = config.get("seed_sizes", [])
    seeds = config.get("random_seeds", [])
    if not sizes or sorted(set(sizes)) != sizes or any(size <= 0 for size in sizes):
        raise ValueError("seed_sizes must be unique, positive, and increasing")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("random_seeds must be non-empty and unique")
    parts = config["data"]["dev_partitions"]
    if abs(sum(float(value) for value in parts.values()) - 1.0) > 1e-9:
        raise ValueError("dev partition fractions must sum to one")
    training = config["training"]
    if training["stride"] >= training["max_length"] - 2:
        raise ValueError("stride must leave room for non-overlapping content")


def current_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def compute_run_fingerprint(
    config: dict[str, Any], *, code_commit: str | None = None, lock_file: Path = LOCK_FILE
) -> str:
    identity = {
        "config": canonical_config(config),
        "code_commit": code_commit or current_commit(),
        "dependency_lock_sha256": _file_hash(lock_file),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def expand_experiment_plan(config: dict[str, Any]) -> dict[str, Any]:
    sizes = config["seed_sizes"]
    seeds = config["random_seeds"]
    jobs = []
    for size in sizes:
        for seed in seeds:
            jobs.append({"stage": "budget", "variant": "slovakbert", "size": size, "seed": seed})
    jobs.append({"stage": "lapt", "variant": "slovakbert-czech-lapt"})
    jobs.extend([
        {"stage": "transfer", "variant": "slovakbert-slovak-upos"},
        {"stage": "transfer", "variant": "slovakbert-czech-lapt-slovak-upos"},
    ])
    for seed in seeds:
        jobs.append({"stage": "reference", "variant": "xlmr", "size": "selected", "seed": seed})
    for variant in (
        "slovakbert-czech-lapt",
        "slovakbert-slovak-upos",
        "slovakbert-czech-lapt-slovak-upos",
    ):
        for seed in seeds:
            jobs.append({"stage": "adapt", "variant": variant, "size": "selected", "seed": seed})
    return {
        "run_name": config["run_name"],
        "jobs": jobs,
        "supervised_invocations": sum(job["stage"] != "lapt" for job in jobs),
        "lapt_invocations": sum(job["stage"] == "lapt" for job in jobs),
    }
