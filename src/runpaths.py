import os
from pathlib import Path

STUDY = os.environ.get("STUDY", "mt")

REPO = Path(__file__).resolve().parent.parent
REPO_DATA = REPO / "data" / "processed" / "arabic_full"
LATIN_DATA = REPO / "data" / "processed" / "latin"


def _suffix():
    return "" if STUDY == "mt" else f"_{STUDY}"


def drive_root():
    mydrive = Path("/content/drive/MyDrive")
    kaggle = Path("/kaggle/working")
    if mydrive.exists():
        root = mydrive / "sanna_m1"
    elif kaggle.exists():
        root = kaggle / "sanna_m1"
    else:
        root = REPO / "local_run"
    for sub in (f"results{_suffix()}", f"cache{_suffix()}"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def results_dir():
    return drive_root() / f"results{_suffix()}"


def cache_dir():
    return drive_root() / f"cache{_suffix()}"


def data_dir():
    cached = cache_dir() / "data"
    if STUDY == "mt" and not (cached / "mt_mudt-ud-train.conllu").exists():
        return REPO_DATA
    return cached
