from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPO_DATA = REPO / "data" / "processed" / "arabic_full"
LATIN_DATA = REPO / "data" / "processed" / "latin"


def drive_root():
    mydrive = Path("/content/drive/MyDrive")
    root = mydrive / "sanna_m1" if mydrive.exists() else REPO / "local_run"
    for sub in ("results", "cache"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def results_dir():
    return drive_root() / "results"


def cache_dir():
    return drive_root() / "cache"


def data_dir():
    cached = cache_dir() / "data"
    if (cached / "mt_mudt-ud-train.conllu").exists():
        return cached
    return REPO_DATA
