import csv
import os
import time

from runpaths import results_dir


def append_row(path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def log_problem(msg):
    path = results_dir() / "problems.log"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        f.flush()
        os.fsync(f.fileno())


def log_runtime(notebook, key, seconds):
    append_row(results_dir() / "runtimes.csv",
               ["notebook", "key", "seconds"],
               {"notebook": notebook, "key": key, "seconds": round(seconds, 1)})
