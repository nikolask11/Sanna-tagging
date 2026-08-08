"""Command-line interface for the manifest-driven v2 experiment stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG, REPO_ROOT
from .runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sanna-tagging",
        description="Run the reproducible Czech UPOS v2 stage DAG.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="v2 YAML configuration (default: configs/cs_kaggle_v2.yaml)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs",
        help="artifact root; supports /kaggle/working/runs stage handoffs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the requested exact stage/job plan without writing artifacts",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="non-production smoke mode with a distinct fingerprint and bounded updates",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="materialize the exact DAG and 30-job plan")

    prepare = subparsers.add_parser(
        "prepare", help="download pinned corpora and prepare leakage-safe data"
    )
    prepare.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="optional pinned download cache (default: <run>/raw)",
    )

    subparsers.add_parser(
        "budget", help="run 5 sizes x 3 seeds and aggregate/select the budget"
    )
    subparsers.add_parser(
        "adapt",
        help="run LAPT, transfers, references, adapted arms, selection, and calibration",
    )
    subparsers.add_parser(
        "benchmark",
        help="optional non-selective throughput hook for the frozen winning ensemble",
    )
    subparsers.add_parser(
        "report-draft",
        help="write the no-test draft and freeze selection.lock.json",
    )

    finalize = subparsers.add_parser(
        "finalize-test",
        help="explicit one-shot official test evaluation from a matching frozen lock",
    )
    finalize.add_argument(
        "--selection-lock",
        type=Path,
        required=True,
        help="the exact selection.lock.json produced by report-draft",
    )
    finalize.add_argument(
        "--official-test",
        type=Path,
        required=True,
        help="pinned official gold CoNLL-U; opened only after the one-shot marker",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    runner = ExperimentRunner(
        config_path=arguments.config,
        runs_root=arguments.runs_root,
        dry_run=arguments.dry_run,
        tiny=arguments.tiny,
    )
    try:
        if arguments.command == "plan":
            runner.write_plan()
        elif arguments.command == "prepare":
            runner.prepare(raw_root=arguments.raw_root)
        elif arguments.command == "budget":
            runner.budget()
        elif arguments.command == "adapt":
            runner.adapt()
        elif arguments.command == "benchmark":
            runner.benchmark()
        elif arguments.command == "report-draft":
            runner.report_draft()
        elif arguments.command == "finalize-test":
            if arguments.tiny:
                parser.error("finalize-test is forbidden in --tiny non-production mode")
            runner.finalize_test(
                selection_lock=arguments.selection_lock,
                official_test_path=arguments.official_test,
            )
        else:  # pragma: no cover - argparse enforces the command choices.
            parser.error(f"unknown command: {arguments.command}")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"sanna-tagging: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
