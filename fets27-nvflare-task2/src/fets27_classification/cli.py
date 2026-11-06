"""CLI entrypoint for the FeTS27 Task 2 challenge repo."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import COHORT_NAMES, DEFAULT_DATA_LOADER_WORKERS
from .runtime import run_challenge
from .synthetic_data import prepare_assets


def configure_logging():
    """Configure the default logging configuration for the CLI tools."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None):
    """FeTS27 Task 2 CLI entry point parser and runner.

    Parses command line arguments and delegates to the appropriate sub-command
    (prepare-assets, run-local, or run-official).

    Args:
        argv: Optional list of command line argument strings. Defaults to sys.argv.
    """
    configure_logging()

    parser = argparse.ArgumentParser(
        description="FeTS27 Task 2 NVFLARE classification challenge tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-assets", help="Generate toy data and baseline checkpoints"
    )
    prepare_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    prepare_parser.add_argument("--data-root", type=Path, required=True)

    for command_name in ("run-local", "run-official"):
        run_parser = subparsers.add_parser(
            command_name, help=f"{command_name} federated training and evaluation"
        )
        run_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        run_parser.add_argument("--data-root", type=Path, required=True)
        run_parser.add_argument("--workspace", type=Path, required=True)
        run_parser.add_argument("--output-dir", type=Path, required=True)
        run_parser.add_argument(
            "--cohort", choices=[*COHORT_NAMES, "all"], default="all"
        )
        run_parser.add_argument("--num-rounds", type=int, default=2)
        run_parser.add_argument("--threads", type=int, default=None)
        run_parser.add_argument("--gpu", type=str, default=None)
        run_parser.add_argument(
            "--data-loader-workers",
            type=int,
            default=DEFAULT_DATA_LOADER_WORKERS,
            help="Worker processes per client DataLoader (default: 2)",
        )
        run_parser.add_argument(
            "--validation-limit",
            type=int,
            default=None,
            help=(
                "Optional diagnostic cap on client-side validation cases. "
                "The official evaluator always scores the full validation split."
            ),
        )

    args = parser.parse_args(argv)

    if args.command == "prepare-assets":
        created = prepare_assets(args.repo_root, args.data_root)
        print(f"Prepared assets: {created}")
        return

    cohort_names = list(COHORT_NAMES) if args.cohort == "all" else [args.cohort]
    json_path, csv_path, cohort_scores = run_challenge(
        repo_root=args.repo_root,
        mode="official" if args.command == "run-official" else "local",
        cohort_names=cohort_names,
        data_root=args.data_root,
        workspace_root=args.workspace,
        output_dir=args.output_dir,
        num_rounds=args.num_rounds,
        threads=args.threads,
        gpu=args.gpu,
        data_loader_workers=args.data_loader_workers,
        validation_limit=args.validation_limit,
    )
    print(f"JSON summary: {json_path}")
    print(f"CSV summary: {csv_path}")
    print(f"Cohorts evaluated: {[score.cohort for score in cohort_scores]}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
