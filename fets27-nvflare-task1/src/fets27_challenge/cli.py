"""CLI entrypoint for the FeTS27 Task 1 challenge repo."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import COHORT_NAMES
from .runtime import run_challenge
from .submission import package_submission, validate_submission_state, write_manifest
from .synthetic_data import prepare_assets
from .training_dummy import prepare_training_dummy_layout


def configure_logging():
    """Configure the default logging configuration for the CLI tools."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None):
    """FeTS27 Task 1 CLI entry point parser and runner.

    Parses command line arguments and delegates to the appropriate sub-command (such as
    prepare-assets, validate-submission, prepare-training-dummy, package-submission,
    write-manifest, run-local, or run-official).

    Args:
        argv: Optional list of command line argument strings. Defaults to sys.argv.
    """
    configure_logging()

    parser = argparse.ArgumentParser(
        description="FeTS27 Task 1 NVFLARE challenge tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-assets", help="Generate toy data and baseline checkpoints"
    )
    prepare_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    prepare_parser.add_argument("--data-root", type=Path, required=True)

    validate_parser = subparsers.add_parser(
        "validate-submission", help="Validate that only participant files are editable"
    )
    validate_parser.add_argument("--repo-root", type=Path, default=Path.cwd())

    dummy_parser = subparsers.add_parser(
        "prepare-training-dummy",
        help="Create a glioma data-root layout from FeTS training dummy folders",
    )
    dummy_parser.add_argument("--source-root", type=Path, required=True)
    dummy_parser.add_argument("--data-root", type=Path, required=True)
    dummy_parser.add_argument("--site-count", type=int, default=2)
    dummy_parser.add_argument("--validation-count", type=int, default=2)
    dummy_parser.add_argument(
        "--file-mode",
        choices=["absolute", "copy"],
        default="absolute",
        help="Use absolute source paths in datalists or copy files into glioma/dataset",
    )

    package_parser = subparsers.add_parser(
        "package-submission", help="Create a submission zip with only the allowed files"
    )
    package_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    package_parser.add_argument("--output", type=Path, required=True)

    manifest_parser = subparsers.add_parser(
        "write-manifest", help="Write the locked-file manifest"
    )
    manifest_parser.add_argument("--repo-root", type=Path, default=Path.cwd())

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

    args = parser.parse_args(argv)

    if args.command == "prepare-assets":
        created = prepare_assets(args.repo_root, args.data_root)
        print(f"Prepared assets: {created}")
        return

    if args.command == "validate-submission":
        validate_submission_state(args.repo_root)
        print("Submission surface is valid.")
        return

    if args.command == "prepare-training-dummy":
        summary = prepare_training_dummy_layout(
            source_root=args.source_root,
            data_root=args.data_root,
            site_count=args.site_count,
            validation_count=args.validation_count,
            file_mode=args.file_mode,
        )
        print(f"Prepared training dummy layout: {summary}")
        return

    if args.command == "package-submission":
        output_path = package_submission(args.repo_root, args.output)
        print(f"Submission written to {output_path}")
        return

    if args.command == "write-manifest":
        manifest_path = write_manifest(args.repo_root)
        print(f"Manifest written to {manifest_path}")
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
    )
    print(f"JSON summary: {json_path}")
    print(f"CSV summary: {csv_path}")
    print(f"Cohorts evaluated: {[score.cohort for score in cohort_scores]}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
