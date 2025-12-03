#!/usr/bin/env python3
# scripts/run_pipeline.py
"""
Main entry point for running the complete QC model pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env at project root
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def run_extraction(args):
    """Run data extraction from GitHub."""
    from src.data_extraction import GitHubExtractor, should_include_pr
    import json
    from config.settings import DATA_DIR

    extractor = GitHubExtractor()

    if args.repos:
        repos = args.repos.split(",")
    else:
        from config.settings import TARGET_REPOS
        repos = TARGET_REPOS

    # Check for --force flag
    resume = not getattr(args, 'force', False)

    print(f"Extracting from repos: {repos}")
    if not resume:
        print("(Force mode: ignoring cached data)")

    records = extractor.extract_all_repos(repos=repos, resume=resume)

    # Apply filters
    filtered = [r for r in records if should_include_pr(r)]
    print(f"\nAfter filtering: {len(filtered)} PRs")

    # Save filtered data
    output_path = Path(DATA_DIR) / "filtered_prs.json"
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"Saved to {output_path}")


def run_features(args):
    """Run feature engineering."""
    from src.feature_engineering import prepare_training_data

    X, y = prepare_training_data()
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")


def run_training(args):
    """Run model training."""
    from src.model_training import train_pipeline

    result = train_pipeline(
        use_grid_search=args.grid_search,
        quick_mode=args.quick,
    )

    print(f"\nModel saved to: {result['model_path']}")
    print(f"Final AUC: {result['metrics']['calibrated']['auc']:.4f}")
    print(f"Final Brier: {result['metrics']['calibrated']['brier']:.4f}")


def run_api(args):
    """Run the inference API server."""
    from src.api import run_server
    run_server()


def run_full_pipeline(args):
    """Run the complete pipeline end-to-end."""
    print("=" * 60)
    print("Running Full QC Model Pipeline")
    print("=" * 60)

    if not args.skip_extraction:
        print("\n[Step 1/4] Data Extraction")
        print("-" * 40)
        run_extraction(args)
    else:
        print("\n[Step 1/4] Skipping extraction (using cached data)")

    print("\n[Step 2/4] Feature Engineering")
    print("-" * 40)
    run_features(args)

    print("\n[Step 3/4] Model Training")
    print("-" * 40)
    run_training(args)

    if args.start_api:
        print("\n[Step 4/4] Starting API Server")
        print("-" * 40)
        run_api(args)
    else:
        print("\n[Step 4/4] Skipping API (use --start-api to run)")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="QC Model Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_pipeline.py full
  
  # Run only data extraction
  python scripts/run_pipeline.py extract --repos "microsoft/vscode,pytorch/pytorch"
  
  # Run only training with grid search
  python scripts/run_pipeline.py train --grid-search
  
  # Run quick training (no grid search)
  python scripts/run_pipeline.py train --quick
  
  # Start API server
  python scripts/run_pipeline.py api
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract data from GitHub")
    extract_parser.add_argument(
        "--repos",
        type=str,
        help="Comma-separated list of repos (default: use config)"
    )
    extract_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction, ignore cached data"
    )

    # Features command
    features_parser = subparsers.add_parser("features", help="Run feature engineering")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run hyperparameter grid search"
    )
    train_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with minimal tuning"
    )

    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run complete pipeline")
    full_parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip data extraction (use cached data)"
    )
    full_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction, ignore cached data"
    )
    full_parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run hyperparameter grid search"
    )
    full_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with minimal tuning"
    )
    full_parser.add_argument(
        "--start-api",
        action="store_true",
        help="Start API server after training"
    )
    full_parser.add_argument(
        "--repos",
        type=str,
        help="Comma-separated list of repos"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate function
    if args.command == "extract":
        run_extraction(args)
    elif args.command == "features":
        run_features(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "api":
        run_api(args)
    elif args.command == "full":
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
