# src/feature_engineering.py
"""
Feature engineering for QC model.
Transforms raw PR data into model-ready features.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR, COLD_START_THRESHOLD, DEFAULT_PASS_RATE_PRIOR
)


# Feature columns for the model
FEATURE_COLUMNS = [
    # Author features
    "author_pass_rate",
    "author_prior_prs",
    "author_avg_cycles",
    "author_experience_level",
    
    # Submission size features
    "log_lines_changed",
    "log_files_changed",
    "lines_per_file",
    
    # Documentation features
    "title_length",
    "body_length",
    "has_description",
    
    # Commit features
    "commits",
    "lines_per_commit",
    
    # Label features
    "is_bug_fix",
    "is_documentation",
    "is_feature",
    "is_refactor",
    
    # Time features
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

TARGET_COLUMN = "passed_first_review"


def compute_author_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling author statistics.
    Uses only data available at prediction time (no future leakage).
    """
    # Sort by creation time
    df = df.sort_values("created_at").copy()
    
    # Initialize author stats tracking
    author_stats = {}
    
    # Lists to store computed features
    author_pass_rates = []
    author_prior_prs = []
    author_avg_cycles = []
    
    for idx, row in df.iterrows():
        author = row["author"]
        
        if author not in author_stats:
            author_stats[author] = {
                "total_prs": 0,
                "passed_first": 0,
                "total_cycles": 0,
            }
        
        stats = author_stats[author]
        
        # Compute features BEFORE updating (rolling window - no leakage)
        if stats["total_prs"] > 0:
            pass_rate = stats["passed_first"] / stats["total_prs"]
            avg_cycles = stats["total_cycles"] / stats["total_prs"]
        else:
            # Cold start: use prior
            pass_rate = DEFAULT_PASS_RATE_PRIOR
            avg_cycles = 1.0
        
        author_pass_rates.append(pass_rate)
        author_prior_prs.append(stats["total_prs"])
        author_avg_cycles.append(avg_cycles)
        
        # Update stats for next PR
        stats["total_prs"] += 1
        if row["passed_first_review"]:
            stats["passed_first"] += 1
        stats["total_cycles"] += row.get("review_cycles", 0)
    
    df["author_pass_rate"] = author_pass_rates
    df["author_prior_prs"] = author_prior_prs
    df["author_avg_cycles"] = author_avg_cycles
    
    return df


def extract_label_features(labels: list) -> dict:
    """Extract boolean features from PR labels."""
    labels_lower = [l.lower() for l in labels]
    labels_str = " ".join(labels_lower)
    
    return {
        "is_bug_fix": any(
            kw in labels_str for kw in ["bug", "fix", "hotfix", "patch"]
        ),
        "is_documentation": any(
            kw in labels_str for kw in ["doc", "documentation", "docs"]
        ),
        "is_feature": any(
            kw in labels_str for kw in ["feature", "enhancement", "new"]
        ),
        "is_refactor": any(
            kw in labels_str for kw in ["refactor", "cleanup", "tech debt"]
        ),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw PR data into model features.
    """
    df = df.copy()
    
    # Parse dates if needed
    if df["created_at"].dtype == object:
        df["created_at"] = pd.to_datetime(df["created_at"])
    
    # Compute author history features
    print("Computing author history features...")
    df = compute_author_history(df)
    
    # Size features (log transform for skewed distributions)
    df["lines_changed"] = df["lines_added"] + df["lines_deleted"]
    df["log_lines_changed"] = np.log1p(df["lines_changed"])
    df["log_files_changed"] = np.log1p(df["files_changed"])
    df["lines_per_file"] = df["lines_changed"] / df["files_changed"].clip(lower=1)
    
    # Documentation features
    df["has_description"] = (df["body_length"] > 50).astype(int)
    
    # Commit features
    df["lines_per_commit"] = df["lines_changed"] / df["commits"].clip(lower=1)
    
    # Label features
    print("Extracting label features...")
    label_features = df["labels"].apply(extract_label_features)
    label_df = pd.DataFrame(label_features.tolist())
    for col in label_df.columns:
        df[col] = label_df[col].astype(int)
    
    # Time features
    df["hour_of_day"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Author experience level (categorical -> numeric)
    df["author_experience_level"] = pd.cut(
        df["author_prior_prs"],
        bins=[-1, 5, 20, 100, float("inf")],
        labels=[0, 1, 2, 3]  # new, junior, regular, veteran
    ).astype(int)
    
    return df


def prepare_training_data(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw data, engineer features, and prepare for training.
    
    Returns:
        X: Feature matrix
        y: Target labels
    """
    data_dir = Path(DATA_DIR)
    
    # Load data
    if input_path is None:
        input_path = data_dir / "filtered_prs.json"
    
    print(f"Loading data from {input_path}...")
    with open(input_path) as f:
        records = json.load(f)
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records")
    
    # Build features
    print("Building features...")
    df = build_features(df)
    
    # Select feature columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        # Add missing columns with default values
        for col in missing_cols:
            df[col] = 0
    
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Handle any missing values
    X = X.fillna(0)
    
    # Save processed data
    if output_path is None:
        output_path = data_dir / "processed_features.parquet"
    
    # Combine for saving
    processed = X.copy()
    processed[TARGET_COLUMN] = y
    processed["created_at"] = df["created_at"]
    processed["author"] = df["author"]
    processed["repo"] = df["repo"]
    processed["pr_number"] = df["pr_number"]
    
    processed.to_parquet(output_path)
    print(f"Saved processed data to {output_path}")
    
    # Print feature statistics
    print("\nFeature statistics:")
    print(X.describe().round(2))
    
    print(f"\nTarget distribution:")
    print(y.value_counts(normalize=True).round(3))
    
    return X, y


def load_processed_data(
    path: Optional[str] = None
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load previously processed feature data.
    
    Returns:
        X: Feature matrix
        y: Target labels
        metadata: Additional columns (created_at, author, etc.)
    """
    if path is None:
        path = Path(DATA_DIR) / "processed_features.parquet"
    
    df = pd.read_parquet(path)
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    metadata = df[["created_at", "author", "repo", "pr_number"]]
    
    return X, y, metadata


if __name__ == "__main__":
    X, y = prepare_training_data()
    print(f"\nFinal shapes: X={X.shape}, y={y.shape}")
