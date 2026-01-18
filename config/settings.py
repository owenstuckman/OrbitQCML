# config/settings.py
"""
Configuration settings for QC Model training pipeline.
Copy this to config/settings_local.py and update with your values.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# GitHub API
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# Target repositories for training data
TARGET_REPOS = [
    "microsoft/vscode",
    "kubernetes/kubernetes",
    "pytorch/pytorch",
    "rust-lang/rust",
    "facebook/react",
    "tensorflow/tensorflow",
    "freeCodeCamp/freeCodeCamp",
]

# Data collection settings
MIN_PR_LINES = 10  # Minimum lines changed to include PR
MAX_PRS_PER_REPO = 5000  # Limit PRs per repo (None for all)
LOOKBACK_DAYS = 365  # How far back to collect data

# Feature engineering
COLD_START_THRESHOLD = 5  # PRs before using full model
DEFAULT_PASS_RATE_PRIOR = 0.5  # Prior for new authors

# Model training
TEST_RATIO = 0.15
CALIBRATION_RATIO = 0.15
RANDOM_STATE = 42

# Hyperparameter grid for Random Forest
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", "balanced_subsample"],
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Database (optional - for production)
DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    f"sqlite:///{DATA_DIR}/qc_model.db"
)

# Monitoring thresholds
BRIER_SCORE_RETRAIN_THRESHOLD = 0.10  # Retrain if Brier degrades by 10%
CALIBRATION_CHECK_INTERVAL_DAYS = 7
