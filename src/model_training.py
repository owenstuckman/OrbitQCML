# src/model_training.py
"""
Train and calibrate the QC prediction model.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR, MODELS_DIR, TEST_RATIO, CALIBRATION_RATIO,
    RANDOM_STATE, RF_PARAM_GRID
)
from src.feature_engineering import (
    FEATURE_COLUMNS, TARGET_COLUMN, load_processed_data
)


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    test_ratio: float = TEST_RATIO,
    calibration_ratio: float = CALIBRATION_RATIO,
) -> dict:
    """
    Split data temporally (train on older, test on newer).
    This prevents data leakage from future author statistics.
    
    Returns dict with keys: X_train, X_cal, X_test, y_train, y_cal, y_test
    """
    # Sort by creation time
    sort_idx = metadata["created_at"].argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)
    metadata = metadata.iloc[sort_idx].reset_index(drop=True)
    
    n = len(X)
    train_end = int(n * (1 - test_ratio - calibration_ratio))
    cal_end = int(n * (1 - test_ratio))
    
    splits = {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_cal": X.iloc[train_end:cal_end],
        "y_cal": y.iloc[train_end:cal_end],
        "X_test": X.iloc[cal_end:],
        "y_test": y.iloc[cal_end:],
        "metadata_train": metadata.iloc[:train_end],
        "metadata_test": metadata.iloc[cal_end:],
    }
    
    print(f"Temporal split sizes:")
    print(f"  Train: {len(splits['X_train'])} ({len(splits['X_train'])/n:.1%})")
    print(f"  Calibration: {len(splits['X_cal'])} ({len(splits['X_cal'])/n:.1%})")
    print(f"  Test: {len(splits['X_test'])} ({len(splits['X_test'])/n:.1%})")
    
    # Show time ranges
    print(f"\nTime ranges:")
    print(f"  Train: {metadata.iloc[0]['created_at']} to {metadata.iloc[train_end-1]['created_at']}")
    print(f"  Test: {metadata.iloc[cal_end]['created_at']} to {metadata.iloc[-1]['created_at']}")
    
    return splits


def handle_class_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "class_weight"
) -> tuple:
    """
    Handle class imbalance in training data.
    
    Args:
        method: "smote" for oversampling, "class_weight" for weighted loss
    
    Returns:
        X_balanced, y_balanced, sample_weights (or None)
    """
    print(f"\nClass distribution in training:")
    print(y_train.value_counts(normalize=True).round(3))
    
    if method == "smote":
        print("Applying SMOTE oversampling...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {len(X_balanced)} samples")
        return X_balanced, y_balanced, None
    
    else:
        # Use class weights (handled by classifier)
        return X_train, y_train, None


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[dict] = None,
    cv_folds: int = 5,
    use_grid_search: bool = True,
) -> RandomForestClassifier:
    """
    Train Random Forest with optional hyperparameter tuning.
    """
    param_grid = param_grid or RF_PARAM_GRID
    
    if use_grid_search:
        print("\nRunning GridSearchCV for hyperparameter tuning...")
        print(f"  Parameter grid: {param_grid}")
        print(f"  CV folds: {cv_folds}")
        
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    else:
        # Train with reasonable defaults
        print("\nTraining Random Forest with default parameters...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        return rf


def calibrate_model(
    model: RandomForestClassifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """
    Calibrate model probabilities using isotonic regression or Platt scaling.
    
    Critical for Shapley payouts: ensures p0 reflects true probabilities.
    """
    print(f"\nCalibrating model probabilities using {method} regression...")
    
    calibrated = CalibratedClassifierCV(
        model,
        method=method,  # "isotonic" or "sigmoid" (Platt)
        cv="prefit",  # Use pre-fitted model
    )
    
    calibrated.fit(X_cal, y_cal)
    
    return calibrated


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Comprehensive model evaluation.
    """
    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name}")
    print("=" * 50)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Probability metrics
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nProbability Metrics:")
    print(f"  ROC AUC: {auc:.4f}")
    print(f"  Brier Score: {brier:.4f} (lower is better)")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Calibration analysis
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    print(f"  Mean Calibration Error: {calibration_error:.4f}")
    
    # Probability distribution
    print(f"\nPredicted probability distribution:")
    print(f"  Min: {y_prob.min():.3f}")
    print(f"  Max: {y_prob.max():.3f}")
    print(f"  Mean: {y_prob.mean():.3f}")
    print(f"  Std: {y_prob.std():.3f}")
    
    return {
        "auc": auc,
        "brier": brier,
        "accuracy": accuracy,
        "calibration_error": calibration_error,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list,
) -> pd.DataFrame:
    """Get feature importance from trained model."""
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print("\nFeature Importance:")
    print(importance.to_string(index=False))
    
    return importance


def save_model(
    model,
    feature_columns: list,
    metrics: dict,
    version: Optional[str] = None,
) -> str:
    """
    Save trained model with metadata.
    """
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "version": version,
        "trained_at": datetime.now().isoformat(),
    }
    
    # Save model
    model_path = models_dir / f"qc_model_{version}.joblib"
    joblib.dump(artifact, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Save as "latest" for easy loading
    latest_path = models_dir / "qc_model_latest.joblib"
    joblib.dump(artifact, latest_path)
    print(f"Saved as latest: {latest_path}")
    
    # Save metrics separately as JSON
    metrics_path = models_dir / f"metrics_{version}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return str(model_path)


def load_model(path: Optional[str] = None) -> tuple:
    """
    Load trained model.
    
    Returns:
        model, feature_columns, metadata
    """
    if path is None:
        path = Path(MODELS_DIR) / "qc_model_latest.joblib"
    
    artifact = joblib.load(path)
    
    return (
        artifact["model"],
        artifact["feature_columns"],
        {k: v for k, v in artifact.items() if k not in ["model", "feature_columns"]},
    )


def train_pipeline(
    use_grid_search: bool = False,
    quick_mode: bool = False,
) -> dict:
    """
    Full training pipeline.
    
    Args:
        use_grid_search: Whether to run hyperparameter search
        quick_mode: Use smaller param grid for faster iteration
    """
    print("=" * 60)
    print("QC Model Training Pipeline")
    print("=" * 60)
    
    # Load processed data
    print("\n1. Loading processed data...")
    X, y, metadata = load_processed_data()
    print(f"   Loaded {len(X)} samples with {len(FEATURE_COLUMNS)} features")
    
    # Temporal split
    print("\n2. Splitting data (temporal)...")
    splits = temporal_train_test_split(X, y, metadata)
    
    # Handle class imbalance
    print("\n3. Handling class imbalance...")
    X_train, y_train, _ = handle_class_imbalance(
        splits["X_train"], 
        splits["y_train"],
        method="class_weight"
    )
    
    # Train model
    print("\n4. Training Random Forest...")
    if quick_mode:
        param_grid = {
            "n_estimators": [100],
            "max_depth": [20],
            "class_weight": ["balanced"],
        }
    else:
        param_grid = RF_PARAM_GRID
    
    model = train_random_forest(
        X_train, 
        y_train,
        param_grid=param_grid,
        use_grid_search=use_grid_search,
    )
    
    # Evaluate uncalibrated model
    print("\n5. Evaluating uncalibrated model...")
    metrics_uncal = evaluate_model(
        model, 
        splits["X_test"], 
        splits["y_test"],
        "Uncalibrated Random Forest"
    )
    
    # Calibrate
    print("\n6. Calibrating probabilities...")
    calibrated_model = calibrate_model(
        model,
        splits["X_cal"],
        splits["y_cal"],
    )
    
    # Evaluate calibrated model
    print("\n7. Evaluating calibrated model...")
    metrics_cal = evaluate_model(
        calibrated_model,
        splits["X_test"],
        splits["y_test"],
        "Calibrated Random Forest"
    )
    
    # Feature importance (from base model)
    print("\n8. Analyzing feature importance...")
    importance = get_feature_importance(model, FEATURE_COLUMNS)
    
    # Save model
    print("\n9. Saving model...")
    all_metrics = {
        "uncalibrated": metrics_uncal,
        "calibrated": metrics_cal,
        "feature_importance": importance.to_dict("records"),
    }
    model_path = save_model(
        calibrated_model,
        FEATURE_COLUMNS,
        all_metrics,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return {
        "model": calibrated_model,
        "metrics": all_metrics,
        "model_path": model_path,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train QC prediction model")
    parser.add_argument(
        "--grid-search", 
        action="store_true",
        help="Run hyperparameter grid search"
    )
    parser.add_argument(
        "--quick",
        action="store_true", 
        help="Quick mode with reduced param grid"
    )
    
    args = parser.parse_args()
    
    result = train_pipeline(
        use_grid_search=args.grid_search,
        quick_mode=args.quick,
    )
