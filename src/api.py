# src/api.py
"""
FastAPI inference server for QC predictions.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import API_HOST, API_PORT, MODELS_DIR
from src.model_training import load_model
from src.feature_engineering import FEATURE_COLUMNS


# Initialize FastAPI app
app = FastAPI(
    title="QC Model API",
    description="Predict probability of first-pass QC approval",
    version="1.0.0",
)

# Global model reference (loaded on startup)
MODEL = None
FEATURE_COLS = None


class PredictionRequest(BaseModel):
    """Request schema for QC prediction."""
    
    # Required features
    worker_id: str = Field(..., description="Unique worker identifier")
    task_id: str = Field(..., description="Unique task identifier")
    
    # Submission features
    lines_changed: int = Field(..., ge=0, description="Total lines added + deleted")
    files_changed: int = Field(..., ge=1, description="Number of files modified")
    title_length: int = Field(..., ge=0, description="Title character count")
    body_length: int = Field(0, ge=0, description="Description character count")
    commits: int = Field(1, ge=1, description="Number of commits")
    
    # Task type (from labels or category)
    is_bug_fix: bool = Field(False, description="Is this a bug fix?")
    is_documentation: bool = Field(False, description="Is this documentation?")
    is_feature: bool = Field(False, description="Is this a new feature?")
    is_refactor: bool = Field(False, description="Is this a refactor?")
    
    # Optional: worker history (if available)
    worker_pass_rate: Optional[float] = Field(
        None, ge=0, le=1, description="Worker's historical pass rate"
    )
    worker_prior_tasks: Optional[int] = Field(
        None, ge=0, description="Worker's prior completed tasks"
    )
    worker_avg_cycles: Optional[float] = Field(
        None, ge=0, description="Worker's average review cycles"
    )
    
    # Optional: time context
    submission_hour: Optional[int] = Field(
        None, ge=0, le=23, description="Hour of submission (0-23)"
    )
    submission_day: Optional[int] = Field(
        None, ge=0, le=6, description="Day of week (0=Monday)"
    )


class PredictionResponse(BaseModel):
    """Response schema for QC prediction."""
    
    task_id: str
    worker_id: str
    p0: float = Field(..., description="Probability of passing first QC review")
    confidence_bucket: str = Field(
        ..., description="Confidence level: high, medium, or low"
    )
    recommended_action: str = Field(
        ..., description="Suggested workflow action"
    )
    feature_contributions: Optional[dict] = Field(
        None, description="Feature contributions to prediction"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


def get_confidence_bucket(p0: float) -> tuple[str, str]:
    """
    Determine confidence bucket and recommended action.
    
    Returns:
        (bucket_name, recommended_action)
    """
    if p0 >= 0.9:
        return "high", "auto_approve_eligible"
    elif p0 >= 0.7:
        return "medium", "standard_review"
    elif p0 >= 0.5:
        return "low", "careful_review"
    else:
        return "very_low", "priority_review"


def build_feature_vector(request: PredictionRequest) -> pd.DataFrame:
    """
    Build feature vector from request.
    Handles missing optional features with defaults.
    """
    now = datetime.now()
    
    features = {
        # Author features
        "author_pass_rate": request.worker_pass_rate or 0.5,
        "author_prior_prs": request.worker_prior_tasks or 0,
        "author_avg_cycles": request.worker_avg_cycles or 1.0,
        "author_experience_level": (
            0 if (request.worker_prior_tasks or 0) <= 5
            else 1 if (request.worker_prior_tasks or 0) <= 20
            else 2 if (request.worker_prior_tasks or 0) <= 100
            else 3
        ),
        
        # Submission features
        "log_lines_changed": np.log1p(request.lines_changed),
        "log_files_changed": np.log1p(request.files_changed),
        "lines_per_file": request.lines_changed / max(request.files_changed, 1),
        
        # Documentation features
        "title_length": request.title_length,
        "body_length": request.body_length,
        "has_description": 1 if request.body_length > 50 else 0,
        
        # Commit features
        "commits": request.commits,
        "lines_per_commit": request.lines_changed / max(request.commits, 1),
        
        # Label features
        "is_bug_fix": int(request.is_bug_fix),
        "is_documentation": int(request.is_documentation),
        "is_feature": int(request.is_feature),
        "is_refactor": int(request.is_refactor),
        
        # Time features
        "hour_of_day": request.submission_hour or now.hour,
        "day_of_week": request.submission_day or now.weekday(),
        "is_weekend": 1 if (request.submission_day or now.weekday()) >= 5 else 0,
    }
    
    # Ensure correct column order
    df = pd.DataFrame([features])[FEATURE_COLS]
    
    return df


@app.on_event("startup")
async def load_model_on_startup():
    """Load model when server starts."""
    global MODEL, FEATURE_COLS
    
    try:
        MODEL, FEATURE_COLS, metadata = load_model()
        print(f"Model loaded successfully: version {metadata.get('version', 'unknown')}")
    except FileNotFoundError:
        print("Warning: No trained model found. Train a model first.")
        MODEL = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        model_version=None,  # Could extract from metadata
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict probability of first-pass QC approval.
    
    Returns p0 for use in Shapley payout calculations.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    # Build features
    features = build_feature_vector(request)
    
    # Get prediction
    p0 = float(MODEL.predict_proba(features)[0, 1])
    
    # Get confidence bucket and action
    bucket, action = get_confidence_bucket(p0)
    
    return PredictionResponse(
        task_id=request.task_id,
        worker_id=request.worker_id,
        p0=round(p0, 4),
        confidence_bucket=bucket,
        recommended_action=action,
    )


@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """Batch prediction endpoint."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    results = []
    for req in requests:
        features = build_feature_vector(req)
        p0 = float(MODEL.predict_proba(features)[0, 1])
        bucket, action = get_confidence_bucket(p0)
        
        results.append({
            "task_id": req.task_id,
            "worker_id": req.worker_id,
            "p0": round(p0, 4),
            "confidence_bucket": bucket,
            "recommended_action": action,
        })
    
    return {"predictions": results}


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    
    return {
        "feature_columns": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "model_type": type(MODEL).__name__,
    }


def run_server():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    run_server()
