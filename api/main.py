import os
from typing import Optional
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model/qc-model.joblib")
API_KEY = os.getenv("API_KEY", "")

# --- Models ---

class Artifact(BaseModel):
    type: str
    data: dict

class SubmissionData(BaseModel):
    notes: str = ""
    artifacts: list[Artifact] = []

class TaskContext(BaseModel):
    title: str
    description: str = ""
    requirements: Optional[str] = None
    story_points: Optional[int] = None

class ConfidenceRequest(BaseModel):
    task_id: str
    submission_data: SubmissionData
    task_context: TaskContext

class ConfidenceResponse(BaseModel):
    pass_probability: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: dict
    summary: str
    issues: list[str]
    recommendations: list[str]

# --- App ---

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"Model load failed: {e}")
    yield

app = FastAPI(title="QC Confidence API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Auth ---

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing/invalid Authorization")
    if authorization.split(" ")[1] != API_KEY:
        raise HTTPException(401, "Invalid API key")
    return True

# --- Feature Extraction (CUSTOMIZE FOR YOUR MODEL) ---

def extract_features(req: ConfidenceRequest) -> np.ndarray:
    """
    Convert request to feature vector.
    CUSTOMIZE THIS to match your model's training features.
    """
    return np.array([[
        min(len(req.submission_data.notes) / 500, 1.0),
        min(len(req.submission_data.artifacts) / 5, 1.0),
        1.0 if any(a.type == "github_pr" for a in req.submission_data.artifacts) else 0.0,
        1.0 if any(a.type == "file" for a in req.submission_data.artifacts) else 0.0,
        min((req.task_context.story_points or 5) / 13, 1.0),
    ]])

# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/v1/review/confidence", dependencies=[Depends(verify_api_key)])
async def get_confidence(req: ConfidenceRequest) -> ConfidenceResponse:
    features = extract_features(req)

    # Get prediction
    if model is not None:
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                conf = float(np.clip(model.predict(features)[0], 0.0, 1.0))
        except:
            conf = 0.8
    else:
        # Heuristic fallback
        conf = min(0.5 + len(req.submission_data.notes)/1000 + len(req.submission_data.artifacts)*0.1, 0.95)

    # Build response
    issues = []
    if len(req.submission_data.notes) < 20:
        issues.append("Notes too brief")
    if not req.submission_data.artifacts:
        issues.append("No artifacts attached")

    return ConfidenceResponse(
        pass_probability=round(conf, 4),
        confidence_breakdown={
            "completeness": round(min(conf + 0.05, 1.0), 4),
            "quality": round(conf, 4),
            "requirements_met": round(max(conf - 0.05, 0.0), 4),
        },
        summary=f"Confidence: {conf:.0%}. {len(req.submission_data.notes)} chars, {len(req.submission_data.artifacts)} artifacts.",
        issues=issues,
        recommendations=["Link GitHub PR if code changes"] if not any(a.type == "github_pr" for a in req.submission_data.artifacts) else [],
    )

# --- Optional Endpoints (used by Orbit frontend via edge function) ---

class ComplexityRequest(BaseModel):
    title: str
    description: str = ""

@app.post("/api/v1/task/complexity", dependencies=[Depends(verify_api_key)])
async def get_complexity(req: ComplexityRequest):
    """Suggest story points based on task description. Customize with your model."""
    word_count = len(f"{req.title} {req.description}".split())
    score = min(word_count / 200, 1.0)
    points = max(1, min(13, round(score * 13)))
    return {
        "suggested_story_points": points,
        "complexity_score": round(score, 4),
        "reasoning": f"Based on {word_count} words in title/description",
    }

@app.get("/api/v1/review/quality/{task_id}", dependencies=[Depends(verify_api_key)])
async def get_quality(task_id: str):
    """Quality assessment for QC reviewers. Customize with your model."""
    return {
        "overall_quality": 0.8,
        "areas_of_concern": [],
        "strengths": [],
        "comparison_to_similar": 0.5,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

