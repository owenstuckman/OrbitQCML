# QC Model Training Pipeline

A machine learning pipeline for predicting first-pass QC approval probability (p₀) using GitHub PR data. This model powers the Shapley value payout system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Configuration](#configuration)
6. [API Usage](#api-usage)
7. [Project Structure](#project-structure)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Clone/copy project and install dependencies
cd qc_model
pip install -r requirements.txt

# 2. Option A: Use demo data (no GitHub token needed)
python scripts/generate_demo_data.py --n-prs 2000

# 2. Option B: Extract real GitHub data
export GITHUB_TOKEN="your_token_here"
python scripts/run_pipeline.py extract

# 3. Run feature engineering
python scripts/run_pipeline.py features

# 4. Train the model
python scripts/run_pipeline.py train --quick

# 5. Start the API server
python scripts/run_pipeline.py api
```

---


## Step-by-Step Guide

#### Option B: Real GitHub Data

Extract PRs from open-source repositories:

```bash
# Use default repositories (vscode, kubernetes, pytorch, etc.)
python scripts/run_pipeline.py extract

# Or specify custom repositories
python scripts/run_pipeline.py extract --repos "facebook/react,vercel/next.js"
```

**Expected output:**
```
Extracting PRs from microsoft/vscode...
  Since: 2024-01-01
  Max PRs: 5000
  Processing: 3,241 PRs [00:45:32]
  Extracted 2,847 PRs
Saved to data/raw_microsoft_vscode.json

After filtering: 2,534 PRs
Saved filtered data to data/filtered_prs.json
```

**Time estimates:**
- Small repo (< 1000 PRs): 5-10 minutes
- Medium repo (1000-5000 PRs): 30-60 minutes  
- Large repo (5000+ PRs): 1-2 hours

### Phase 2: Feature Engineering

Transform raw PR data into model features:

```bash
python scripts/run_pipeline.py features
```

**What this does:**
1. Computes rolling author statistics (pass rate, avg cycles)
2. Extracts label features (bug, feature, docs, refactor)
3. Creates time features (hour, day, weekend)
4. Log-transforms size features
5. Saves to `data/processed_features.parquet`

**Expected output:**
```
Loading data from data/filtered_prs.json...
Loaded 2534 records
Computing author history features...
Extracting label features...
Saved processed data to data/processed_features.parquet

Feature statistics:
                   author_pass_rate  author_prior_prs  ...
count                   2534.00           2534.00     ...
mean                       0.72             23.45     ...

Target distribution:
1    0.734
0    0.266
```

### Phase 3: Model Training

Train and calibrate the Random Forest model:

```bash
# Quick training (no hyperparameter search)
python scripts/run_pipeline.py train --quick

# Full training with grid search (slower, better results)
python scripts/run_pipeline.py train --grid-search
```

**What this does:**
1. Temporal train/test split (no data leakage)
2. Handles class imbalance with balanced class weights
3. Trains Random Forest classifier
4. Calibrates probabilities with isotonic regression
5. Evaluates on held-out test set
6. Saves model to `models/qc_model_latest.joblib`

**Expected output:**
```
==================================================
QC Model Training Pipeline
==================================================

1. Loading processed data...
   Loaded 2534 samples with 20 features

2. Splitting data (temporal)...
   Train: 1773 (70.0%)
   Calibration: 380 (15.0%)
   Test: 381 (15.0%)

3. Handling class imbalance...
   Class distribution: {0: 0.27, 1: 0.73}

4. Training Random Forest...
   Best CV AUC: 0.8234

5. Evaluating calibrated model...
   ROC AUC: 0.8156
   Brier Score: 0.1423
   Mean Calibration Error: 0.0312

Training complete!
```

### Phase 4: Run API Server

Start the inference API:

```bash
python scripts/run_pipeline.py api
```

Server runs at http://localhost:8000

**Test the API:**

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "worker_id": "user123",
    "task_id": "task456",
    "lines_changed": 150,
    "files_changed": 3,
    "title_length": 45,
    "body_length": 200,
    "commits": 2,
    "is_bug_fix": true,
    "worker_pass_rate": 0.85,
    "worker_prior_tasks": 25
  }'
```

**Response:**
```json
{
  "task_id": "task456",
  "worker_id": "user123",
  "p0": 0.8234,
  "confidence_bucket": "medium",
  "recommended_action": "standard_review"
}
```

---

## Configuration

Edit `config/settings.py` to customize:

```python
# Target repositories for data collection
TARGET_REPOS = [
    "microsoft/vscode",
    "kubernetes/kubernetes",
    # Add your repos here
]

# Data filtering
MIN_PR_LINES = 10  # Minimum lines to include PR
MAX_PRS_PER_REPO = 5000  # Limit per repo
LOOKBACK_DAYS = 365  # How far back to collect

# Model training
TEST_RATIO = 0.15
CALIBRATION_RATIO = 0.15

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
```

---

## API Usage

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |

### Request Schema

```json
{
  "worker_id": "string (required)",
  "task_id": "string (required)",
  "lines_changed": "int >= 0 (required)",
  "files_changed": "int >= 1 (required)",
  "title_length": "int >= 0 (required)",
  "body_length": "int >= 0 (default: 0)",
  "commits": "int >= 1 (default: 1)",
  "is_bug_fix": "bool (default: false)",
  "is_documentation": "bool (default: false)",
  "is_feature": "bool (default: false)",
  "is_refactor": "bool (default: false)",
  "worker_pass_rate": "float 0-1 (optional)",
  "worker_prior_tasks": "int >= 0 (optional)",
  "worker_avg_cycles": "float >= 0 (optional)",
  "submission_hour": "int 0-23 (optional)",
  "submission_day": "int 0-6 (optional)"
}
```

### Response Schema

```json
{
  "task_id": "string",
  "worker_id": "string",
  "p0": "float (0-1, probability of first-pass approval)",
  "confidence_bucket": "high|medium|low|very_low",
  "recommended_action": "auto_approve_eligible|standard_review|careful_review|priority_review"
}
```

### Confidence Buckets

| Bucket | p0 Range | Action |
|--------|----------|--------|
| high | >= 0.9 | Auto-approve eligible |
| medium | 0.7 - 0.9 | Standard review |
| low | 0.5 - 0.7 | Careful review |
| very_low | < 0.5 | Priority review |

---

## Project Structure

```
qc_model/
├── config/
│   └── settings.py          # Configuration settings
├── data/                     # Generated data files
│   ├── raw_*.json           # Raw PR data per repo
│   ├── filtered_prs.json    # Filtered combined data
│   └── processed_features.parquet  # Feature matrix
├── models/                   # Trained models
│   ├── qc_model_latest.joblib
│   └── metrics_*.json
├── scripts/
│   ├── run_pipeline.py      # Main CLI entry point
│   └── generate_demo_data.py # Demo data generator
├── src/
│   ├── __init__.py
│   ├── data_extraction.py   # GitHub API extraction
│   ├── feature_engineering.py # Feature transformation
│   ├── model_training.py    # Training pipeline
│   └── api.py               # FastAPI server
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### Common Issues

#### GitHub Rate Limiting

```
Error: RateLimitExceededException
```

**Solution:** The extractor handles this automatically, but you can also:
- Use authenticated requests (set GITHUB_TOKEN)
- Reduce MAX_PRS_PER_REPO in config
- Wait for rate limit reset

#### No Model Found

```
Error: FileNotFoundError - No trained model found
```

**Solution:** Train a model first:
```bash
python scripts/run_pipeline.py train --quick
```

#### Memory Error During Training

```
MemoryError: Unable to allocate array
```

**Solution:**
- Reduce dataset size in config
- Use `--quick` flag for training
- Increase system swap space

#### Import Errors

```
ModuleNotFoundError: No module named 'src'
```

**Solution:** Run from project root:
```bash
cd /path/to/qc_model
python scripts/run_pipeline.py ...
```

### Getting Help

1. Check the logs for detailed error messages
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure GITHUB_TOKEN is set for data extraction
4. Try demo data first to isolate issues

---

## Integration with Shapley Payouts

The p0 prediction feeds directly into the payout formula:

```
d₁ = β × p₀ × V
```

Where:
- `d₁` = First QC marginal value
- `β` = Confidence scaling coefficient (typically 0.15-0.35)
- `p₀` = Model prediction (this model's output)
- `V` = Total task value

Higher p₀ → Higher expected pass → More value to worker
Lower p₀ → Lower expected pass → More value to QC reviewer

---

## Next Steps

1. **Customize for your domain**: Modify feature engineering for non-code tasks
2. **Add monitoring**: Track calibration drift in production
3. **Retrain schedule**: Set up periodic retraining (weekly/monthly)
4. **A/B testing**: Compare model predictions to actual outcomes

---

## License

Proprietary - Internal use only.
