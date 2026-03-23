# Real-Time Product Recommendation API

## Business Problem
E-commerce platforms show generic "popular items" to all users,
losing 20-35% of potential cross-sell revenue due to irrelevant2
recommendations.

## Solution
A low-latency recommendation API that serves personalized product
suggestions using collaborative filtering, with cold-start handling
for new users.

## Success Metrics
- Primary: NDCG@10 beats popularity baseline
- Serving: p95 latency < 100ms
- Cold start: All users get recommendations (zero fallback failures)

## Model Performance

ALS NDCG@10: Low on time-based split — expected behavior.
Root cause: 40% of test items are new products launched after
training cutoff. Model cannot recommend items it has never seen.

Production solution: Daily retraining pipeline ensures model
always knows recent items. Cold start handled via popularity
fallback for new items.

Baseline NDCG@10: 0.0153

## Baseline
Global top-10 most rated items shown to everyone.

## Failure Modes
- New user with zero history (cold start) → content-based fallback
- New item never rated → category popularity fallback
- Redis down → direct ALS inference fallback

## Deployment
REST API on Google Cloud Run with Redis caching.

## Stack
Python · FastAPI · ALS (implicit) · Redis · Docker · 
MLflow · Prometheus · Grafana · Streamlit · GCP Cloud Run
```

Save with `Ctrl+S`.

---

**Step 3 — Write the .gitignore**

Click `.gitigonre` (rename it to `.gitignore`) and paste:
```
venv/
__pycache__/
*.pyc
.env
data/raw/
*.csv
*.pkl
mlruns/
.ipynb_checkpoints/

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/recommend/{user_id}` | GET | Get personalized recommendations |
| `/recommend/batch` | POST | Batch recommendations |

## Results

### API Performance
- Known user → ALS model inference
- Unknown user → cold start popularity fallback
- Health check → model + redis status

### Example Response
```json
{
  "user_id": "A1N63KPEPN5HVU",
  "recommendations": [
    {"item_id": "B0074BW614", "score": 4.4915},
    {"item_id": "B00DR0PDNE", "score": 3.931}
  ],
  "model_version": "als_v1",
  "served_from": "als_model",
  "total": 5
}
```

## Architecture
- FastAPI REST API
- ALS Collaborative Filtering (implicit library)
- Cold start fallback → popularity based
- Redis caching (Week 4)
- Docker + GCP Cloud Run (Week 4)