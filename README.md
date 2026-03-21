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