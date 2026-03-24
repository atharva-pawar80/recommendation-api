# Real-Time Product Recommendation API

Real-time personalized product recommendations using 
ALS Collaborative Filtering, deployed as a production REST API.

## Live Demo
- API Docs: https://recommendation-api-production-9231.up.railway.app/docs
- Health: https://recommendation-api-production-9231.up.railway.app/health

## Business Problem
E-commerce platforms show generic popular items to all users,
losing 20-35% of potential cross-sell revenue. This system
serves personalized recommendations under 100ms with cold
start handling for new users.

## Architecture
User Request → FastAPI → Redis Cache Check → ALS Model / Fallback → JSON Response

## Tech Stack
- Model: ALS Collaborative Filtering (implicit library)
- API: FastAPI + Pydantic validation
- Cache: Redis
- Container: Docker
- Deployment: Railway
- Tracking: MLflow
- Data: 7.8M Amazon Electronics interactions

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/recommend/{user_id}` | GET | Personalized recommendations |
| `/recommend/batch` | POST | Batch recommendations |

## Example Response
```json
{
  "user_id": "A1N63KPEPN5HVU",
  "recommendations": [
    {"item_id": "B0074BW614", "score": 4.4915},
    {"item_id": "B00DR0PDNE", "score": 3.9310}
  ],
  "model_version": "als_v1",
  "served_from": "als_model",
  "total": 5
}
```

## Key Design Decisions

**Why ALS?**
7.8M interaction matrix is 99.99% sparse.
ALS is specifically designed for sparse implicit feedback data.

**Why Redis caching?**
Precomputed recommendations served in ~2ms vs ~50ms for
live inference. Critical for p95 latency target.

**Why cold start fallback?**
40% of production users are new. Popularity fallback ensures
zero failures for unknown users.

**Why time-based split?**
Random splits leak future data into training. Time-based
splits simulate real production conditions.

## Run Locally
```bash
git clone https://github.com/atharva-pawar80/recommendation-api
cd recommendation-api
pip install -r requirements_full.txt
python src/train.py
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

## Project Structure
```
recommendation-api/
├── api/
│   ├── main.py          # FastAPI app
│   ├── recommender.py   # ALS inference + Redis
│   └── schemas.py       # Pydantic models
├── src/
│   └── train.py         # Model training + MLflow
├── notebook/
│   └── 01_eda.ipynb     # EDA + baseline
├── models/              # Trained model artifacts
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## What I Learned
- ALS collaborative filtering for sparse implicit feedback
- Cold start problem and fallback strategies
- FastAPI production patterns (validation, error handling)
- Redis caching for low latency serving
- Docker containerization
- CI/CD with Railway deployment