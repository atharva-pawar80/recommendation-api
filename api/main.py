from fastapi import FastAPI, HTTPException
from api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
    HealthResponse
)
from api.recommender import Recommender
import time

# ── App setup ─────────────────────────────────────────
app        = FastAPI(
    title       = "Product Recommendation API",
    description = "Real-time product recommendations using ALS collaborative filtering",
    version     = "1.0.0"
)
recommender = Recommender()

# ── Startup event ─────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    recommender.load_model()
    recommender.connect_redis()

# ── Health check ──────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status         = "healthy",
        model_loaded   = recommender.is_loaded,
        redis_connected= recommender.redis_ok
    )

# ── Main recommendation endpoint ──────────────────────
@app.get("/recommend/{user_id}",
         response_model=RecommendationResponse)
def get_recommendations(user_id: str, n: int = 10):

    # Validate n
    if n < 1 or n > 100:
        raise HTTPException(
            status_code=422,
            detail="n must be between 1 and 100"
        )

    if not recommender.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet"
        )

    start_time = time.time()

    # Get recommendations
    items, served_from = recommender.recommend(user_id, n)

    latency_ms = round((time.time() - start_time) * 1000, 2)
    print(f"[{user_id}] served_from={served_from} latency={latency_ms}ms")

    return RecommendationResponse(
        user_id         = user_id,
        recommendations = [RecommendedItem(**i) for i in items],
        model_version   = recommender.model_version,
        served_from     = served_from,
        total           = len(items)
    )

# ── Batch endpoint ────────────────────────────────────
@app.post("/recommend/batch")
def get_batch_recommendations(user_ids: list[str], n: int = 10):
    if len(user_ids) > 100:
        raise HTTPException(
            status_code=422,
            detail="Maximum 100 users per batch request"
        )

    results = {}
    for user_id in user_ids:
        items, served_from = recommender.recommend(user_id, n)
        results[user_id]   = {
            "recommendations": items,
            "served_from"    : served_from
        }
    return results