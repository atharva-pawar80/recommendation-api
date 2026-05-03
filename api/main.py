from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
    HealthResponse
)
from api.recommender import Recommender
import time
import json
from datetime import datetime

# ── App setup ─────────────────────────────────────────
app = FastAPI(
    title       = "Product Recommendation API",
    description = "Real-time product recommendations using ALS collaborative filtering",
    version     = "1.0.0"
)
# ── Dashboard route ───────────────────────────────────
@app.get("/")
def serve_dashboard():
    return FileResponse("dashboard.html")
# ── CORS middleware ───────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Prometheus metrics ────────────────────────────────
Instrumentator().instrument(app).expose(app)

# ── Request log storage (in memory) ──────────────────
request_logs = []
MAX_LOGS     = 1000
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

    # ── Log request for monitoring ────────────────────
    log_entry = {
        "timestamp"  : datetime.now().isoformat(),
        "user_id"    : user_id,
        "served_from": served_from,
        "latency_ms" : latency_ms,
        "n"          : n,
        "is_cold_start": served_from == "cold_start_fallback"
    }
    request_logs.append(log_entry)
    if len(request_logs) > MAX_LOGS:
        request_logs.pop(0)

    print(f"[{user_id}] served_from={served_from} latency={latency_ms}ms")

    return RecommendationResponse(
        user_id         = user_id,
        recommendations = [RecommendedItem(**i) for i in items],
        model_version   = recommender.model_version,
        served_from     = served_from,
        total           = len(items)
    )

# ── Monitoring data endpoint ──────────────────────────
@app.get("/monitoring/stats")
def get_monitoring_stats():
    if not request_logs:
        return {
            "total_requests"   : 0,
            "avg_latency_ms"   : 0,
            "cold_start_rate"  : 0,
            "cache_hit_rate"   : 0,
            "served_from_breakdown": {}
        }

    total         = len(request_logs)
    avg_latency   = round(sum(r['latency_ms'] for r in request_logs) / total, 2)
    cold_starts   = sum(1 for r in request_logs if r['is_cold_start'])
    cache_hits    = sum(1 for r in request_logs if r['served_from'] == 'cache')

    served_breakdown = {}
    for r in request_logs:
        served_breakdown[r['served_from']] = served_breakdown.get(r['served_from'], 0) + 1

    return {
        "total_requests"      : total,
        "avg_latency_ms"      : avg_latency,
        "cold_start_rate"     : round(cold_starts / total * 100, 1),
        "cache_hit_rate"      : round(cache_hits / total * 100, 1),
        "served_from_breakdown": served_breakdown,
        "recent_requests"     : request_logs[-10:]
    }

# ── Monitoring logs endpoint ──────────────────────────
@app.get("/monitoring/logs")
def get_logs():
    return {
        "logs" : request_logs[-50:],
        "total": len(request_logs)
    }