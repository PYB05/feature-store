"""
============================================================
ML Feature Store — FastAPI Prediction Service
============================================================
The central API that serves real-time fraud predictions.

Flow for each prediction request:
    1. Receive transaction JSON via POST /predict
    2. Check Redis cache (key = transaction_id)
       → Cache HIT:  return cached result instantly (~0.1ms)
       → Cache MISS: continue to step 3
    3. Scale the input features using the saved StandardScaler
    4. Run the Random Forest model to get prediction + confidence
    5. Store result in Redis (TTL=3600s) for future cache hits
    6. Store result in PostgreSQL for permanent record
    7. Return JSON with prediction, confidence, and latency

Endpoints:
    POST /predict    — Real-time fraud prediction
    GET  /benchmark  — Compare Redis vs PostgreSQL latency
    GET  /health     — Service health check
    GET  /metrics    — Prometheus metrics for monitoring

Environment Variables:
    REDIS_HOST, REDIS_PORT
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
============================================================
"""

import os
import sys
import time
import json
import uuid
import logging
from typing import Optional

import numpy as np
import joblib
import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

# Add parent directory to path so we can import db module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import DatabaseManager

# ============================================================
# Configure Logging
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Prometheus Metrics
# ============================================================
# These metrics are scraped by Prometheus every 15 seconds
# and visualized in Grafana dashboards

REQUEST_COUNT = Counter(
    'request_count', 
    'Total number of prediction requests',
    ['endpoint', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction requests',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

FRAUD_DETECTION_RATE = Gauge(
    'fraud_detection_rate',
    'Percentage of transactions flagged as fraud'
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate',
    'Percentage of requests served from Redis cache'
)

# ============================================================
# Tracking Variables for Metrics
# ============================================================
total_predictions = 0
total_frauds = 0
total_cache_hits = 0
total_cache_requests = 0

# ============================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================
class TransactionRequest(BaseModel):
    """
    Input schema for a fraud prediction request.
    
    Fields:
        transaction_id — Unique UUID for this transaction
        amount         — Dollar amount of the transaction
        time           — Seconds elapsed since first transaction in dataset
        v1 through v28 — PCA-transformed features (anonymized for privacy)
    """
    transaction_id: str
    amount: float
    time: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float


class PredictionResponse(BaseModel):
    """Output schema for a fraud prediction."""
    transaction_id: str
    is_fraud: bool
    confidence: float
    latency_ms: float
    cache_hit: bool


# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="Fraud Detection Feature Store",
    description="Real-time fraud detection API with Redis caching and PostgreSQL persistence",
    version="1.0.0"
)

# CORS — allow the frontend to call the API from the same origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, JS) at /static
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Static files mounted from {static_dir}")

# Global variables for model, scaler, Redis, and DB connections
model = None
scaler = None
redis_client = None
db = None


@app.on_event("startup")
async def startup():
    """
    Load model and establish connections when the server starts.
    
    This runs once when uvicorn boots up, before any requests
    are served. If any connection fails, the server still starts
    but those features will be degraded.
    """
    global model, scaler, redis_client, db
    
    # --- Load the trained model ---
    model_path = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
    scaler_path = os.getenv("SCALER_PATH", "models/scaler.pkl")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Scaler loaded from {scaler_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # --- Connect to Redis ---
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    
    try:
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,  # Return strings instead of bytes
            socket_connect_timeout=5
        )
        redis_client.ping()  # Test the connection
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        logger.warning(f"Redis not available: {e} (will skip caching)")
        redis_client = None
    
    # --- Connect to PostgreSQL ---
    try:
        db = DatabaseManager()
        logger.info("Connected to PostgreSQL")
    except Exception as e:
        logger.warning(f"PostgreSQL not available: {e} (will skip persistence)")
        db = None


# ============================================================
# POST /predict — Core Prediction Endpoint
# ============================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest):
    """
    Predict whether a transaction is fraudulent.
    
    Pipeline:
        1. Check Redis cache for this transaction_id
        2. If cache miss: run model prediction
        3. Store result in Redis (TTL=1 hour) and PostgreSQL
        4. Update Prometheus metrics
    """
    global total_predictions, total_frauds, total_cache_hits, total_cache_requests
    
    start_time = time.time()
    cache_hit = False
    
    # --- Step 1: Check Redis Cache ---
    total_cache_requests += 1
    
    if redis_client:
        try:
            cached = redis_client.get(f"txn:{transaction.transaction_id}")
            if cached:
                # Cache HIT — return immediately without running the model
                result = json.loads(cached)
                latency_ms = (time.time() - start_time) * 1000
                
                total_cache_hits += 1
                CACHE_HIT_RATE.set(
                    (total_cache_hits / total_cache_requests) * 100
                )
                REQUEST_COUNT.labels(endpoint='/predict', status='cache_hit').inc()
                
                logger.info(
                    f"🟢 Cache HIT for {transaction.transaction_id} "
                    f"(latency: {latency_ms:.2f}ms)"
                )
                
                return PredictionResponse(
                    transaction_id=transaction.transaction_id,
                    is_fraud=result['is_fraud'],
                    confidence=result['confidence'],
                    latency_ms=latency_ms,
                    cache_hit=True
                )
        except Exception as e:
            logger.warning(f"Redis error: {e}")
    
    # --- Step 2: Prepare Features for Model ---
    # Build the feature DataFrame with the exact column names the model was trained on.
    # Using a named DataFrame (not a raw numpy array) silences the sklearn
    # UserWarning: "X does not have valid feature names"
    import pandas as pd
    feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    features = pd.DataFrame([[
        transaction.time,
        transaction.v1, transaction.v2, transaction.v3, transaction.v4,
        transaction.v5, transaction.v6, transaction.v7, transaction.v8,
        transaction.v9, transaction.v10, transaction.v11, transaction.v12,
        transaction.v13, transaction.v14, transaction.v15, transaction.v16,
        transaction.v17, transaction.v18, transaction.v19, transaction.v20,
        transaction.v21, transaction.v22, transaction.v23, transaction.v24,
        transaction.v25, transaction.v26, transaction.v27, transaction.v28,
        transaction.amount
    ]], columns=feature_cols)

    # --- Step 3: Scale Features ---
    # Only scale Time and Amount — V1-V28 are already PCA-transformed
    features[['Time', 'Amount']] = scaler.transform(features[['Time', 'Amount']])
    
    # --- Step 4: Run Model Prediction ---
    prediction = model.predict(features)[0]
    confidence = float(model.predict_proba(features)[0][1])  # Probability of fraud
    is_fraud = bool(prediction == 1)
    
    latency_ms = (time.time() - start_time) * 1000
    
    # --- Step 5: Update Metrics ---
    total_predictions += 1
    if is_fraud:
        total_frauds += 1
    
    PREDICTION_LATENCY.observe(latency_ms / 1000)  # Convert to seconds
    FRAUD_DETECTION_RATE.set((total_frauds / total_predictions) * 100)
    CACHE_HIT_RATE.set(
        (total_cache_hits / total_cache_requests) * 100 if total_cache_requests > 0 else 0
    )
    REQUEST_COUNT.labels(endpoint='/predict', status='success').inc()
    
    # --- Step 6: Store in Redis Cache (TTL = 1 hour) ---
    if redis_client:
        try:
            cache_data = json.dumps({
                'is_fraud': is_fraud,
                'confidence': confidence,
                'latency_ms': latency_ms
            })
            redis_client.setex(
                f"txn:{transaction.transaction_id}",
                3600,  # TTL: 1 hour
                cache_data
            )
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    # --- Step 7: Store in PostgreSQL ---
    if db:
        try:
            db.insert_transaction({
                'transaction_id': transaction.transaction_id,
                'amount': transaction.amount,
                'is_fraud': is_fraud,
                'confidence': confidence,
                'latency_ms': latency_ms,
                'source': 'api'
            })
        except Exception as e:
            logger.warning(f"Failed to store in PostgreSQL: {e}")
    
    # --- Step 8: Log and Return ---
    status_label = "FRAUD" if is_fraud else "LEGIT"
    logger.info(
        f"{status_label} | txn={transaction.transaction_id[:8]}... | "
        f"amount=${transaction.amount:.2f} | confidence={confidence:.4f} | "
        f"latency={latency_ms:.2f}ms"
    )
    
    return PredictionResponse(
        transaction_id=transaction.transaction_id,
        is_fraud=is_fraud,
        confidence=confidence,
        latency_ms=latency_ms,
        cache_hit=False
    )


# ============================================================
# GET /benchmark — Redis vs PostgreSQL Latency Comparison
# ============================================================
@app.get("/benchmark")
async def benchmark():
    """
    Run 1000 operations against Redis and PostgreSQL to compare latency.

    Improvements over naive benchmarking:
    - Redis pipelining: batches SET+GET into a single round trip
    - Connection pooling: PostgreSQL reuses pooled connections
    - Warmup phase: 100 iterations to prime caches before measuring
    - Higher iteration count: more stable averages

    Returns the average latency for each and the speedup ratio.
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    if not db:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")

    test_id = str(uuid.uuid4())
    test_data = json.dumps({'is_fraud': False, 'confidence': 0.05, 'latency_ms': 1.0})
    iterations = 1000
    warmup = 100

    # --- Warmup phase (not measured) ---
    # Prime Redis and PostgreSQL caches / connection pools so we measure
    # steady-state latency, not cold-start overhead.
    for i in range(warmup):
        redis_client.setex(f"warmup:{test_id}:{i}", 10, test_data)
        redis_client.get(f"warmup:{test_id}:{i}")
        redis_client.delete(f"warmup:{test_id}:{i}")

    for i in range(warmup):
        txn_id = str(uuid.uuid4())
        db.insert_transaction({
            'transaction_id': txn_id,
            'amount': 1.0,
            'is_fraud': False,
            'confidence': 0.01,
            'latency_ms': 0.1,
            'source': 'warmup'
        })

    # --- Benchmark Redis (pipelined SET + GET) ---
    # Pipelining sends all commands in a single round trip to the server,
    # eliminating per-command network overhead. This is how production
    # systems use Redis.
    redis_times = []
    batch_size = 50
    for batch_start in range(0, iterations, batch_size):
        batch_end = min(batch_start + batch_size, iterations)
        start = time.time()
        pipe = redis_client.pipeline()
        for i in range(batch_start, batch_end):
            pipe.setex(f"bench:{test_id}:{i}", 60, test_data)
            pipe.get(f"bench:{test_id}:{i}")
        pipe.execute()
        elapsed = (time.time() - start) * 1000
        per_op = elapsed / (batch_end - batch_start)
        for _ in range(batch_start, batch_end):
            redis_times.append(per_op)

    # Cleanup Redis benchmark keys (pipelined)
    cleanup_pipe = redis_client.pipeline()
    for i in range(iterations):
        cleanup_pipe.delete(f"bench:{test_id}:{i}")
    cleanup_pipe.execute()

    # --- Benchmark PostgreSQL (insert + read via connection pool) ---
    postgres_times = []
    for i in range(iterations):
        txn_id = str(uuid.uuid4())
        start = time.time()
        db.insert_transaction({
            'transaction_id': txn_id,
            'amount': 100.0,
            'is_fraud': False,
            'confidence': 0.05,
            'latency_ms': 1.0,
            'source': 'benchmark'
        })
        db.get_recent(limit=1)
        postgres_times.append((time.time() - start) * 1000)

    # --- Calculate Results ---
    redis_avg = sum(redis_times) / len(redis_times)
    postgres_avg = sum(postgres_times) / len(postgres_times)
    speedup = postgres_avg / redis_avg if redis_avg > 0 else 0

    result = {
        "iterations": iterations,
        "redis_avg_ms": round(redis_avg, 4),
        "redis_p95_ms": round(sorted(redis_times)[int(iterations * 0.95)], 4),
        "postgres_avg_ms": round(postgres_avg, 4),
        "postgres_p95_ms": round(sorted(postgres_times)[int(iterations * 0.95)], 4),
        "speedup_ratio": round(speedup, 2),
        "conclusion": f"Redis is {speedup:.1f}x faster than PostgreSQL"
    }

    logger.info(f"Benchmark: Redis={redis_avg:.4f}ms, PostgreSQL={postgres_avg:.4f}ms, Speedup={speedup:.1f}x")
    return result


# ============================================================
# GET /health — Health Check
# ============================================================
@app.get("/health")
async def health():
    """
    Check the health of all connected services.
    
    Used by Docker Compose health checks and load balancers
    to verify the API is ready to accept requests.
    """
    health_status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "redis_connected": False,
        "postgres_connected": False,
        "total_predictions": total_predictions,
        "total_frauds_detected": total_frauds,
    }
    
    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            health_status["redis_connected"] = True
        except Exception:
            health_status["redis_connected"] = False
    
    # Check PostgreSQL
    if db:
        try:
            count = db.get_transaction_count()
            health_status["postgres_connected"] = True
            health_status["postgres_transaction_count"] = count
        except Exception:
            health_status["postgres_connected"] = False
    
    # Overall status
    if not health_status["model_loaded"]:
        health_status["status"] = "unhealthy"
    
    return health_status


# ============================================================
# GET /metrics — Prometheus Metrics
# ============================================================
@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics in the expected text format.
    
    Prometheus scrapes this endpoint every 15 seconds.
    Grafana then queries Prometheus to render dashboards.
    
    Metrics exposed:
        - request_count:              total API calls (by endpoint/status)
        - prediction_latency_seconds: histogram of response times
        - fraud_detection_rate:       % of transactions flagged as fraud
        - cache_hit_rate:             % of requests served from Redis
    """
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================
# GET /recent — Recent Transactions (for Dashboard)
# ============================================================
@app.get("/recent")
async def recent(limit: int = 50):
    """
    Return recent transactions from PostgreSQL.
    Used by the frontend dashboard to show the live feed.
    """
    if not db:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    
    transactions = db.get_recent(limit=min(limit, 200))
    return {
        "transactions": transactions,
        "count": len(transactions),
        "total_predictions": total_predictions,
        "total_frauds": total_frauds,
    }


# ============================================================
# GET / — Dashboard (serves static/index.html)
# ============================================================
@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the fraud detection dashboard."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Fraud Detection API is running. Visit /docs for API docs."}
