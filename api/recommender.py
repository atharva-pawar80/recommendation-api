import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import redis
import json
import os

MODEL_DIR = "models"
CACHE_TTL = 3600

class Recommender:
    def __init__(self):
        self.model         = None
        self.user_encoder  = None
        self.item_encoder  = None
        self.user_decoder  = None
        self.item_decoder  = None
        self.train_matrix  = None
        self.redis_client  = None
        self.model_version = "als_v1"
        self.is_loaded     = False
        self.redis_ok      = False

    def load_model(self):
        print("Loading ALS model...")
        try:
            with open(f"{MODEL_DIR}/als_model.pkl",    "rb") as f:
                self.model        = pickle.load(f)
            with open(f"{MODEL_DIR}/user_encoder.pkl", "rb") as f:
                self.user_encoder = pickle.load(f)
            with open(f"{MODEL_DIR}/item_encoder.pkl", "rb") as f:
                self.item_encoder = pickle.load(f)
            with open(f"{MODEL_DIR}/user_decoder.pkl", "rb") as f:
                self.user_decoder = pickle.load(f)
            with open(f"{MODEL_DIR}/item_decoder.pkl", "rb") as f:
                self.item_decoder = pickle.load(f)

            self._build_matrix()
            self.is_loaded = True
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            self.is_loaded = False

    def _build_matrix(self):
        print("Building interaction matrix...")
        try:
            n_users = len(self.user_encoder)
            n_items = len(self.item_encoder)
            self.train_matrix = sparse.csr_matrix(
                (n_users, n_items)
            )
            print(f"✓ Matrix built: {self.train_matrix.shape}")
        except Exception as e:
            print(f"✗ Matrix build failed: {e}")

    def connect_redis(self):
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_ok = True
            print("✓ Redis connected")
        except Exception:
            self.redis_ok = False
            print("⚠ Redis not available — running without cache")

    def recommend(self, user_id: str, n: int = 10):
        cache_key = f"rec:{user_id}:{n}"
        if self.redis_ok:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached), "cache"
            except Exception:
                pass

        if user_id in self.user_encoder:
            items       = self._als_recommend(
                self.user_encoder[user_id], n
            )
            served_from = "als_model"
        else:
            items       = self._popularity_fallback(n)
            served_from = "cold_start_fallback"

        if self.redis_ok and items:
            try:
                self.redis_client.setex(
                    cache_key, CACHE_TTL, json.dumps(items)
                )
            except Exception:
                pass

        return items, served_from

    def _als_recommend(self, user_idx: int, n: int):
        try:
            # Build user vector from encoder
            n_items = len(self.item_encoder)
            user_vector = sparse.csr_matrix((1, n_items))

            result    = self.model.recommend(
                user_idx,
                user_vector,
                N=n,
                filter_already_liked_items=False
            )
            item_indices = result[0]
            scores       = result[1]

            items = []
            for idx, score in zip(item_indices, scores):
                item_id = self.item_decoder.get(
                    int(idx), f"item_{idx}"
                )
                items.append({
                    "item_id": item_id,
                    "score"  : round(float(score), 4)
                })
            return items
        except Exception as e:
            print(f"ALS inference error: {e}")
            return self._popularity_fallback(n)

    def _popularity_fallback(self, n: int):
        try:
            baseline_path = "data/processed/popularity_baseline.csv"
            if os.path.exists(baseline_path):
                df    = pd.read_csv(baseline_path)
                items = []
                for _, row in df.head(n).iterrows():
                    items.append({
                        "item_id": str(row.get(
                            'item_id', row.name
                        )),
                        "score"  : round(float(
                            row.get('avg_rating', 4.5)
                        ), 4)
                    })
                return items
            else:
                top_items = list(
                    self.item_decoder.values()
                )[:n]
                return [
                    {"item_id": str(i), "score": 4.5}
                    for i in top_items
                ]
        except Exception:
            return [
                {"item_id": f"popular_{i}", "score": 4.5}
                for i in range(n)
            ]