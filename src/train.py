import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
import mlflow
import mlflow.sklearn
import pickle
import os
import json
from datetime import datetime

# ── Config ────────────────────────────────────────────
DATA_PATH     = "data/processed/ratings_clean.csv"
MODEL_DIR     = "models"
BASELINE_PATH = "data/processed/popularity_baseline.csv"

ALS_CONFIG = {
    "factors"        : 50,   # number of latent factors
    "iterations"     : 20,   # training iterations
    "regularization" : 0.1,  # prevents overfitting
    "alpha"          : 40,   # confidence scaling
}

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Step 1: Load Data ─────────────────────────────────
def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df):,} interactions")
    print(f"✓ Users: {df['user_id'].nunique():,}")
    print(f"✓ Items: {df['item_id'].nunique():,}")
    return df

# ── Step 2: Encode user/item IDs ──────────────────────
def encode_ids(df):
    print("\nEncoding user and item IDs...")

    # Map string IDs to integers (ALS needs integers)
    user_encoder = {u: i for i, u in enumerate(df['user_id'].unique())}
    item_encoder = {v: i for i, v in enumerate(df['item_id'].unique())}

    # Reverse mappings (for recommendations later)
    user_decoder = {i: u for u, i in user_encoder.items()}
    item_decoder = {i: v for v, i in item_encoder.items()}

    df['user_idx'] = df['user_id'].map(user_encoder)
    df['item_idx'] = df['item_id'].map(item_encoder)

    print(f"✓ Encoded {len(user_encoder):,} users")
    print(f"✓ Encoded {len(item_encoder):,} items")

    return df, user_encoder, item_encoder, user_decoder, item_decoder

# ── Step 3: Train/Test Split ──────────────────────────
def train_test_split(df):
    print("\nCreating time-based train/test split...")

    # Sort by timestamp — critical for time series
    df = df.sort_values('timestamp')

    # 80% train, 20% test — chronological
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    print(f"✓ Train: {len(train_df):,} interactions")
    print(f"✓ Test:  {len(test_df):,} interactions")
    print(f"✓ Time-based split (no data leakage) ✓")

    return train_df, test_df

# ── Step 4: Build Sparse Matrix ───────────────────────
def build_sparse_matrix(df, n_users, n_items):
    print("\nBuilding sparse interaction matrix...")

    # Convert ratings to implicit feedback
    # Any rating = interaction (1.0), weighted by confidence
    confidence = df['rating'].values * ALS_CONFIG['alpha']

    matrix = sparse.csr_matrix(
        (confidence,
         (df['user_idx'].values, df['item_idx'].values)),
        shape=(n_users, n_items)
    )

    print(f"✓ Matrix shape: {matrix.shape}")
    print(f"✓ Non-zero entries: {matrix.nnz:,}")
    print(f"✓ Sparsity: {1 - matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.4%}")

    return matrix

# ── Step 5: Train ALS Model ───────────────────────────
def train_als(train_matrix):
    print("\nTraining ALS model...")
    print(f"Config: {ALS_CONFIG}")

    model = implicit.als.AlternatingLeastSquares(
        factors        = ALS_CONFIG['factors'],
        iterations     = ALS_CONFIG['iterations'],
        regularization = ALS_CONFIG['regularization'],
        random_state   = 42
    )

    # ALS expects item-user matrix (transposed)
    model.fit(train_matrix.T)

    print("✓ ALS training complete!")
    return model

# ── Step 6: Evaluate with NDCG ───────────────────────
def evaluate_model(model, train_matrix, test_df, k=10):
    print(f"\nEvaluating model (NDCG@{k})...")

    # Only evaluate users within model's known range
    max_user_idx = train_matrix.shape[0] - 1
    valid_test = test_df[test_df['user_idx'] <= max_user_idx]

    print(f"Valid test users (known to model): {valid_test['user_idx'].nunique():,}")
    print(f"Skipped (cold start users):        {test_df['user_idx'].nunique() - valid_test['user_idx'].nunique():,}")

    test_users  = valid_test['user_idx'].unique()[:500]
    ndcg_scores = []

    for user_idx in test_users:
        actual_items = set(
            valid_test[valid_test['user_idx'] == user_idx]['item_idx'].values
        )
        if len(actual_items) == 0:
            continue

        # Skip if user row is empty in train matrix
        if train_matrix[user_idx].nnz == 0:
            continue

        rec_items = []
        try:
            result   = model.recommend(
                user_idx,
                train_matrix[user_idx],
                N=k,
                filter_already_liked_items=True
            )
            rec_items = list(result[0])
        except Exception:
            pass

        if not rec_items:
            continue

        dcg  = 0
        idcg = sum([1/np.log2(i+2) for i in range(min(len(actual_items), k))])

        for i, item in enumerate(rec_items):
            if item in actual_items:
                dcg += 1 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    print(f"✓ NDCG@{k}: {mean_ndcg:.4f}")
    return mean_ndcg

# ── Step 7: Evaluate Baseline ─────────────────────────
def evaluate_baseline(test_df, k=10):
    print(f"\nEvaluating popularity baseline (NDCG@{k})...")

    baseline_df = pd.read_csv(BASELINE_PATH)

    # Get top-k item IDs from baseline (string IDs)
    top_k_item_ids = set(baseline_df.head(k)['item_id'].values) \
        if 'item_id' in baseline_df.columns \
        else set(baseline_df.head(k).index.astype(str).values)

    test_users  = test_df['user_idx'].unique()[:500]
    ndcg_scores = []

    for user_idx in test_users:
        # Get actual item STRING IDs this user rated
        actual_items = set(
            test_df[test_df['user_idx'] == user_idx]['item_id'].values
        )
        if len(actual_items) == 0:
            continue

        dcg  = 0
        idcg = sum([1/np.log2(i+2) for i in range(min(len(actual_items), k))])

        for i, item in enumerate(list(top_k_item_ids)[:k]):
            if item in actual_items:
                dcg += 1 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    print(f"✓ Baseline NDCG@{k}: {mean_ndcg:.4f}")
    return mean_ndcg

# ── Step 8: Save Model + Mappings ─────────────────────
def save_model(model, user_encoder, item_encoder,
               user_decoder, item_decoder):
    print("\nSaving model and mappings...")

    # Save ALS model
    with open(f"{MODEL_DIR}/als_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save encoders (needed at inference time)
    with open(f"{MODEL_DIR}/user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)
    with open(f"{MODEL_DIR}/item_encoder.pkl", "wb") as f:
        pickle.dump(item_encoder, f)
    with open(f"{MODEL_DIR}/user_decoder.pkl", "wb") as f:
        pickle.dump(user_decoder, f)
    with open(f"{MODEL_DIR}/item_decoder.pkl", "wb") as f:
        pickle.dump(item_decoder, f)

    print(f"✓ Model saved to {MODEL_DIR}/als_model.pkl")
    print(f"✓ Encoders saved to {MODEL_DIR}/")

# ── Main Training Pipeline ────────────────────────────
def main():
    print("=" * 45)
    print("   ALS RECOMMENDATION MODEL TRAINING")
    print("=" * 45)

    # Start MLflow run
    mlflow.set_experiment("recommendation-als")

    with mlflow.start_run(run_name=f"als_{datetime.now().strftime('%Y%m%d_%H%M')}"):

        # Log config
        mlflow.log_params(ALS_CONFIG)

        # Pipeline
        df                                              = load_data()
        df, user_enc, item_enc, user_dec, item_dec      = encode_ids(df)
        train_df, test_df                               = train_test_split(df)

        n_users      = df['user_idx'].max() + 1
        n_items      = df['item_idx'].max() + 1

        train_matrix = build_sparse_matrix(train_df, n_users, n_items)

        model        = train_als(train_matrix)

        # Add encoded indices to test_df using same encoder
        test_df['user_idx'] = test_df['user_id'].map(user_enc).fillna(-1).astype(int)
        test_df['item_idx'] = test_df['item_id'].map(item_enc).fillna(-1).astype(int)

        # Keep only users that exist in BOTH train and test
        train_users = set(train_df['user_idx'].unique())
        test_df = test_df[
            (test_df['user_idx'] >= 0) &
            (test_df['item_idx'] >= 0) &
            (test_df['user_idx'].isin(train_users))
        ]

        print(f"✓ Test users overlapping with train: {test_df['user_idx'].nunique():,}")
        print(f"✓ Test interactions after filtering: {len(test_df):,}")

        # Evaluate
        als_ndcg      = evaluate_model(model, train_matrix, test_df)
        baseline_ndcg = evaluate_baseline(test_df)

        # Log metrics to MLflow
        mlflow.log_metric("ndcg_at_10",          als_ndcg)
        mlflow.log_metric("baseline_ndcg_at_10", baseline_ndcg)
        mlflow.log_metric("improvement",         als_ndcg - baseline_ndcg)

        # Save model
        save_model(model, user_enc, item_enc, user_dec,item_dec)

        # Final report
        print(f"\n{'=' * 45}")
        print(f"   TRAINING COMPLETE")
        print(f"{'=' * 45}")
        print(f"ALS NDCG@10      : {als_ndcg:.4f}")
        print(f"Baseline NDCG@10 : {baseline_ndcg:.4f}")
        print(f"Improvement      : {als_ndcg - baseline_ndcg:+.4f}")

        if als_ndcg > baseline_ndcg:
            print(f"✓ ALS BEATS BASELINE — model is worth deploying!")
        else:
            print(f"✗ ALS does not beat baseline — tune hyperparameters")

        print(f"{'=' * 45}")
        print(f"✓ MLflow run logged")
        print(f"✓ Model saved to models/")

if __name__ == "__main__":
    main()