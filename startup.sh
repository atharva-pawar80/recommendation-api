#!/bin/bash
mkdir -p data/processed

python -c "
import pickle, os, pandas as pd

baseline_path = 'data/processed/popularity_baseline.csv'
if not os.path.exists(baseline_path):
    with open('models/item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    items = list(item_encoder.keys())[:20]
    df = pd.DataFrame({
        'item_id': items,
        'rating_count': range(20, 0, -1),
        'avg_rating': [4.5] * 20
    })
    df.to_csv(baseline_path, index=False)
    print('Baseline generated!')
"

uvicorn api.main:app --host 0.0.0.0 --port 7860
