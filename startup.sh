#!/bin/bash

# Create directories
mkdir -p data/processed

# Generate popularity baseline if not exists
python -c "
import pandas as pd
import os

baseline_path = 'data/processed/popularity_baseline.csv'

if not os.path.exists(baseline_path):
    print('Generating popularity baseline...')
    # Create simple hardcoded baseline from model encoders
    import pickle
    with open('models/item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    
    # Top items from encoder (first 20)
    items = list(item_encoder.keys())[:20]
    df = pd.DataFrame({
        'item_id': items,
        'rating_count': range(20, 0, -1),
        'avg_rating': [4.5] * 20
    })
    df.to_csv(baseline_path, index=False)
    print('Baseline generated!')
"

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
