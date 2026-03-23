FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/popularity_baseline.csv ./data/processed/
COPY data/processed/ratings_clean.csv ./data/processed/

# Expose port
EXPOSE 8080

# Start command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]