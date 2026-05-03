FFROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.0 \
    pydantic==2.7.0 \
    pandas==2.2.0 \
    numpy==1.26.0 \
    scikit-learn==1.4.0 \
    implicit==0.7.2 \
    scipy==1.13.0 \
    redis==5.0.4 \
    prometheus-fastapi-instrumentator==7.0.0

COPY api/ ./api/
COPY models/ ./models/
COPY dashboard.html .
COPY startup.sh .

RUN mkdir -p data/processed

EXPOSE 7860

CMD ["bash", "startup.sh"]