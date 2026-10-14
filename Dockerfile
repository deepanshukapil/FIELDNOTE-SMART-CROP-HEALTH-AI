FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PORT=3000
WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.14.1+cpu torchvision==0.29.1+cpu && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY server.py ./
COPY src/ ./src/
COPY models/crop_risk.joblib models/leaf_cnn.pt models/leaf_classes.json ./models/
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 3000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-3000}"]
