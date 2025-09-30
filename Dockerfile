# Dockerfile for serving the DendroDetector API via FastAPI/uvicorn.
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Default configuration values that can be overridden at runtime.
ENV PORT=8000 \
    DENDROTECTOR_DEVICE=

EXPOSE 8000

CMD ["/bin/bash", "-lc", "uvicorn dendrotector.api:app --host 0.0.0.0 --port ${PORT}"]
