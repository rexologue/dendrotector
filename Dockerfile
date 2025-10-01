# syntax=docker/dockerfile:1.7

# Базовый образ (CUDA runtime)
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---------- Builder: собираем и кешируем wheels ----------
FROM base AS builder

# Инструменты сборки для git-пакетов и C/CPP расширений
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Только манифест зависимостей — максимально высокий кэш-хит
COPY requirements_docker.txt /app/requirements.txt

# Кэш pip между билдами (BuildKit) + сборка колёс
# /root/.cache/pip будет сохранён как build cache и не будет каждый раз всё тянуть заново
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip wheel --wheel-dir=/wheels -r requirements.txt

# ---------- Runtime: тонкий рантайм без тулчейнов ----------
FROM base AS runtime

# Рантайм-зависимости (без компилятора): git для HF и libgl1 для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 cudnn9-cuda-12 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ставим из заранее собранных колёс (без сети и без компиляции)
COPY --from=builder /wheels /wheels
COPY requirements_docker.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels

# Далее — только твой код (меняется часто, не ломает кэш зависимостей)
COPY . .

# Конфиги по умолчанию (можно переопределить в docker run)
ENV PORT=8000 \
    DENDROTECTOR_DEVICE= \
    HF_HOME=/root/.dendrocache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.dendrocache/huggingface

EXPOSE 8000

CMD ["/bin/bash","-lc","uvicorn dendrotector.api:app --host 0.0.0.0 --port ${PORT}"]
