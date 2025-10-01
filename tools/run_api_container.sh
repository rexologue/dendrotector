#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   PORT=58000 DEVICE=cuda MODELS_DIR=~/dendrocache ./tools/run_api_container.sh
#
# Description:
#   Builds (optionally) and starts the Dendrotector FastAPI container with the
#   requested device and model cache mount. Tweak behaviour through the
#   following environment variables before invoking the script:
#     IMAGE_NAME       – image tag (default: dendrotector-api)
#     PORT             – host/container port (default: 8000)
#     DEVICE           – auto|cpu|cuda|cuda:N (default: auto)
#     MODELS_DIR       – host dir bound to /app/models (default: ~/.dendrocache)
#     DO_BUILD         – 1 to (re)build the Docker image (default: 0)
#     DEV              – 1 to mount the repo as /app for live-editing (default: 0)

IMAGE_NAME=${IMAGE_NAME:-dendrotector-api}
PORT=${PORT:-8000}
DEVICE=${DEVICE:-auto}
HF_TOKEN=${HF_TOKEN:-not}
MODELS_DIR=${MODELS_DIR:-~/.dendrocache}
DEV=${DEV:-0}

if ! [[ ${PORT} =~ ^[0-9]+$ ]]; then
  echo "Error: PORT must be numeric (got '${PORT}')." >&2
  exit 1
fi

MODELS_DIR_ABS=$(python - "${MODELS_DIR}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)
mkdir -p "${MODELS_DIR_ABS}"

GPU_ARGS=()
DEVICE_ENV=""
case "${DEVICE}" in
  auto)
    # оставляем автоопределение на приложении (пустая переменная)
    DEVICE_ENV=""
    ;;
  cpu)
    DEVICE_ENV="cpu"
    ;;
  cuda)
    DEVICE_ENV="cuda"
    GPU_ARGS=(--gpus all)
    ;;
  cuda:*)
    GPU_INDEX="${DEVICE#cuda:}"
    if ! [[ ${GPU_INDEX} =~ ^[0-9]+$ ]]; then
      echo "Error: GPU index must be numeric (got '${GPU_INDEX}')." >&2
      exit 1
    fi
    DEVICE_ENV="cuda:${GPU_INDEX}"
    GPU_ARGS=(--gpus "device=${GPU_INDEX}")
    ;;
  *)
    echo "Error: unsupported DEVICE '${DEVICE}'. Use auto, cpu, cuda, or cuda:N." >&2
    exit 1
    ;;
esac

# Build (opt-in). Включаем BuildKit для кэша pip в Dockerfile.
if [[ ${DO_BUILD:-0} -eq 1 ]]; then
  echo "Building image '${IMAGE_NAME}' with BuildKit cache..."
  DOCKER_BUILDKIT=1 docker build -t "${IMAGE_NAME}" .
fi

RUN_ARGS=(
  docker run --rm
  -p "${PORT}:${PORT}"
  -e "PORT=${PORT}"
  -e "HF_TOKEN=${HF_TOKEN}"
  -v "${MODELS_DIR_ABS}:/app/models"
)

# DEV mode: монтируем исходники вместо пересборки образа
if [[ ${DEV} -eq 1 ]]; then
  RUN_ARGS+=( -v "$(pwd):/app:ro" )
fi

# Пробрасываем устройство в приложение
if [[ -n ${DEVICE_ENV} ]]; then
  RUN_ARGS+=( -e "DENDROTECTOR_DEVICE=${DEVICE_ENV}" )
fi

# GPU флаги (если нужны)
if [[ ${#GPU_ARGS[@]} -gt 0 ]]; then
  RUN_ARGS+=( "${GPU_ARGS[@]}" )
fi

RUN_ARGS+=( "${IMAGE_NAME}" )

if [[ "${HF_TOKEN}" != "not" ]]; then
  echo "Executing docker run with Hugging Face credentials forwarded (command redacted)."
else
  echo "Executing: ${RUN_ARGS[*]}"
fi

exec "${RUN_ARGS[@]}"
