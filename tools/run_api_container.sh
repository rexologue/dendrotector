#!/usr/bin/env bash
set -Eeuo pipefail

# Helper for building and launching the DendroDetector API container fast.

# Env:
#   IMAGE_NAME         – image tag (default: dendrotector-api)
#   PORT               – host/container port (default: 8000)
#   DEVICE             – auto|cpu|cuda|cuda:N (default: auto)
#   DENDROCACHE_PATH   – host dir for HF/cache (default: ~/.dendrocache)
#   DO_BUILD           – 1 to (re)build image, 0 to skip (default: 0)
#   DEV                – 1 to mount source tree as a volume (fast iteration; default: 0)

IMAGE_NAME=${IMAGE_NAME:-dendrotector-api}
PORT=${PORT:-8000}
DEVICE=${DEVICE:-auto}
DEV=${DEV:-0}

USER_SUPPLIED_CACHE=${DENDROCACHE_PATH-}
RAW_CACHE_PATH=${DENDROCACHE_PATH:-$HOME/.dendrocache}

HOST_CACHE_PATH=$(
RAW_CACHE_PATH="$RAW_CACHE_PATH" python - <<'PY'
import os
raw = os.environ["RAW_CACHE_PATH"]
print(os.path.abspath(os.path.expanduser(raw)))
PY
)

if [[ -z ${HOST_CACHE_PATH} ]]; then
  echo "Failed to resolve cache directory path." >&2
  exit 1
fi

# Container cache path: respect explicit host path, else default to /root/.dendrocache
if [[ -n ${USER_SUPPLIED_CACHE} ]]; then
  CONTAINER_CACHE_PATH="${HOST_CACHE_PATH}"
else
  CONTAINER_CACHE_PATH="/root/.dendrocache"
fi

HF_CACHE_CONTAINER="${CONTAINER_CACHE_PATH}/huggingface"

mkdir -p -- "${HOST_CACHE_PATH}" "${HOST_CACHE_PATH}/huggingface"

if ! [[ ${PORT} =~ ^[0-9]+$ ]]; then
  echo "Error: PORT must be numeric (got '${PORT}')." >&2
  exit 1
fi

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
  -e "DENDROCACHE_PATH=${CONTAINER_CACHE_PATH}"
  -e "HF_HOME=${HF_CACHE_CONTAINER}"
  -e "HUGGINGFACE_HUB_CACHE=${HF_CACHE_CONTAINER}"
  -v "${HOST_CACHE_PATH}:${CONTAINER_CACHE_PATH}"
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

echo "Executing: ${RUN_ARGS[*]}"
exec "${RUN_ARGS[@]}"
