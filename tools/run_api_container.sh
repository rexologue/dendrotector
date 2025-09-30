#!/usr/bin/env bash
set -euo pipefail

# Helper script for building and launching the DendroDetector API container.
#
# Configuration is controlled via three environment variables:
#   PORT              – host port that will be forwarded to the container.
#   DEVICE            – inference device (auto|cpu|cuda|cuda:N).
#   DENDROCACHE_PATH  – host directory used to persist downloaded model weights.
#                        If unset, defaults to "$HOME/.dendrocache" and is
#                        mounted into the container at "/root/.dendrocache".

IMAGE_NAME=${IMAGE_NAME:-dendrotector-api}
PORT=${PORT:-8000}
DEVICE=${DEVICE:-auto}

USER_SUPPLIED_CACHE=${DENDROCACHE_PATH-}
RAW_CACHE_PATH=${DENDROCACHE_PATH:-$HOME/.dendrocache}

HOST_CACHE_PATH=$(RAW_CACHE_PATH="$RAW_CACHE_PATH" python - <<'PY'
import os
raw = os.environ["RAW_CACHE_PATH"]
print(os.path.abspath(os.path.expanduser(raw)))
PY
)

if [[ -z $HOST_CACHE_PATH ]]; then
    echo "Failed to resolve cache directory path." >&2
    exit 1
fi

# The container path defaults to /root/.dendrocache unless the user explicitly
# provided DENDROCACHE_PATH, in which case we reuse the same absolute path.
if [[ -n $USER_SUPPLIED_CACHE ]]; then
    CONTAINER_CACHE_PATH="$HOST_CACHE_PATH"
else
    CONTAINER_CACHE_PATH="/root/.dendrocache"
fi

HF_CACHE_CONTAINER="${CONTAINER_CACHE_PATH}/huggingface"

mkdir -p -- "$HOST_CACHE_PATH"
mkdir -p -- "${HOST_CACHE_PATH}/huggingface"

if ! [[ $PORT =~ ^[0-9]+$ ]]; then
    echo "Error: PORT must be a numeric value (received '$PORT')." >&2
    exit 1
fi

GPU_ARGS=()
DEVICE_ENV=""
case "$DEVICE" in
    auto)
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
        if ! [[ $GPU_INDEX =~ ^[0-9]+$ ]]; then
            echo "Error: GPU index must be numeric (received '$GPU_INDEX')." >&2
            exit 1
        fi
        DEVICE_ENV="cuda:${GPU_INDEX}"
        GPU_ARGS=(--gpus "device=${GPU_INDEX}")
        ;;
    *)
        echo "Error: unsupported DEVICE '$DEVICE'. Use auto, cpu, cuda, or cuda:N." >&2
        exit 1
        ;;
esac

if [[ ${SKIP_BUILD:-0} -ne 1 ]]; then
    docker build -t "$IMAGE_NAME" .
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

if [[ -n $DEVICE_ENV ]]; then
    RUN_ARGS+=( -e "DENDROTECTOR_DEVICE=${DEVICE_ENV}" )
fi

if [[ ${GPU_ARGS[*]-} ]]; then
    RUN_ARGS+=( "${GPU_ARGS[@]}" )
fi

RUN_ARGS+=("$IMAGE_NAME")

echo "Executing: ${RUN_ARGS[*]}"
"${RUN_ARGS[@]}"
