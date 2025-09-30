#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="dendrotector-api"
PORT=8000
DEVICE="auto"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [--port PORT] [--device {cpu|cuda|cuda:N}] [--no-build]

Options:
  --port PORT       Host port to expose the API on (default: 8000).
  --device VALUE    Execution device for the detector. Accepts "cpu", "cuda", or
                    "cuda:N" to target a specific GPU index (default: auto).
  --no-build        Skip rebuilding the Docker image before running.
  -h, --help        Show this help message and exit.
USAGE
}

SHOULD_BUILD=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-build)
            SHOULD_BUILD=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! [[ $PORT =~ ^[0-9]+$ ]]; then
    echo "Error: port must be a numeric value." >&2
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
        echo "Error: unsupported device '$DEVICE'. Use cpu, cuda, or cuda:N." >&2
        exit 1
        ;;
esac

if [[ $SHOULD_BUILD -eq 1 ]]; then
    docker build -t "$IMAGE_NAME" .
fi

RUN_ARGS=(docker run --rm -p "${PORT}:${PORT}" -e "PORT=${PORT}")

if [[ -n $DEVICE_ENV ]]; then
    RUN_ARGS+=( -e "DENDROTECTOR_DEVICE=${DEVICE_ENV}" )
fi

if [[ ${GPU_ARGS[*]-} ]]; then
    RUN_ARGS+=( "${GPU_ARGS[@]}" )
fi

RUN_ARGS+=("$IMAGE_NAME")

echo "Executing: ${RUN_ARGS[*]}"
"${RUN_ARGS[@]}"
