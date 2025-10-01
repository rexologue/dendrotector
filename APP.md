# Dendrotector service deployment

This guide covers running the packaged FastAPI service with Docker. The
repository bundles a helper script that builds the image, forwards the required
ports, and binds the shared model cache so that large checkpoints persist across
container restarts.【F:tools/run_api_container.sh†L1-L109】

## Launching the container

Use `tools/run_api_container.sh` to build the image and start the API server. The
wrapper exposes three environment variables:

- `PORT` – host port forwarded to the container (defaults to `58000`).
- `DEVICE` – `auto`, `cpu`, `cuda`, or `cuda:N` to pin a specific GPU.
- `DENDROCACHE_PATH` – optional override for the shared model cache.

```bash
export PORT=8080
export DEVICE=cuda:0
export DENDROCACHE_PATH=~/dendrocache  # optional; defaults to ~/.dendrocache
./tools/run_api_container.sh
```

When `DENDROCACHE_PATH` is unset the script mounts the host’s `~/.dendrocache`
into `/root/.dendrocache` inside the container. If you do set
`DENDROCACHE_PATH`, its absolute path is reused both on the host and inside the
container so that the directory is always a Docker volume. In either case the
script exports `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to the same bind mount,
ensuring that the Hugging Face Hub never duplicates checkpoints.【F:tools/run_api_container.sh†L1-L109】

The resulting container binds uvicorn to `0.0.0.0:${PORT}`, meaning requests sent
to `http://<host>:${PORT}` reach the FastAPI application directly without extra
port forwarding.【F:Dockerfile†L16-L20】【F:dendrotector/api.py†L8-L116】

## Hugging Face authentication

Private Hugging Face repositories require a token. Export it before running the
helper script and the API will log in automatically during the first model
initialisation. The wrapper forwards `DENDROTECTOR_HF_TOKEN`, `HF_TOKEN`, and
`HUGGING_FACE_HUB_TOKEN` into the container without echoing secrets.

```bash
export DENDROTECTOR_HF_TOKEN="hf_..."
./tools/run_api_container.sh
```

Interactive authentication via `huggingface-cli login` remains available inside
the running container if you prefer that approach.【F:dendrotector/__init__.py†L47-L122】

## Submitting detection requests

After the container is up you can submit a detection job with a simple `curl`
command. The response is a ZIP archive containing per-instance artefacts
alongside a `summary.json` file with the total instance count.【F:dendrotector/api.py†L58-L116】

```bash
curl -X POST "http://localhost:${PORT}/detect?top_k=3" \
  -F "image=@/path/to/trees.jpg" \
  --output detections.zip
```

## Troubleshooting image builds

The public PyPI wheel (`groundingdino-py`) omits CUDA sources, leading to build
failures such as
`cc1plus: fatal error: .../groundingdino/models/GroundingDINO/csrc/vision.cpp: No such file or directory`.
This project pins the official Git repository instead, which includes the needed
sources. When building on Debian or Ubuntu, install the system dependencies
before running `pip install` inside the container:

```bash
sudo apt-get update
sudo apt-get install -y build-essential ninja-build
```
