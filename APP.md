 # Dendrotector service deployment
 
 This guide covers running the packaged FastAPI service with Docker. The
 repository bundles a helper script that builds the image, forwards the required
 ports, and binds the shared model cache so that large checkpoints persist across
 container restarts.【F:tools/run_api_container.sh†L1-L109】
 
 ## Launching the container
 
Use `tools/run_api_container.sh` to build the image and start the API server. The
wrapper is driven entirely through environment variables:
 
- `PORT` – host port forwarded to the container (defaults to `8000`).
- `DEVICE` – `auto`, `cpu`, `cuda`, or `cuda:N` to pin a specific GPU.
- `MODELS_DIR` – host directory mounted at `/app/models` (defaults to `~/.dendrocache`).
- `DO_BUILD` – set to `1` to rebuild the Docker image before running.
- `DEV` – set to `1` to mount the repository into the container for live code edits.
 
```bash
PORT=8080 DEVICE=cuda:0 MODELS_DIR=~/dendrocache  # optional; defaults to ~/.dendrocache
./tools/run_api_container.sh
```
 
When `MODELS_DIR` is unset the script binds the host’s `~/.dendrocache`
directory into `/app/models` inside the container. If you do set `MODELS_DIR`
the provided absolute path is reused both on the host and inside the container
so that the directory remains a Docker volume.

The resulting container binds uvicorn to `0.0.0.0:${PORT}`, meaning requests sent
to `http://<host>:${PORT}` reach the FastAPI application directly without extra
port forwarding.【F:Dockerfile†L16-L20】【F:dendrotector/api.py†L8-L116】
 
 ## Hugging Face authentication
 
Private Hugging Face repositories require a token. Export it before running the
helper script and the API will log in automatically during the first model
initialisation. The wrapper forwards `HF_TOKEN` into the container without echoing secrets.
 
 ```bash
 HF_TOKEN="hf_..." ./tools/run_api_container.sh
 ```

## Submitting detection requests
 
After the container is up you can submit a detection job with a simple `curl`
command. The response is a ZIP archive containing per-instance artefacts
alongside a `summary.json` file with the total instance count.【F:dendrotector/api.py†L58-L116】

Each `instance_*` folder mirrors the CLI output: `overlay.png`, `bbox.png`, an
optional `disease.png` if the pathology detector fired, and `report.json` with
species probabilities plus structured disease scores.【F:dendrotector/detector.py†L249-L305】
