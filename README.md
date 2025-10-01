# Dendrotector

<p align="center">
  <img src="assets/logo.png" alt="Stylised canopy view with the Dendrotector logotype" width="512">
</p>

Dendrotector is an end-to-end tree and shrub analysis toolkit built around the
latest open-vocabulary detection and segmentation models. The package combines
GroundingDINO for text-prompted object discovery, SAM 2 for crisp instance
masks, and a fine-tuned ViT-Large classifier to hypothesize possible species
for every detection. The result is a reproducible pipeline that produces
high-quality overlays, per-instance crops, detailed JSON reports, and
species-ranked predictions from a single photograph.

## Key capabilities

- **Open-vocabulary detection** – GroundingDINO locates trees, shrubs and bushes
  based on the prompt `"tree . shrub . bush ."`, with configurable confidence
  thresholds for both bounding boxes and text logits.【F:dendrotector/detector.py†L42-L109】
- **High-fidelity instance masks** – SAM 2 refines each detection into a binary
  mask and overlay, optionally selecting the best mask when multimask output is
  enabled.【F:dendrotector/detector.py†L111-L157】【F:dendrotector/detector.py†L182-L216】
- **Species suggestions** – A ViT-L/16 classifier fine-tuned on tree imagery
  returns the top-k most probable species for every instance, complete with
  confidences and an auto-capped `k` if fewer classes are available.【F:dendrotector/species_identifier.py†L15-L74】
- **Geometric attributes** – Basic lean-angle estimation is calculated from each
  mask, allowing quick screening of tilt or fall risk.【F:dendrotector/detector.py†L248-L321】
- **Self-contained exports** – Each detection is written to its own
  `instance_XX/` directory with the mask, overlay, bbox crop, and structured
  `report.json` describing scores, geometry, and species hypotheses.【F:dendrotector/detector.py†L160-L223】

## Project layout

```
dendrotector/
  detector.py            # High level GroundingDINO + SAM 2 runner and exporters
  species_identifier.py  # ViT-L/16 classifier wrapper with Hugging Face weights
example.py               # CLI entry point for local experimentation
requirements.txt         # Python dependencies (GroundingDINO + SAM 2 from Git)
```

The package exposes `DendroDetector` via `dendrotector.detector`. When running
from the repository root, remember to add the `dendrotector/` folder to
`PYTHONPATH` (see [`example.py`](example.py)).

## Installation

1. Create and activate a Python 3.9+ virtual environment (recommended).
2. Install dependencies from `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the GroundingDINO, SAM 2, and species classifier weights
from the Hugging Face Hub into `~/.dendrocache` (or a custom directory). The
resolver also wires Hugging Face’s cache (`HF_HOME`/`HUGGINGFACE_HUB_CACHE`) to
`~/.dendrocache/huggingface`, ensuring that every download lands inside the same
volume-backed directory rather than the ephemeral container filesystem.【F:dendrotector/__init__.py†L13-L122】

### Model cache behaviour

If you already have the checkpoints you can drop them into the cache hierarchy
and they will be used as-is. `DendroDetector` and the species classifier look
for exact filenames such as
`~/.dendrocache/huggingface/sam2/<checkpoint>.pth` before contacting the Hugging
Face Hub, falling back to a download only when the file is missing.【F:dendrotector/__init__.py†L74-L122】【F:dendrotector/detector.py†L63-L88】【F:dendrotector/species_identifier.py†L27-L44】

See [`APP.md`](APP.md) for containerised deployment and FastAPI usage, including
the `tools/run_api_container.sh` helper for spinning up the FastAPI service with
GPU passthrough and a shared model cache.

## Command-line usage

Run the detector on a single image with the helper script:

```bash
python example.py path/to/image.jpg --output-dir results/
```

Key options:

| Flag | Description |
| ---- | ----------- |
| `--device` | Force `cpu`, `cuda`, or a specific GPU (defaults to CUDA when available). |
| `--models-dir` | Directory that stores downloaded weights (`groundingdino/`, `sam2/`, `specifier/`). |
| `--box-threshold` | Minimum GroundingDINO box confidence (default `0.3`). |
| `--text-threshold` | Minimum GroundingDINO text confidence (default `0.25`). |
| `--top-k` | Number of species predictions to retain (auto-clamped to the label set). |
| `--multimask-output` | Ask SAM 2 for multiple masks and keep the highest-IoU proposal. |
| `--print-reports` | Pretty-print each generated `report.json` to stdout. |

The command exits with:
- `0` when detections succeed (including the “no instances” case),
- `1` if detection fails,
- `2` if the input image is missing.【F:example.py†L26-L104】

## Python API

Use the high-level API for tighter integration in notebooks or services:

```python
from pathlib import Path
from dendrotector.detector import DendroDetector

images = Path("./images")
outputs = Path("./results")
outputs.mkdir(exist_ok=True)

detector = DendroDetector(models_dir=Path("./weights"))
instance_dirs = detector.detect(
    image_path=images / "forest.jpg",
    output_dir=outputs / "forest",
    top_k=5,
    multimask_output=False,
)

for inst in instance_dirs:
    report = inst / "report.json"
    print(report.read_text(encoding="utf-8"))
```

`detect` returns the list of created `instance_*` directories. Each `report.json`
contains the instance type (`tree` or `shrub`), detection score, pixel-aligned
bounding box, optional lean angle, the final species pick, and the full top-k
species list. Mask alpha channels are encoded in BGRA format for OpenCV
compatibility.【F:dendrotector/detector.py†L160-L233】

## Output artifacts

For every detected instance, the pipeline saves:

- `overlay.png` – RGB image with bounding box, mask tint, and index annotation.
- `mask.png` – Binary mask stored as a semi-transparent BGRA image.
- `bbox.png` – Tight crop of the detection’s bounding box.
- `report.json` – Structured metadata including species probabilities and lean
  angle.

The parent output directory mirrors your chosen input (e.g. `results/forest/`).
If no detections are found the directory remains empty.【F:dendrotector/detector.py†L160-L233】

## Performance notes

- CUDA-capable GPUs drastically speed up GroundingDINO, SAM 2, and the species
  classifier. If CUDA is unavailable the code automatically falls back to CPU.【F:dendrotector/detector.py†L31-L88】【F:dendrotector/species_identifier.py†L31-L44】
- Lean-angle estimation expects reasonably vertical trees; very small or noisy
  masks may yield `null` angles.【F:dendrotector/detector.py†L248-L321】
- The species classifier prints a warning if `top_k` exceeds the label count and
  quietly clamps the value.【F:dendrotector/species_identifier.py†L55-L70】

## License

This repository bundles third-party models from the Hugging Face Hub. Check each
model repository for its specific license before commercial use.
