"""FastAPI application for running DendroDetector inference remotely.

The module exposes a ``FastAPI`` application that loads the heavy computer
vision models once and serves an HTTP endpoint for performing detections.  A
client can upload an image, and the service will respond with a ZIP archive
containing the per-instance artefacts produced by :class:`~dendrotector.detector.DendroDetector`.

Typical usage when deploying on a remote machine::

    uvicorn dendrotector.api:app --host 0.0.0.0 --port 8000

The detector device can be overridden by setting the ``DENDROTECTOR_DEVICE``
environment variable before starting the server (for example ``cpu`` or
``cuda:0``).

"""
from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import io
import json
import logging
import zipfile
from typing import List
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from huggingface_hub import login

from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

from detector import DendroDetector, GROUNDING_WEIGHTS, GROUNDING_CONFIG, SAM2_REPO, PROMPT

app = FastAPI(
    title="DendroDetector API",
    version="1.0.0",
    description=(
        "Remote inference API for the dendro-detector. Upload an image of a "
        "tree or shrub and receive a ZIP archive with masks, bounding boxes "
        "and per-instance reports."
    ),
)

_MODELS_DIR = Path("/app/models")
_DEVICE_ENV_VAR = "DENDROTECTOR_DEVICE"
_HF_TOKEN = "HF_TOKEN"
_LOGGER = logging.getLogger("dendrotector.detector")

_detector_instance: DendroDetector | None = None


def _check_models() -> bool:
    _, sam2_checkpoint_name = HF_MODEL_ID_TO_FILENAMES[SAM2_REPO]
        
    dino_config = (_MODELS_DIR / "groundingdino" / GROUNDING_CONFIG).exists()
    dino_weights = (_MODELS_DIR / "groundingdino" / GROUNDING_WEIGHTS).exists()
    bert_repo = (_MODELS_DIR / "groundingdino" / "bert").exists()
    sam2_weights = (_MODELS_DIR / "sam2" / sam2_checkpoint_name).exists()
    specifier_labels = (_MODELS_DIR / "specifier" / "labels.json").exists()
    specifier_weights = (_MODELS_DIR / "specifier" / "pytorch_model.bin").exists()
    diseaser_weights = (_MODELS_DIR / "diseaser" / "model.onnx").exists()

    return dino_config and dino_weights and bert_repo and sam2_weights and specifier_labels and specifier_weights and diseaser_weights


def _get_detector() -> DendroDetector:
    """Lazily instantiate the heavy detection pipeline."""

    global _detector_instance

    if _detector_instance is None:
        _LOGGER.info("Detector not initialised yet; performing startup")

        # Log In Hugging Face in case if models are not installed
        if not _check_models():
            token = os.getenv(_HF_TOKEN)

            if not token:
                _LOGGER.info("Models are not installed and HF TOKEN is not set! Set HF_TOKEN and run again!")
                exit(-1)

            login(token=token)

        requested_device = os.getenv(_DEVICE_ENV_VAR)

        if requested_device:
            _LOGGER.info("Initialising detector on device requested via %s=%s", _DEVICE_ENV_VAR, requested_device)
        else:
            _LOGGER.info("Initialising detector on auto-detected device")

        _detector_instance = DendroDetector(device=requested_device, models_dir=_MODELS_DIR)
        _LOGGER.info("Detector initialisation complete")

    return _detector_instance


def _write_summary(output_dir: Path, instance_dirs: List[Path]) -> Path:
    """Store a high-level summary to include inside the archive."""

    summary = {
        "instances": len(instance_dirs),
        "instance_dirs": [p.name for p in instance_dirs],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


@app.post("/detect", response_class=StreamingResponse)
async def detect(
    image: UploadFile = File(..., description="Input photograph with trees or shrubs."),
    top_k: int = 1,
    disease_score: float = 0.5,
    prompt: str = PROMPT,
    multimask_output: bool = False,
):
    """Run detection on the uploaded image and return results as a ZIP archive."""

    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.")
    
    if disease_score > 1.0 or disease_score < 0.0:
        raise HTTPException(status_code=400, detail="disease_score must be in [0, 1.0] interval.")

    filename = image.filename or "uploaded_image"
    suffix = Path(filename).suffix or ".png"

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    _LOGGER.info(
        "Received /detect request: filename=%s, top_k=%s, prompt_length=%d, multimask_output=%s",
        filename,
        top_k,
        len(prompt),
        multimask_output,
    )

    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_path = tmp_dir_path / f"input{suffix}"
        output_dir = tmp_dir_path / "detections"
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path.write_bytes(contents)
        _LOGGER.debug("Stored uploaded image at %s (%d bytes)", input_path, len(contents))

        detector = _get_detector()
        _LOGGER.info("Running detector.detect on %s", input_path)
        instance_dirs = detector.detect(
            image_path=input_path,
            output_dir=output_dir,
            top_k=top_k,
            api_format=True,
            prompt=prompt,
            multimask_output=multimask_output,
        )
        _LOGGER.info("Detection finished with %d instance(s)", len(instance_dirs))

        summary_path = _write_summary(output_dir, instance_dirs)
        _LOGGER.debug("Summary written to %s", summary_path)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            # Even if there are no detections, the archive will contain the summary.
            for file_path in sorted(output_dir.rglob("*")):
                if file_path.is_file():
                    arcname = Path("detections") / file_path.relative_to(output_dir)
                    archive.write(file_path, arcname.as_posix())
                    _LOGGER.debug("Added %s to response archive", arcname)

        zip_buffer.seek(0)
        _LOGGER.info("Prepared ZIP response for %s (%.1f KiB)", filename, len(zip_buffer.getbuffer()) / 1024)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={Path(filename).stem}_detections.zip",
                "X-Detections-Count": str(len(instance_dirs)),
                "X-Summary-File": str(Path("detections") / summary_path.relative_to(output_dir)),
            },
        )


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Lightweight healthcheck endpoint for uptime monitoring."""

    return {"status": "ok"}


__all__ = ["app"]

