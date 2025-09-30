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
import io
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .detector import DendroDetector, PROMPT

app = FastAPI(
    title="DendroDetector API",
    version="1.0.0",
    description=(
        "Remote inference API for the dendro-detector. Upload an image of a "
        "tree or shrub and receive a ZIP archive with masks, bounding boxes "
        "and per-instance reports."
    ),
)


_detector_instance: DendroDetector | None = None
_DEVICE_ENV_VAR = "DENDROTECTOR_DEVICE"


def _get_detector() -> DendroDetector:
    """Lazily instantiate the heavy detection pipeline."""

    global _detector_instance
    if _detector_instance is None:
        requested_device = os.getenv(_DEVICE_ENV_VAR) or None
        _detector_instance = DendroDetector(device=requested_device)
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
    prompt: str = PROMPT,
    multimask_output: bool = False,
):
    """Run detection on the uploaded image and return results as a ZIP archive."""

    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.")

    filename = image.filename or "uploaded_image"
    suffix = Path(filename).suffix or ".png"

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_path = tmp_dir_path / f"input{suffix}"
        output_dir = tmp_dir_path / "detections"
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path.write_bytes(contents)

        detector = _get_detector()
        instance_dirs = detector.detect(
            image_path=input_path,
            output_dir=output_dir,
            top_k=top_k,
            prompt=prompt,
            multimask_output=multimask_output,
        )

        summary_path = _write_summary(output_dir, instance_dirs)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            # Even if there are no detections, the archive will contain the summary.
            for file_path in sorted(output_dir.rglob("*")):
                if file_path.is_file():
                    arcname = Path("detections") / file_path.relative_to(output_dir)
                    archive.write(file_path, arcname.as_posix())

        zip_buffer.seek(0)
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

