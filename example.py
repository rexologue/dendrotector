#!/usr/bin/env python3
"""
Example runner for DendroDetector.

Creates per-instance folders like:
output/
  instance_00/
    overlay.png
    mask.png
    bbox.png
    report.json
"""

from __future__ import annotations

import sys
sys.path.append("dendrotector")

import json
import argparse
from pathlib import Path

from detector import DendroDetector

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run DendroDetector on a single image.")
    p.add_argument("image", type=Path, help="Path to input image.")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory to store instance_* folders.")
    p.add_argument("--models-dir", type=Path, default=Path("~/.dendrocache"),
                   help="Models/cache root (will contain groundingdino/, sam2/, specifier/).")
    p.add_argument("--device", default=None,
                   help="Computation device, e.g. 'cuda', 'cuda:0' or 'cpu'. Defaults to CUDA if available.")
    p.add_argument("--box-threshold", type=float, default=0.3, help="GroundingDINO box confidence threshold.")
    p.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text confidence threshold.")
    p.add_argument("--top-k", type=int, default=5, help="How many top species predictions to keep.")
    p.add_argument("--multimask-output", action="store_true",
                   help="If set, SAM2 proposes multiple masks and the best one is selected.")
    p.add_argument("--print-reports", action="store_true",
                   help="Print each report.json to stdout after creation.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    image_path = args.image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()

    if not image_path.exists():
        print(f"[error] Image not found: {image_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create detector and run
    det = DendroDetector(
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        models_dir=models_dir,
    )

    try:
        instance_dirs = det.detect(
            image_path=image_path,
            output_dir=output_dir,
            top_k=args.top_k,
            multimask_output=args.multimask_output,
        )
    except Exception as e:
        print(f"[error] Detection failed: {e}", file=sys.stderr)
        return 1

    if not instance_dirs:
        print("[info] No instances detected.")
        return 0

    print(f"[ok] Created {len(instance_dirs)} instance folder(s) under: {output_dir}")
    for i, inst in enumerate(instance_dirs):
        print(f"  - instance_{i:02d}: {inst}")

        if args.print_reports:
            rpt = inst / "report.json"
            try:
                data = json.loads(rpt.read_text(encoding="utf-8"))
                print(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"    [warn] Could not read report.json: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
