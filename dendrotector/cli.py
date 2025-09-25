"""Command line interface for running dendrotector."""
from __future__ import annotations

import argparse
from pathlib import Path

from .detector import DendroDetector


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect trees and shrubs with instance segmentation")
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where detections and masks are saved",
    )
    parser.add_argument("--box-threshold", type=float, default=0.3, help="GroundingDINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold")
    parser.add_argument(
        "--multimask",
        action="store_true",
        help="Return multiple SAM masks per detection and keep the best IoU mask",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device to run on (defaults to CUDA if available)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    detector = DendroDetector(
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    results = detector.detect(args.image, args.output_dir, multimask_output=args.multimask)

    if not results:
        print("No trees or shrubs detected.")
    else:
        print(f"Detected {len(results)} instances. Metadata saved to {args.output_dir}.")


if __name__ == "__main__":
    main()
