"""Command line interface for running dendrotector."""
from __future__ import annotations

import argparse
from pathlib import Path

from .detector import DendroDetector, PROMPT
from .species_identifier import SpeciesIdentifier


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
    parser.add_argument("--prompt", default=PROMPT, help="Text prompt provided to GroundingDINO.")
    parser.add_argument(
        "--multimask",
        action="store_true",
        help="Return multiple SAM 2 masks per detection and keep the best IoU mask",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device to run on (defaults to CUDA if available)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory where model weights should be stored",
    )
    parser.add_argument(
        "--species-model",
        default=SpeciesIdentifier.DEFAULT_MODEL_ID,
        help="Hugging Face model to use for species classification.",
    )
    parser.add_argument(
        "--species-device",
        type=str,
        default=None,
        help="Computation device for the species classifier (defaults to detector device).",
    )
    parser.add_argument(
        "--species-top-k",
        type=int,
        default=5,
        help="Number of highest probability species predictions to store per detection.",
    )
    parser.add_argument(
        "--species-crop-padding",
        type=float,
        default=0.05,
        help="Extra padding around each detection when cropping for classification.",
    )
    parser.add_argument(
        "--species-batch-size",
        type=int,
        default=4,
        help="Maximum number of crops to classify simultaneously.",
    )
    parser.add_argument(
        "--species-keep-background",
        action="store_true",
        help="Disable SAM mask application and keep the original background in crops.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("bbox", "mask"),
        default="mask",
        help="How to derive the crop region before classification.",
    )
    parser.add_argument(
        "--skip-species",
        action="store_true",
        help="Skip the species classification stage and only run detection.",
    )
    parser.add_argument(
        "--report-filename",
        default=None,
        help="Optional custom filename for the generated JSON report.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    detector = DendroDetector(
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        models_dir=args.models_dir,
    )
    species_model = None if args.skip_species else args.species_model

    report = detector.generate_report(
        image_path=args.image,
        output_dir=args.output_dir,
        prompt=args.prompt,
        multimask_output=args.multimask,
        crop_mode=args.crop_mode,
        species_model=species_model,
        species_device=args.species_device or args.device,
        species_top_k=args.species_top_k,
        species_crop_padding=args.species_crop_padding,
        species_batch_size=args.species_batch_size,
        species_apply_mask=not args.species_keep_background,
        report_filename=args.report_filename,
    )

    if not report.instances:
        print("No trees or shrubs detected.")
        return

    print(f"Saved overlay to {report.general.overlay_path}")
    print(f"Saved detection metadata to {report.general.detection_metadata_path}")
    if report.report_path is not None:
        print(f"Saved consolidated report to {report.report_path}")

    if species_model is not None:
        classified = sum(1 for instance in report.instances if instance.species is not None)
        print(f"Classified species for {classified} instances using {report.general.species_model}.")


if __name__ == "__main__":
    main()
