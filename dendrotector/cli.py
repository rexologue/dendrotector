"""Command line interface for running dendrotector."""
from __future__ import annotations

import argparse
from pathlib import Path

from .detector import DendroDetector
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
        "--classify-species",
        action="store_true",
        help=(
            "After detection, run the pretrained tree species classifier on each "
            "instance and save the results."
        ),
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
    results = detector.detect(args.image, args.output_dir, multimask_output=args.multimask)

    if not results:
        print("No trees or shrubs detected.")
        return

    print(f"Detected {len(results)} instances. Metadata saved to {args.output_dir}.")

    if not args.classify_species:
        return

    taxonomy_dir = args.output_dir / "species"
    identifier = SpeciesIdentifier(
        model_name_or_path=args.species_model,
        device=args.species_device or args.device,
        top_k=args.species_top_k,
        crop_padding=args.species_crop_padding,
        apply_mask=not args.species_keep_background,
        batch_size=args.species_batch_size,
        models_dir=args.models_dir,
    )
    predictions = identifier.identify(args.image, results, taxonomy_dir)

    if not predictions:
        print("Species identifier did not return any predictions.")
    else:
        print(f"Classified species for {len(predictions)} instances. Metadata saved to {taxonomy_dir}.")


if __name__ == "__main__":
    main()
