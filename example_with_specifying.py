"""Extended example running detection and species identification."""
from __future__ import annotations

import argparse
from pathlib import Path

from dendrotector import DendroDetector
from dendrotector.detector import DEFAULT_SPECIES_MODEL, PROMPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the image to analyse.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Where to store the generated masks and metadata.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Optional directory where detector model weights should be cached.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Computation device for the detector (e.g. 'cuda', 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--species-device",
        default=None,
        help="Computation device for the species identifier. Defaults to --device.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.3,
        help="Box confidence threshold for Grounding DINO.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Text confidence threshold for Grounding DINO.",
    )
    parser.add_argument(
        "--prompt",
        default="tree . shrub . bush .",
        help="Text prompt used to query Grounding DINO.",
    )
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="Ask SAM 2 for multiple mask hypotheses per detection.",
    )
    parser.add_argument(
        "--species-model",
        default=None,
        help=(
            "Optional Hugging Face model identifier for species classification. "
            "Defaults to rexologue/vit_large_384_for_trees."
        ),
    )
    parser.add_argument(
        "--species-top-k",
        type=int,
        default=5,
        help="Number of top predictions to keep per detection.",
    )
    parser.add_argument(
        "--species-crop-padding",
        type=float,
        default=0.05,
        help="Extra padding added around each detection before cropping.",
    )
    parser.add_argument(
        "--species-batch-size",
        type=int,
        default=4,
        help="Maximum number of crops to classify at once.",
    )
    parser.add_argument(
        "--species-no-mask",
        action="store_true",
        help="Disable applying the SAM mask before classification.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    detector = DendroDetector(
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        models_dir=args.models_dir,
    )

    species_model = args.species_model or DEFAULT_SPECIES_MODEL
    report = detector.generate_report(
        image_path=args.image,
        output_dir=args.output_dir,
        prompt=args.prompt or PROMPT,
        multimask_output=args.multimask_output,
        crop_mode="mask",
        species_model=species_model,
        species_device=args.species_device or args.device,
        species_top_k=args.species_top_k,
        species_crop_padding=args.species_crop_padding,
        species_batch_size=args.species_batch_size,
        species_apply_mask=not args.species_no_mask,
        report_filename=f"{args.image.stem}_report.json",
    )

    if not report.instances:
        print("No trees or shrubs detected.")
        return

    print(f"Detection overlay saved to {report.general.overlay_path}")
    print(f"Detection metadata saved to {report.general.detection_metadata_path}")
    if report.report_path is not None:
        print(f"Consolidated report saved to {report.report_path}")

    for instance in report.instances:
        base_summary = (
            f"instance#{instance.index} type={instance.instance_type} "
            f"bbox={instance.detection.bbox} score={instance.detection.score:.3f}"
        )
        if instance.species is None:
            print(base_summary)
            continue
        top_k_summary = ", ".join(
            f"{label} ({score:.3f})" for label, score in instance.species.top_k
        )
        print(
            f"{base_summary} species={instance.species.label!r} "
            f"top_k=[{top_k_summary}]"
        )


if __name__ == "__main__":
    main()

