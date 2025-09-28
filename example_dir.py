"""Script showcasing how to run the Dendrotector pipeline on a folder of images."""
from __future__ import annotations

import argparse
from pathlib import Path

from dendrotector import DendroDetector
from dendrotector.detector import DEFAULT_SPECIES_MODEL, PROMPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Path to the folder with images.")
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
        help="Optional directory where model weights should be cached.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Computation device (e.g. 'cuda', 'cuda:0' or 'cpu').",
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
        "--skip-species",
        action="store_true",
        help="Skip the species classification stage.",
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
        "--species-device",
        default=None,
        help="Device for the species classifier (defaults to --device).",
    )
    parser.add_argument(
        "--species-top-k",
        type=int,
        default=5,
        help="Number of top species predictions to retain per instance.",
    )
    parser.add_argument(
        "--species-crop-padding",
        type=float,
        default=0.05,
        help="Padding ratio when cropping instances for classification.",
    )
    parser.add_argument(
        "--species-batch-size",
        type=int,
        default=4,
        help="Maximum number of crops to classify at once.",
    )
    parser.add_argument(
        "--species-keep-background",
        action="store_true",
        help="Disable mask application and keep original backgrounds in crops.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("bbox", "mask"),
        default="mask",
        help="How to compute crops for classification.",
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

    image_files = [
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]
    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    for idx, image_path in enumerate(sorted(image_files)):
        out_dir = args.output_dir / str(idx)
        out_dir.mkdir(parents=True, exist_ok=True)

        species_model = None if args.skip_species else (args.species_model or DEFAULT_SPECIES_MODEL)
        report = detector.generate_report(
            image_path=image_path,
            output_dir=out_dir,
            prompt=args.prompt or PROMPT,
            multimask_output=args.multimask_output,
            crop_mode=args.crop_mode,
            species_model=species_model,
            species_device=args.species_device or args.device,
            species_top_k=args.species_top_k,
            species_crop_padding=args.species_crop_padding,
            species_batch_size=args.species_batch_size,
            species_apply_mask=not args.species_keep_background,
        )

        if not report.instances:
            print(f"[{idx}] No trees or shrubs detected in {image_path.name}.")
            continue

        print(f"[{idx}] Saved overlay to {report.general.overlay_path}")
        print(f"[{idx}] Report stored at {report.report_path}")
        for instance in report.instances:
            desc = (
                f"  instance#{instance.index} type={instance.instance_type} "
                f"bbox={instance.detection.bbox} score={instance.detection.score:.3f}"
            )
            if instance.species is not None:
                desc += f" species={instance.species.label!r} ({instance.species.score:.3f})"
            print(desc)


if __name__ == "__main__":
    main()
