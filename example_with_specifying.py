"""Extended example running detection and species identification."""
from __future__ import annotations

import argparse
from pathlib import Path

from dendrotector import DendroDetector, SpeciesIdentifier


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
        "--species-models-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory where species identification model weights should be "
            "cached. Defaults to --models-dir when omitted."
        ),
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
        default=SpeciesIdentifier.DEFAULT_MODEL_ID,
        help="Hugging Face model identifier for species classification.",
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
    parser.add_argument(
        "--species-trust-remote-code",
        action="store_true",
        help="Allow loading species models that require custom code.",
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

    results = detector.detect(
        image_path=args.image,
        output_dir=args.output_dir,
        prompt=args.prompt,
        multimask_output=args.multimask_output,
    )

    if not results:
        print("No trees or shrubs detected.")
        return

    print(f"Saved detection outputs to {args.output_dir.resolve()}")
    for result in results:
        print(
            f"label={result.label!r} score={result.score:.3f} "
            f"bbox={result.bbox} mask={result.mask_path}"
        )

    taxonomy_output_dir = args.output_dir / "species"
    species_identifier = SpeciesIdentifier(
        model_name_or_path=args.species_model,
        device=args.species_device or args.device,
        top_k=args.species_top_k,
        crop_padding=args.species_crop_padding,
        apply_mask=not args.species_no_mask,
        batch_size=args.species_batch_size,
        models_dir=args.species_models_dir or args.models_dir,
    )

    species_predictions = species_identifier.identify(
        image_path=args.image,
        detections=results,
        output_dir=taxonomy_output_dir,
    )

    if not species_predictions:
        print("Species identifier did not return any predictions.")
        return

    print(f"Saved species outputs to {taxonomy_output_dir.resolve()}")
    for prediction in species_predictions:
        top_k_summary = ", ".join(
            f"{label} ({score:.3f})" for label, score in prediction.top_k
        )
        print(
            f"crop={prediction.crop_path} label={prediction.label!r} "
            f"score={prediction.score:.3f} top_k=[{top_k_summary}]"
        )


if __name__ == "__main__":
    main()

