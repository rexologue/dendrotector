"""Minimal script showcasing how to run the Dendrotector pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from dendrotector import DendroDetector


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

    print(f"Saved outputs to {args.output_dir.resolve()}")
    for result in results:
        print(
            f"label={result.label!r} score={result.score:.3f} "
            f"bbox={result.bbox} mask={result.mask_path}"
        )


if __name__ == "__main__":
    main()
