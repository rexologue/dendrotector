"""Script showcasing how to run the Dendrotector pipeline on a folder of images."""
from __future__ import annotations

import argparse
from pathlib import Path

from dendrotector import DendroDetector


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

        results = detector.detect(
            image_path=image_path,
            output_dir=out_dir,
            prompt=args.prompt,
            multimask_output=args.multimask_output,
        )

        if not results:
            print(f"[{idx}] No trees or shrubs detected in {image_path.name}.")
            continue

        print(f"[{idx}] Saved outputs to {out_dir.resolve()}")
        for result in results:
            print(
                f"  label={result.label!r} score={result.score:.3f} "
                f"bbox={result.bbox} mask={result.mask_path}"
            )


if __name__ == "__main__":
    main()
