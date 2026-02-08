"""Command-line interface for change detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.change_detection import detect_change, load_image, threshold_change


def save_change_map(change_map: np.ndarray, output_path: Path) -> None:
    """Save binary change map as an 8-bit PNG."""

    output_image = (change_map * 255).astype(np.uint8)
    Image.fromarray(output_image).save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect change between two satellite images.")
    parser.add_argument("--image-a", required=True, help="Path to the earlier image.")
    parser.add_argument("--image-b", required=True, help="Path to the later image.")
    parser.add_argument("--output", required=True, help="Output PNG path for change map.")
    parser.add_argument("--band", type=int, default=1, help="Band index for multiband imagery.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold for change score (defaults to Otsu).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    image_a = load_image(args.image_a, band=args.band)
    image_b = load_image(args.image_b, band=args.band)

    result = detect_change(image_a, image_b, threshold=args.threshold)

    if args.threshold is None:
        _, used_threshold = threshold_change(result.change_score)
        print(f"Computed Otsu threshold: {used_threshold:.4f}")

    save_change_map(result.change_map, Path(args.output))


if __name__ == "__main__":
    main()
