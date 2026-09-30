"""Command line entry point for local testing.

    python main.py fire_1.jpg
    python test_pipeline.py fire_1.jpg --pretty

Prints the JSON result and exits.  No GUI, no browser, no web server.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

from .analyzer import FlameAnalyzer
from .config import Settings
from .errors import AnalysisError
from .imaging import imread

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flame-analyzer",
        description="Run the FlameAnalyzer inference pipeline on a local image and print JSON.",
    )
    parser.add_argument("image", help="path to the image file to analyse")
    parser.add_argument("--det-model", default=None, help="path to OBJ_best.pt")
    parser.add_argument("--seg-model", default=None, help="path to SEG_best.pt")
    parser.add_argument("--dataset", default=None, help="path to flame_dataset.json")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=None, help="detection confidence threshold")
    parser.add_argument("--clusters", type=int, default=None, help="fixed cluster count (k)")
    parser.add_argument(
        "--k-selection",
        choices=("silhouette", "fixed"),
        default=None,
        help="how k is chosen: silhouette search (default) or the fixed --clusters value",
    )
    parser.add_argument("--device", default=None, help="torch device, e.g. '0' or 'cpu'")
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help=(
            "omit the base64 PNG flame mask from the printed JSON. The mask is "
            "produced either way; this only keeps terminal output readable."
        ),
    )
    parser.add_argument(
        "--pretty", action="store_true", help="indent the JSON output (default: compact)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="enable backend logging (errors only by default)"
    )
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    overrides: dict[str, object] = {}
    if args.det_model:
        overrides["detection_model_path"] = args.det_model
    if args.seg_model:
        overrides["segmentation_model_path"] = args.seg_model
    if args.dataset:
        overrides["dataset_path"] = args.dataset
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.conf is not None:
        overrides["detection_conf"] = args.conf
    if args.clusters is not None:
        overrides["n_clusters"] = args.clusters
    if args.k_selection:
        overrides["k_selection"] = args.k_selection
    if args.device:
        overrides["device"] = args.device
    if args.no_mask:
        overrides["emit_mask"] = False
    return settings.with_overrides(**overrides) if overrides else settings


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline once and print the JSON result.  Returns a shell exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        image = imread(args.image)
        analyzer = FlameAnalyzer(_settings_from_args(args))
        result = analyzer.analyze_image(image)
    except AnalysisError as exc:
        print(json.dumps({"success": False, "error": exc.to_dict()}, indent=2 if args.pretty else None))
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130

    print(json.dumps(result, indent=2 if args.pretty else None))
    return 0 if result.get("success") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
