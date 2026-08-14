#!/usr/bin/env python3
"""Run the public Growth-auto demo on an image or directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# The recorded training/evaluation launcher used the legacy cuDNN convolution
# path for this checkpoint. Set it before importing PyTorch.
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import torch
import yaml

from posture_estimation import GrowthAutoEstimator, write_summary_csv
from utils.io import iter_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Banana-Gpose segmentation and Growth-auto posture estimation."
    )
    parser.add_argument("--weights", default="weights/Banana-Gpose-best.pt")
    parser.add_argument("--source", default="demo/images")
    parser.add_argument("--output", default="demo/results")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None, help="CUDA device such as 0, or cpu.")
    parser.add_argument(
        "--sam-model",
        default=None,
        help="SAM/SAM2 checkpoint name or local path. The default is downloaded by Ultralytics when absent.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    requested_device = str(
        args.device if args.device is not None else config.get("device", "0")
    )
    device = (
        "cpu"
        if args.device is None and requested_device != "cpu" and not torch.cuda.is_available()
        else requested_device
    )
    sam_model = str(args.sam_model or config.get("sam_model", "sam2.1_t.pt"))

    images = iter_images(args.source)
    if not images:
        raise SystemExit(f"No supported images found in: {args.source}")

    estimator = GrowthAutoEstimator(
        weights=args.weights,
        device=device,
        confidence=float(config.get("confidence", 0.25)),
        grounding_threshold=float(config.get("grounding_threshold", 0.18)),
        sam_model=sam_model,
        extend_ratio=float(config.get("extend_ratio", 0.30)),
        allow_large_sam_prior=bool(config.get("allow_large_sam_prior", True)),
    )

    output = Path(args.output)
    rows = []
    for image in images:
        result = estimator.process(image, output)
        rows.append(result)
        angle = (
            f"{result.signed_angle_from_vertical_deg:+.3f} deg"
            if result.signed_angle_from_vertical_deg is not None
            else "n/a"
        )
        print(f"{result.image}: {result.status}, posture angle={angle}")

    write_summary_csv(output / "results.csv", rows)
    ok = sum(row.status == "ok" for row in rows)
    print(f"Completed {len(rows)} images: {ok} successful. Results: {output}")
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
