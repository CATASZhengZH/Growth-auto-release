"""End-to-end Growth-auto posture estimation using the frozen paper logic."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import growth_auto as algorithm
from segmentation import BananaGposeSegmenter
from utils.io import write_json
from utils.visualization import (
    draw_growth_auto_geometry,
    draw_prior,
    polygon_mask,
)


@dataclass
class PostureResult:
    image: str
    status: str
    class_name: str = ""
    confidence: float | None = None
    mask_area_px2: float | None = None
    axis_length_px: float | None = None
    angle_from_horizontal_deg: float | None = None
    angle_from_vertical_deg: float | None = None
    signed_angle_from_vertical_deg: float | None = None
    growth_side: str = ""
    axis_method: str = "growth-auto"
    fruit_prior: str = "no"
    fruit_prior_source: str = ""
    mask_output: str = ""
    prior_output: str = ""
    optimization_output: str = ""
    visualization_output: str = ""


class GrowthAutoEstimator:
    """Frozen Banana-Gpose + fruit-region prior + Growth-auto pipeline."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "0",
        confidence: float = 0.25,
        grounding_threshold: float = 0.18,
        sam_model: str = "sam2.1_t.pt",
        extend_ratio: float = 0.30,
        allow_large_sam_prior: bool = True,
    ):
        self.segmenter = BananaGposeSegmenter(weights, device=device, confidence=confidence)
        self.device = str(device)
        self.grounding_threshold = float(grounding_threshold)
        self.sam_model = str(sam_model)
        self.extend_ratio = float(extend_ratio)
        self.allow_large_sam_prior = bool(allow_large_sam_prior)

    def process(self, image_path: str | Path, output_dir: str | Path) -> PostureResult:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        image = cv2.imread(str(image_path))
        if image is None:
            return PostureResult(image=image_path.name, status="image_read_failed")

        prediction = self.segmenter.predict(image_path)
        if prediction is None:
            return PostureResult(image=image_path.name, status="no_detection")

        polygon = prediction.polygon
        boxes, scores = algorithm.detect_banana_regions(
            image, self.device, self.grounding_threshold
        )
        fruit_masks = algorithm.segment_fruit_boxes_with_sam(
            image_path, boxes, self.sam_model, self.device
        )
        fruit_prior = algorithm.fruit_prior_from_masks(
            polygon,
            fruit_masks,
            image.shape,
            allow_large_masks=self.allow_large_sam_prior,
        )

        axis = algorithm.fit_stalk_axis(
            polygon,
            image.shape,
            "growth-auto",
            self.extend_ratio,
            fruit_prior,
        )
        if axis is None:
            return PostureResult(
                image=image_path.name,
                status="axis_fit_failed",
                class_name=prediction.class_name,
                confidence=prediction.confidence,
                mask_area_px2=prediction.area_px2,
                fruit_prior="yes" if fruit_prior is not None else "no",
            )

        (
            p1,
            p2,
            angle_horizontal,
            angle_vertical,
            signed_angle,
            length_px,
            reconstructed_band,
            curve_points,
        ) = axis

        mask_path = output_dir / f"{stem}_mask.png"
        prior_path = output_dir / f"{stem}_fruit_prior.png"
        optimization_path = output_dir / f"{stem}_growth_auto.png"
        visualization_path = output_dir / f"{stem}_visualization.png"
        json_path = output_dir / f"{stem}_result.json"

        cv2.imwrite(str(mask_path), polygon_mask(image.shape, polygon))
        cv2.imwrite(str(prior_path), draw_prior(image, fruit_masks, fruit_prior, polygon))
        cv2.imwrite(
            str(optimization_path),
            draw_growth_auto_geometry(
                image, polygon, p1, p2, reconstructed_band, curve_points
            ),
        )
        visualization = algorithm.draw_result(
            image,
            polygon,
            p1,
            p2,
            angle_horizontal,
            angle_vertical,
            signed_angle,
            prediction.confidence,
            prediction.class_name,
            reconstructed_band,
            curve_points,
            None,
        )
        cv2.imwrite(str(visualization_path), visualization)

        result = PostureResult(
            image=image_path.name,
            status="ok",
            class_name=prediction.class_name,
            confidence=prediction.confidence,
            mask_area_px2=prediction.area_px2,
            axis_length_px=float(length_px),
            angle_from_horizontal_deg=float(angle_horizontal),
            angle_from_vertical_deg=float(angle_vertical),
            signed_angle_from_vertical_deg=float(signed_angle),
            growth_side="right" if signed_angle >= 0 else "left",
            fruit_prior="yes" if fruit_prior is not None else "no",
            fruit_prior_source=(fruit_prior or {}).get("source", ""),
            mask_output=str(mask_path),
            prior_output=str(prior_path),
            optimization_output=str(optimization_path),
            visualization_output=str(visualization_path),
        )
        write_json(json_path, asdict(result))
        return result


def write_summary_csv(path: str | Path, rows: list[PostureResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PostureResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
