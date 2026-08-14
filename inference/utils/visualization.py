"""Paper-consistent output overlays for the public demo."""

from __future__ import annotations

import cv2
import numpy as np


def polygon_mask(shape, polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    output = image.copy()
    layer = np.zeros_like(output)
    layer[:] = color
    keep = mask > 0
    output[keep] = cv2.addWeighted(output, 1.0 - alpha, layer, alpha, 0)[keep]
    return output


def draw_prior(
    image: np.ndarray,
    fruit_masks: list[np.ndarray],
    fruit_prior: dict | None,
    stalk_polygon: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    fruit_binary = np.zeros(image.shape[:2], dtype=np.uint8)
    for polygon in fruit_masks:
        if len(polygon) >= 3:
            cv2.fillPoly(fruit_binary, [np.round(polygon).astype(np.int32)], 255)
    output = overlay_mask(output, fruit_binary, (0, 190, 255), 0.28)

    stalk_center = stalk_polygon.astype(np.float64).mean(axis=0)
    if fruit_prior is not None and fruit_prior.get("center") is not None:
        fruit_center = np.asarray(fruit_prior["center"], dtype=np.float64)
        cv2.circle(output, tuple(np.round(fruit_center).astype(int)), 12, (0, 215, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            output,
            tuple(np.round(fruit_center).astype(int)),
            tuple(np.round(stalk_center).astype(int)),
            (0, 140, 255),
            6,
            cv2.LINE_AA,
            tipLength=0.12,
        )
    cv2.putText(output, "fruit-region prior", (28, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3, cv2.LINE_AA)
    return output


def draw_growth_auto_geometry(
    image: np.ndarray,
    polygon: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    reconstructed_band: np.ndarray | None,
    curve_points: np.ndarray | None,
) -> np.ndarray:
    output = overlay_mask(image, polygon_mask(image.shape, polygon), (255, 220, 80), 0.22)
    if reconstructed_band is not None and len(reconstructed_band) >= 3:
        cv2.polylines(output, [np.round(reconstructed_band).astype(np.int32)], True, (40, 180, 40), 5, cv2.LINE_AA)
    if curve_points is not None and len(curve_points) >= 2:
        cv2.polylines(output, [np.round(curve_points).astype(np.int32)], False, (40, 180, 40), 7, cv2.LINE_AA)
    cv2.arrowedLine(
        output,
        tuple(np.round(p1).astype(int)),
        tuple(np.round(p2).astype(int)),
        (180, 45, 155),
        8,
        cv2.LINE_AA,
        tipLength=0.08,
    )
    cv2.putText(output, "Growth-auto optimization", (28, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3, cv2.LINE_AA)
    return output
