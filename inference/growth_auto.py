import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_GROUNDING_DINO = None
_SAM_MODEL = None


def iter_images(source: Path):
    if source.is_file():
        return [source]
    return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def get_grounding_dino(device: str):
    global _GROUNDING_DINO
    if _GROUNDING_DINO is None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID)
        torch_device = "cuda" if str(device) != "cpu" and torch.cuda.is_available() else "cpu"
        model = model.to(torch_device).eval()
        _GROUNDING_DINO = processor, model, torch_device
    return _GROUNDING_DINO


def detect_banana_regions(image_bgr: np.ndarray, device: str, threshold: float):
    from PIL import Image

    processor, model, torch_device = get_grounding_dino(device)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    text = "banana fruit. banana finger. banana bunch."
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(torch_device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    boxes = results["boxes"].detach().cpu().numpy() if len(results["boxes"]) else np.empty((0, 4), dtype=np.float32)
    scores = results["scores"].detach().cpu().numpy() if len(results["scores"]) else np.empty((0,), dtype=np.float32)
    return boxes, scores


def get_sam_model(model_name: str):
    global _SAM_MODEL
    if _SAM_MODEL is None or _SAM_MODEL[0] != model_name:
        from ultralytics import SAM

        _SAM_MODEL = (model_name, SAM(model_name))
    return _SAM_MODEL[1]


def segment_fruit_boxes_with_sam(image_path: Path, boxes: np.ndarray, sam_model_name: str, device: str):
    if len(boxes) == 0:
        return []
    sam = get_sam_model(sam_model_name)
    results = sam.predict(str(image_path), bboxes=boxes.tolist(), device=device, verbose=False)
    if not results or results[0].masks is None:
        return []
    return [np.asarray(poly, dtype=np.float32) for poly in results[0].masks.xy]


def fruit_prior_from_boxes(points: np.ndarray, boxes: np.ndarray, scores: np.ndarray):
    if len(boxes) == 0:
        return None
    stalk_center = points.astype(np.float64).mean(axis=0)
    best = None
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(float, box)
        box_area = max((x2 - x1) * (y2 - y1), 1.0)
        box_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
        vector = stalk_center - box_center
        distance = float(np.linalg.norm(vector))
        if distance <= 1e-6:
            continue
        # Prefer large fruit/bunch regions that are close to the fruit stalk mask.
        proximity = 1.0 / (1.0 + distance / max(math.sqrt(box_area), 1.0))
        rank = float(score) * math.sqrt(box_area) * proximity
        if best is None or rank > best[0]:
            best = (rank, vector / distance, box, box_center)
    if best is None:
        return None
    return {"direction": best[1], "box": best[2], "center": best[3], "source": "grounding_box"}


def fruit_prior_from_masks(points: np.ndarray, masks, image_shape=None, allow_large_masks=False):
    if not masks:
        return None
    stalk_center = points.astype(np.float64).mean(axis=0)
    image_area = float(image_shape[0] * image_shape[1]) if image_shape is not None else None
    best = None
    for mask_points in masks:
        if len(mask_points) < 5:
            continue
        area = polygon_area(mask_points)
        if area <= 1.0:
            continue
        # Grounded-SAM can merge a whole bunch or nearby leaves when the text box is broad.
        # Usually that is noisy, but for strongly occluded stalks the large bunch mask can
        # still provide the useful growth-direction prior that a human would infer.
        large_mask = image_area is not None and area > image_area * 0.055
        if not allow_large_masks and large_mask:
            continue
        center = mask_points.astype(np.float64).mean(axis=0)
        vector = stalk_center - center
        distance = float(np.linalg.norm(vector))
        if distance <= 1e-6:
            continue
        if not allow_large_masks and distance > max(math.sqrt(area) * 2.2, 900.0):
            continue
        fit = fit_line_axis(mask_points.astype(np.float32))
        if fit is None:
            continue
        p1, p2 = fit[0], fit[1]
        fruit_axis = p2 - p1
        fruit_axis = fruit_axis / max(float(np.linalg.norm(fruit_axis)), 1e-9)
        bunch_to_stalk = vector / distance
        alignment = abs(float(np.dot(fruit_axis, bunch_to_stalk)))
        score = area * (0.65 + 0.35 * alignment) / (1.0 + distance / 1200.0)
        if best is None or score > best[0]:
            source = "large_sam_bunch" if large_mask else "sam_fruit"
            best = (score, bunch_to_stalk, center, source)
    if best is None:
        return None
    return {"direction": best[1], "center": best[2], "source": best[3]}


def fit_line_from_points(points: np.ndarray):
    if len(points) < 2:
        return None
    fit_points = points.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, cx, cy = cv2.fitLine(fit_points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([float(vx), float(vy)], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    center = np.array([float(cx), float(cy)], dtype=np.float64)
    projections = (points.astype(np.float64) - center) @ direction
    p1 = center + direction * float(projections.min())
    p2 = center + direction * float(projections.max())
    angle_from_horizontal, _, _, _ = axis_metrics(p1, p2)
    p1, p2 = orient_axis_endpoints(p1, p2, angle_from_horizontal)
    return p1, p2, *axis_metrics(p1, p2)


def fit_mask_centerline(mask_points: np.ndarray, image_shape):
    height, width = image_shape[:2]
    mask = make_polygon_mask(mask_points, image_shape)
    x, y, w, h = cv2.boundingRect(mask_points.astype(np.int32).reshape(-1, 1, 2))
    if w <= 1 or h <= 1:
        fit = fit_line_from_points(mask_points)
        return (*fit, None) if fit is not None else None

    centers = []
    bins = np.linspace(y, y + h - 1, 34)
    for y1, y2 in zip(bins[:-1], bins[1:]):
        y1_i = max(0, int(round(y1)))
        y2_i = min(height - 1, int(round(y2)))
        if y2_i <= y1_i:
            continue
        ys, xs = np.where(mask[y1_i : y2_i + 1, :] > 0)
        if len(xs) < max(12, w * 0.03):
            continue
        xs = xs.astype(np.float64)
        ys = ys.astype(np.float64) + y1_i
        centers.append([float(np.median(xs)), float(np.mean(ys))])

    if len(centers) < 4:
        fit = fit_line_from_points(mask_points)
        return (*fit, None) if fit is not None else None

    center_points = np.asarray(centers, dtype=np.float64)

    order = np.argsort(center_points[:, 1])
    center_points = center_points[order]
    ys = center_points[:, 1]
    xs = center_points[:, 0]
    y_min = float(ys.min())
    y_max = float(ys.max())
    if y_max - y_min < 1.0:
        fit = fit_line_from_points(center_points)
        return (*fit, None) if fit is not None else None

    degree = 2 if len(center_points) >= 6 else 1
    coeff = np.polyfit(ys, xs, deg=degree)
    curve_y = np.linspace(y_min, y_max, 120)
    curve_x = np.polyval(coeff, curve_y)
    curve_points = np.column_stack([curve_x, curve_y]).astype(np.float64)

    p1 = curve_points[0]
    p2 = curve_points[-1]
    angle_h, angle_v, signed_v, length = axis_metrics(p1, p2)
    if p1[1] > p2[1]:
        p1, p2 = p2, p1
        curve_points = curve_points[::-1].copy()
    return p1, p2, angle_h, angle_v, signed_v, length, curve_points


def fit_bunch_main_axis(stalk_points: np.ndarray, image_shape, boxes, scores, fruit_masks):
    height, width = image_shape[:2]
    image_area = float(height * width)
    stalk_center = stalk_points.astype(np.float64).mean(axis=0)

    candidates = []
    for idx, mask_points in enumerate(fruit_masks or []):
        if len(mask_points) < 8:
            continue
        area = polygon_area(mask_points)
        if area < image_area * 0.02:
            continue
        center = mask_points.astype(np.float64).mean(axis=0)
        x, y, w, h = cv2.boundingRect(mask_points.astype(np.int32).reshape(-1, 1, 2))
        vertical_extent = h / max(float(height), 1.0)
        horizontal_closeness = 1.0 / (1.0 + abs(center[0] - stalk_center[0]) / max(w, width * 0.12, 1.0))
        below_bonus = 1.2 if center[1] >= stalk_center[1] - height * 0.12 else 0.75
        score = area * (0.65 + vertical_extent) * horizontal_closeness * below_bonus
        candidates.append((score, mask_points.astype(np.float64), "sam_bunch"))

    if candidates:
        _, selected, source = max(candidates, key=lambda item: item[0])
        fit = fit_mask_centerline(selected, image_shape)
        if fit is None:
            return None
        p1, p2, angle_h, angle_v, signed_v, length, curve_points = fit
        # For rachis/main-axis visualization, keep it roughly top-to-bottom through the bunch.
        if p1[1] > p2[1]:
            p1, p2 = p2, p1
        return {
            "p1": p1,
            "p2": p2,
            "angle_from_horizontal_deg": angle_h,
            "angle_from_vertical_deg": angle_v,
            "signed_angle_from_vertical_deg": signed_v,
            "length_px": length,
            "curve_points": curve_points,
            "source": source,
        }

    box_candidates = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(float, box)
        area = max((x2 - x1) * (y2 - y1), 1.0)
        if area < image_area * 0.02:
            continue
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
        horizontal_closeness = 1.0 / (1.0 + abs(center[0] - stalk_center[0]) / max(x2 - x1, width * 0.12, 1.0))
        rank = float(score) * area * horizontal_closeness
        box_candidates.append((rank, np.array([x1, y1, x2, y2], dtype=np.float64)))

    if not box_candidates:
        return None

    _, box = max(box_candidates, key=lambda item: item[0])
    x1, y1, x2, y2 = box
    p1 = np.array([(x1 + x2) * 0.5, y1], dtype=np.float64)
    p2 = np.array([(x1 + x2) * 0.5, y2], dtype=np.float64)
    angle_h, angle_v, signed_v, length = axis_metrics(p1, p2)
    return {
        "p1": p1,
        "p2": p2,
        "angle_from_horizontal_deg": angle_h,
        "angle_from_vertical_deg": angle_v,
        "signed_angle_from_vertical_deg": signed_v,
        "length_px": length,
        "curve_points": None,
        "source": "grounding_box",
    }


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def reduce_line_angle(angle: float) -> float:
    angle = abs(angle) % 180.0
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


def axis_metrics(p1: np.ndarray, p2: np.ndarray):
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    angle_from_horizontal = reduce_line_angle(math.degrees(math.atan2(dy, dx)))

    top, bottom = (p1, p2) if p1[1] <= p2[1] else (p2, p1)
    signed_angle_from_vertical = math.degrees(
        math.atan2(float(bottom[0] - top[0]), float(bottom[1] - top[1]) if abs(float(bottom[1] - top[1])) > 1e-9 else 1e-9)
    )
    angle_from_vertical = reduce_line_angle(signed_angle_from_vertical)
    length_px = float(np.linalg.norm(p2 - p1))
    return angle_from_horizontal, angle_from_vertical, signed_angle_from_vertical, length_px


def orient_axis_endpoints(a: np.ndarray, b: np.ndarray, angle_from_horizontal: float):
    if angle_from_horizontal < 45.0:
        return (a, b) if a[0] <= b[0] else (b, a)
    return (a, b) if a[1] <= b[1] else (b, a)


def fit_line_axis(points: np.ndarray):
    if len(points) < 2:
        return None

    fit_points = points.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(fit_points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([float(vx), float(vy)], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return None
    direction /= norm

    center = np.array([float(x0), float(y0)], dtype=np.float64)
    projections = (points.astype(np.float64) - center) @ direction
    a = center + direction * projections.min()
    b = center + direction * projections.max()
    angle_from_horizontal, _, _, _ = axis_metrics(a, b)
    p1, p2 = orient_axis_endpoints(a, b, angle_from_horizontal)
    return p1, p2, *axis_metrics(p1, p2), None, None


def reconstructed_rect_from_axis(center: np.ndarray, direction: np.ndarray, long_len: float, short_len: float):
    normal = np.array([-direction[1], direction[0]])
    return np.array(
        [
            center - direction * long_len * 0.5 - normal * short_len * 0.5,
            center + direction * long_len * 0.5 - normal * short_len * 0.5,
            center + direction * long_len * 0.5 + normal * short_len * 0.5,
            center - direction * long_len * 0.5 + normal * short_len * 0.5,
        ],
        dtype=np.float64,
    )


def clip_line_to_image(center: np.ndarray, direction: np.ndarray, image_shape):
    height, width = image_shape[:2]
    candidates = []
    if abs(direction[0]) > 1e-9:
        for x in (0.0, float(width - 1)):
            t = (x - center[0]) / direction[0]
            y = center[1] + t * direction[1]
            if 0 <= y <= height - 1:
                candidates.append(t)
    if abs(direction[1]) > 1e-9:
        for y in (0.0, float(height - 1)):
            t = (y - center[1]) / direction[1]
            x = center[0] + t * direction[0]
            if 0 <= x <= width - 1:
                candidates.append(t)
    if len(candidates) < 2:
        return None
    return min(candidates), max(candidates)


def capsule_polygon(center: np.ndarray, direction: np.ndarray, length: float, width: float, samples=12):
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    normal = np.array([-direction[1], direction[0]])
    radius = width * 0.5
    a = center - direction * length * 0.5
    b = center + direction * length * 0.5
    theta = math.atan2(direction[1], direction[0])
    angles1 = np.linspace(theta + math.pi / 2.0, theta + math.pi * 3.0 / 2.0, samples)
    angles2 = np.linspace(theta - math.pi / 2.0, theta + math.pi / 2.0, samples)
    cap1 = np.column_stack([a[0] + np.cos(angles1) * radius, a[1] + np.sin(angles1) * radius])
    cap2 = np.column_stack([b[0] + np.cos(angles2) * radius, b[1] + np.sin(angles2) * radius])
    return np.vstack([cap1, cap2]).astype(np.float64)


def capsule_mask(shape, center: np.ndarray, direction: np.ndarray, length: float, width: float):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    poly = capsule_polygon(center, direction, length, width).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [poly], 255)
    return mask, poly.reshape(-1, 2).astype(np.float64)


def fit_rect_axis(points: np.ndarray, extend_ratio: float):
    if len(points) < 3:
        return None

    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    rect = cv2.minAreaRect(hull)
    center = np.array(rect[0], dtype=np.float64)
    width, height = rect[1]
    if width <= 0 or height <= 0:
        return None

    box = cv2.boxPoints(rect).astype(np.float64)
    edges = [(box[(idx + 1) % 4] - box[idx]) for idx in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    long_idx = int(np.argmax(lengths))
    direction = edges[long_idx] / max(lengths[long_idx], 1e-9)

    long_len = max(width, height) * (1.0 + extend_ratio)
    short_len = min(width, height) * 1.15
    a = center - direction * long_len * 0.5
    b = center + direction * long_len * 0.5
    angle_from_horizontal, _, _, _ = axis_metrics(a, b)
    p1, p2 = orient_axis_endpoints(a, b, angle_from_horizontal)

    rect_points = reconstructed_rect_from_axis(center, direction, long_len, short_len)
    return p1, p2, *axis_metrics(p1, p2), rect_points, None


def fit_capsule_axis(points: np.ndarray, image_shape, extend_ratio: float):
    if len(points) < 3:
        return None

    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32).reshape(-1, 1, 2)], 255)
    mask_area = max(float(cv2.countNonZero(mask)), 1.0)

    base = fit_centerline_axis(points, image_shape, 0.0) or fit_line_axis(points)
    if base is None:
        return None
    base_p1, base_p2 = base[0], base[1]
    base_angle = math.degrees(math.atan2(float(base_p2[1] - base_p1[1]), float(base_p2[0] - base_p1[0])))

    x, y, w, h = cv2.boundingRect(points.astype(np.int32).reshape(-1, 1, 2))
    distance = cv2.distanceTransform(mask[y : y + h, x : x + w], cv2.DIST_L2, 5)
    visible_width = max(float(distance.max()) * 2.0, min(w, h) * 0.45, 8.0)
    center0 = points.astype(np.float64).mean(axis=0)

    best = None
    angle_offsets = np.linspace(-22.0, 22.0, 23)
    width_scales = (0.85, 1.0, 1.18, 1.35)
    center_offsets = np.linspace(-visible_width * 0.8, visible_width * 0.8, 7)

    for offset in angle_offsets:
        angle = math.radians(base_angle + float(offset))
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        normal = np.array([-direction[1], direction[0]])
        projections = (points.astype(np.float64) - center0) @ direction
        visible_len = float(projections.max() - projections.min())
        length = max(visible_len * (1.0 + extend_ratio), visible_width * 2.5)
        for width_scale in width_scales:
            cap_width = visible_width * width_scale
            for center_offset in center_offsets:
                center = center0 + normal * center_offset
                cap, poly = capsule_mask(image_shape, center, direction, length, cap_width)
                cap_area = max(float(cv2.countNonZero(cap)), 1.0)
                intersection = float(cv2.countNonZero(cv2.bitwise_and(mask, cap)))
                coverage = intersection / mask_area
                precision = intersection / cap_area
                angle_penalty = abs(float(offset)) / 45.0
                width_penalty = abs(width_scale - 1.0) * 0.08
                score = coverage * 1.5 + precision * 0.55 - angle_penalty * 0.08 - width_penalty
                if best is None or score > best[0]:
                    best = (score, center, direction, length, cap_width, poly, coverage, precision)

    if best is None:
        return None

    _, center, direction, length, cap_width, poly, _, _ = best
    a = center - direction * length * 0.5
    b = center + direction * length * 0.5
    angle_from_horizontal, _, _, _ = axis_metrics(a, b)
    p1, p2 = orient_axis_endpoints(a, b, angle_from_horizontal)
    return p1, p2, *axis_metrics(p1, p2), poly, None


def medial_points(points: np.ndarray, image_shape, ridge_ratio=0.38):
    height, width = image_shape[:2]
    polygon = points.astype(np.int32)
    x, y, w, h = cv2.boundingRect(polygon.reshape(-1, 1, 2))
    pad = int(max(16, min(width, height) * 0.015))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)

    roi_polygon = polygon.copy()
    roi_polygon[:, 0] -= x0
    roi_polygon[:, 1] -= y0
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon.reshape(-1, 1, 2)], 255)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0:
        return None, None, None

    ridge_mask = distance >= max_distance * ridge_ratio
    ys, xs = np.where(ridge_mask)
    if len(xs) < 8:
        ys, xs = np.where(mask > 0)
    ridge_points = np.column_stack([xs + x0, ys + y0]).astype(np.float64)
    weights = distance[ys, xs].astype(np.float64)
    weights = np.maximum(weights, 1.0)
    return ridge_points, weights, max_distance


def fit_curve_axis(points: np.ndarray, image_shape, extend_ratio: float):
    ridge_points, weights, max_distance = medial_points(points, image_shape, ridge_ratio=0.34)
    if ridge_points is None or len(ridge_points) < 8:
        return fit_centerline_axis(points, image_shape, extend_ratio)

    height, width = image_shape[:2]
    x, y, w, h = cv2.boundingRect(points.astype(np.int32).reshape(-1, 1, 2))
    half_width = max(max_distance * 0.95, min(w, h) * 0.22, 6.0)

    horizontal = w >= h
    if horizontal:
        xs = ridge_points[:, 0]
        ys = ridge_points[:, 1]
        coeff = np.polyfit(xs, ys, deg=2, w=weights)
        visible_min = float(points[:, 0].min())
        visible_max = float(points[:, 0].max())
        visible_len = max(visible_max - visible_min, 1.0)
        x_start = max(0.0, visible_min - visible_len * extend_ratio * 0.35)
        x_end = min(float(width - 1), visible_max + visible_len * extend_ratio * 0.45)
        curve_x = np.linspace(x_start, x_end, 80)
        curve_y = np.polyval(coeff, curve_x)
        derivative = 2.0 * coeff[0] * curve_x + coeff[1]
        tangents = np.column_stack([np.ones_like(derivative), derivative])
        p1 = np.array([curve_x[0], curve_y[0]], dtype=np.float64)
        p2 = np.array([curve_x[-1], curve_y[-1]], dtype=np.float64)
        tangent_at_origin = tangents[-1]
    else:
        ys = ridge_points[:, 1]
        xs = ridge_points[:, 0]
        coeff = np.polyfit(ys, xs, deg=2, w=weights)
        visible_min = float(points[:, 1].min())
        visible_max = float(points[:, 1].max())
        visible_len = max(visible_max - visible_min, 1.0)
        y_start = max(0.0, visible_min - visible_len * extend_ratio * 0.35)
        y_end = min(float(height - 1), visible_max + visible_len * extend_ratio * 0.45)
        curve_y = np.linspace(y_start, y_end, 80)
        curve_x = np.polyval(coeff, curve_y)
        derivative = 2.0 * coeff[0] * curve_y + coeff[1]
        tangents = np.column_stack([derivative, np.ones_like(derivative)])
        p1 = np.array([curve_x[0], curve_y[0]], dtype=np.float64)
        p2 = np.array([curve_x[-1], curve_y[-1]], dtype=np.float64)
        tangent_at_origin = tangents[-1]

    curve = np.column_stack([curve_x, curve_y]).astype(np.float64)
    tangent_norms = np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    tangents = tangents / tangent_norms
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    upper = curve + normals * half_width
    lower = curve - normals * half_width
    band = np.vstack([upper, lower[::-1]])

    tangent_at_origin = tangent_at_origin / max(float(np.linalg.norm(tangent_at_origin)), 1e-9)
    axis_len = max(float(np.linalg.norm(p2 - p1)), 1.0)
    pseudo_p1 = p2 - tangent_at_origin * axis_len
    pseudo_p2 = p2
    angle_from_horizontal, angle_from_vertical, signed_angle_from_vertical, _ = axis_metrics(pseudo_p1, pseudo_p2)
    return p1, p2, angle_from_horizontal, angle_from_vertical, signed_angle_from_vertical, axis_len, band, curve


def make_polygon_mask(points: np.ndarray, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask


def polygon_mask_roi(points: np.ndarray, image_shape, extra_bbox=None, pad=8):
    height, width = image_shape[:2]
    polygon = points.astype(np.int32).reshape(-1, 1, 2)
    bx, by, bw, bh = cv2.boundingRect(polygon)
    x0, y0, x1, y1 = bx, by, bx + bw, by + bh
    if extra_bbox is not None:
        ex, ey, ew, eh = extra_bbox
        x0 = min(x0, int(ex))
        y0 = min(y0, int(ey))
        x1 = max(x1, int(ex + ew))
        y1 = max(y1, int(ey + eh))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None, (x0, y0, x1, y1)
    roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = polygon.copy()
    shifted[:, 0, 0] -= x0
    shifted[:, 0, 1] -= y0
    cv2.fillPoly(roi_mask, [shifted], 255)
    return roi_mask, (x0, y0, x1, y1)


def bezier_curve_points(start: np.ndarray, control: np.ndarray, end: np.ndarray, samples=96):
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return ((1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end).astype(np.float64)


def curve_swept_band(curve: np.ndarray, half_width: float):
    deltas = np.gradient(curve, axis=0)
    norms = np.maximum(np.linalg.norm(deltas, axis=1, keepdims=True), 1e-9)
    tangents = deltas / norms
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    upper = curve + normals * half_width
    lower = curve - normals * half_width
    return np.vstack([upper, lower[::-1]]).astype(np.float64)


def fit_handle_curve_axis(points: np.ndarray, image_shape, extend_ratio: float, fruit_prior):
    if fruit_prior is None:
        return fit_curve_axis(points, image_shape, extend_ratio)

    stalk_mask = make_polygon_mask(points, image_shape)
    mask_area = max(float(cv2.countNonZero(stalk_mask)), 1.0)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32).reshape(-1, 1, 2))
    distance = cv2.distanceTransform(stalk_mask[y : y + h, x : x + w], cv2.DIST_L2, 5)
    visible_width = max(float(distance.max()) * 2.0, min(w, h) * 0.42, 8.0)
    half_width = visible_width * 0.52

    d = np.asarray(fruit_prior["direction"], dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    n = np.array([-d[1], d[0]], dtype=np.float64)
    center = points.astype(np.float64).mean(axis=0)
    projections = (points.astype(np.float64) - center) @ d
    visible_min = float(projections.min())
    visible_max = float(projections.max())
    visible_len = max(visible_max - visible_min, 1.0)

    # Fruit prior gives bunch -> stalk. Extend more toward the hidden fruit side,
    # then choose the curved "umbrella handle" whose swept band best covers the mask.
    start_base = center + d * (visible_min - visible_len * extend_ratio * 0.75)
    end_base = center + d * (visible_max + visible_len * extend_ratio * 0.18)
    length = max(float(np.linalg.norm(end_base - start_base)), visible_width * 2.5)
    start_base = end_base - d * length

    best = None
    side_offsets = np.linspace(-visible_width * 1.35, visible_width * 1.35, 13)
    curve_offsets = np.linspace(-visible_len * 0.70, visible_len * 0.70, 25)
    end_offsets = np.linspace(-visible_width * 0.55, visible_width * 0.55, 7)
    width_scales = (0.90, 1.05, 1.22)

    for side_offset in side_offsets:
        start = start_base + n * side_offset
        for end_offset in end_offsets:
            end = end_base + n * end_offset
            chord = end - start
            chord_len = max(float(np.linalg.norm(chord)), 1.0)
            chord_dir = chord / chord_len
            chord_normal = np.array([-chord_dir[1], chord_dir[0]], dtype=np.float64)
            for curve_offset in curve_offsets:
                control = (start + end) * 0.5 + chord_normal * curve_offset
                curve = bezier_curve_points(start, control, end)
                for width_scale in width_scales:
                    band = curve_swept_band(curve, half_width * width_scale)
                    band_mask, (x0, y0, x1, y1) = polygon_mask_roi(
                        band,
                        image_shape,
                        extra_bbox=(x, y, w, h),
                        pad=max(8, int(visible_width)),
                    )
                    if band_mask is None:
                        continue
                    stalk_roi = stalk_mask[y0:y1, x0:x1]
                    band_area = max(float(cv2.countNonZero(band_mask)), 1.0)
                    intersection = float(cv2.countNonZero(cv2.bitwise_and(stalk_roi, band_mask)))
                    coverage = intersection / mask_area
                    precision = intersection / band_area
                    tangent_end = curve[-1] - curve[-3]
                    tangent_end = tangent_end / max(float(np.linalg.norm(tangent_end)), 1e-9)
                    prior_alignment = abs(float(np.dot(tangent_end, d)))
                    curve_penalty = abs(curve_offset) / max(visible_len, 1.0) * 0.05
                    score = coverage * 1.55 + precision * 0.70 + prior_alignment * 0.18 - curve_penalty
                    if best is None or score > best[0]:
                        best = (score, start, end, band, curve, tangent_end, coverage, precision)

    if best is None:
        return fit_curve_axis(points, image_shape, extend_ratio)

    _, start, end, band, curve, tangent_end, _, _ = best
    axis_len = max(float(np.linalg.norm(end - start)), 1.0)
    pseudo_p1 = end - tangent_end * axis_len
    pseudo_p2 = end
    angle_from_horizontal, angle_from_vertical, signed_angle_from_vertical, _ = axis_metrics(pseudo_p1, pseudo_p2)
    p1, p2 = orient_axis_endpoints(start, end, angle_from_horizontal)
    if np.linalg.norm(p2 - end) > np.linalg.norm(p1 - end):
        curve = curve[::-1].copy()
        band = curve_swept_band(curve, half_width)
        p1, p2 = p2, p1
    return p1, p2, angle_from_horizontal, angle_from_vertical, signed_angle_from_vertical, axis_len, band, curve


def fit_fruit_prior_axis(points: np.ndarray, image_shape, extend_ratio: float, fruit_prior):
    if fruit_prior is None:
        line = fit_line_axis(points)
        if line is None:
            return fit_curve_axis(points, image_shape, extend_ratio)
        direction = line[1] - line[0]
        direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
        fruit_prior = {"direction": direction, "center": points.astype(np.float64).mean(axis=0)}
    return fit_handle_curve_axis(points, image_shape, extend_ratio, fruit_prior)


def fit_growth_auto_axis(points: np.ndarray, image_shape, extend_ratio: float, fruit_prior=None):
    obb_axis = fit_obb_spine_axis(points, image_shape, extend_ratio)
    if fruit_prior is None:
        return obb_axis

    prior_axis = fit_handle_curve_axis(points, image_shape, extend_ratio, fruit_prior)
    if prior_axis is None:
        return obb_axis
    if obb_axis is None:
        return prior_axis

    prior_dir = prior_axis[1] - prior_axis[0]
    obb_dir = obb_axis[1] - obb_axis[0]
    prior_dir /= max(float(np.linalg.norm(prior_dir)), 1e-9)
    obb_dir /= max(float(np.linalg.norm(obb_dir)), 1e-9)
    angle_delta = math.degrees(math.acos(min(1.0, max(-1.0, abs(float(np.dot(prior_dir, obb_dir)))))))

    x, y, w, h = cv2.boundingRect(points.astype(np.int32).reshape(-1, 1, 2))
    aspect = max(w, h) / max(min(w, h), 1)
    prior_source = fruit_prior.get("source", "")

    # If the visible fruit stalk is already long and clean, the rotated-box spine is
    # the more stable measurement. Use the growth prior mainly when occlusion makes
    # the visible mask short/irregular or the prior comes from a large bunch mask.
    if prior_source == "large_sam_bunch" or aspect < 2.2 or angle_delta > 14.0:
        return prior_axis
    return obb_axis


def fit_centerline_axis(points: np.ndarray, image_shape, extend_ratio: float):
    ridge_points, _, max_distance = medial_points(points, image_shape, ridge_ratio=0.42)
    if ridge_points is None or len(ridge_points) < 2:
        return fit_line_axis(points)
    ridge_points = ridge_points.astype(np.float32)
    if len(ridge_points) < 2:
        return fit_line_axis(points)

    fit_points = ridge_points.reshape(-1, 1, 2)
    vx, vy, cx, cy = cv2.fitLine(fit_points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([float(vx), float(vy)], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)

    center = np.array([float(cx), float(cy)], dtype=np.float64)
    projections = (points.astype(np.float64) - center) @ direction
    visible_len = float(projections.max() - projections.min())
    long_len = visible_len * (1.0 + extend_ratio)
    short_len = max(max_distance * 2.0, 8.0)
    a = center + direction * (projections.min() - visible_len * extend_ratio * 0.5)
    b = center + direction * (projections.max() + visible_len * extend_ratio * 0.5)

    angle_from_horizontal, _, _, _ = axis_metrics(a, b)
    p1, p2 = orient_axis_endpoints(a, b, angle_from_horizontal)
    rect_points = reconstructed_rect_from_axis((a + b) * 0.5, direction, long_len, short_len)
    return p1, p2, *axis_metrics(p1, p2), rect_points, None


def fit_obb_spine_axis(points: np.ndarray, image_shape, extend_ratio: float):
    if len(points) < 3:
        return None

    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    rect = cv2.minAreaRect(hull)
    rect_center = np.array(rect[0], dtype=np.float64)
    width, height = rect[1]
    if width <= 0 or height <= 0:
        return fit_centerline_axis(points, image_shape, extend_ratio)

    box = cv2.boxPoints(rect).astype(np.float64)
    edges = [box[(idx + 1) % 4] - box[idx] for idx in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    long_idx = int(np.argmax(lengths))
    obb_direction = edges[long_idx] / max(lengths[long_idx], 1e-9)
    short_len = max(min(width, height) * 1.12, 8.0)

    ridge_points, weights, max_distance = medial_points(points, image_shape, ridge_ratio=0.38)
    if ridge_points is not None and len(ridge_points) >= 8:
        fit_points = ridge_points.astype(np.float32).reshape(-1, 1, 2)
        vx, vy, cx, cy = cv2.fitLine(fit_points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        spine_direction = np.array([float(vx), float(vy)], dtype=np.float64)
        spine_direction /= max(float(np.linalg.norm(spine_direction)), 1e-9)
        if abs(float(np.dot(spine_direction, obb_direction))) < math.cos(math.radians(32.0)):
            spine_direction = obb_direction.copy()
        elif float(np.dot(spine_direction, obb_direction)) < 0:
            spine_direction = -spine_direction
        center = np.average(ridge_points, axis=0, weights=weights)
        short_len = max(short_len, float(max_distance) * 2.1)
    else:
        spine_direction = obb_direction.copy()
        center = rect_center

    # Keep the rotated box as the visible fruit-stalk envelope, but compute the
    # "backbone" from points projected onto that local coordinate system.
    projections = (points.astype(np.float64) - center) @ spine_direction
    visible_min = float(projections.min())
    visible_max = float(projections.max())
    visible_len = max(visible_max - visible_min, max(width, height), 1.0)
    long_len = visible_len * (1.0 + extend_ratio)
    a = center + spine_direction * (visible_min - visible_len * extend_ratio * 0.5)
    b = center + spine_direction * (visible_max + visible_len * extend_ratio * 0.5)

    angle_from_horizontal, _, _, _ = axis_metrics(a, b)
    p1, p2 = orient_axis_endpoints(a, b, angle_from_horizontal)
    direction = p2 - p1
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    rect_points = reconstructed_rect_from_axis((p1 + p2) * 0.5, direction, long_len, short_len)

    curve_points = None
    if ridge_points is not None and len(ridge_points) >= 12:
        local_t = (ridge_points - center) @ direction
        local_n = (ridge_points - center) @ np.array([-direction[1], direction[0]])
        bins = np.linspace(local_t.min(), local_t.max(), 11)
        centers = []
        for left, right in zip(bins[:-1], bins[1:]):
            keep = (local_t >= left) & (local_t <= right)
            if keep.sum() < 2:
                continue
            w_keep = weights[keep]
            t_mean = float(np.average(local_t[keep], weights=w_keep))
            n_mean = float(np.average(local_n[keep], weights=w_keep))
            centers.append(center + direction * t_mean + np.array([-direction[1], direction[0]]) * n_mean)
        if len(centers) >= 2:
            curve_points = np.asarray(centers, dtype=np.float64)

    return p1, p2, *axis_metrics(p1, p2), rect_points, curve_points


def fit_stalk_axis(points: np.ndarray, image_shape, method: str, extend_ratio: float, fruit_prior=None):
    if method == "growth-auto":
        return fit_growth_auto_axis(points, image_shape, extend_ratio, fruit_prior)
    if method == "fruit-prior":
        return fit_fruit_prior_axis(points, image_shape, extend_ratio, fruit_prior)
    if method == "obb-spine":
        return fit_obb_spine_axis(points, image_shape, extend_ratio)
    if method == "curve":
        return fit_curve_axis(points, image_shape, extend_ratio)
    if method == "capsule":
        return fit_capsule_axis(points, image_shape, extend_ratio)
    if method == "centerline":
        return fit_centerline_axis(points, image_shape, extend_ratio)
    if method == "line":
        return fit_line_axis(points)
    return fit_rect_axis(points, extend_ratio)


def draw_dashed_line(image, p1, p2, color, thickness=2, dash_length=24, gap_length=14):
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    vector = p2 - p1
    length = float(np.linalg.norm(vector))
    if length <= 0:
        return
    direction = vector / length
    distance = 0.0
    while distance < length:
        start = p1 + direction * distance
        end = p1 + direction * min(distance + dash_length, length)
        cv2.line(
            image,
            tuple(np.round(start).astype(int)),
            tuple(np.round(end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        distance += dash_length + gap_length


def draw_angle_arc(image, center, radius, signed_angle, label_angle, color=(255, 60, 0)):
    start_deg = 0.0
    end_deg = -90.0 - signed_angle
    start = min(start_deg, end_deg)
    end = max(start_deg, end_deg)
    cv2.ellipse(
        image,
        tuple(np.round(center).astype(int)),
        (radius, radius),
        0,
        start,
        end,
        color,
        4,
        cv2.LINE_AA,
    )

    mid_deg = math.radians((start + end) * 0.5)
    text_point = np.array(
        [
            center[0] + math.cos(mid_deg) * (radius + 42),
            center[1] + math.sin(mid_deg) * (radius + 42),
        ]
    )
    label = f"angle = {label_angle:.1f} deg"
    cv2.putText(
        image,
        label,
        tuple(np.round(text_point).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_local_coordinate(image, origin, ref_len, color=(0, 0, 255), thickness=5):
    origin_i = tuple(np.round(origin).astype(int))
    y_top = origin - np.array([0.0, ref_len])
    x_left = origin - np.array([ref_len * 0.52, 0.0])
    x_right = origin + np.array([ref_len * 0.52, 0.0])

    cv2.line(image, tuple(np.round(y_top).astype(int)), origin_i, color, thickness, cv2.LINE_AA)
    cv2.line(
        image,
        tuple(np.round(x_left).astype(int)),
        tuple(np.round(x_right).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.circle(image, origin_i, 7, color, -1, cv2.LINE_AA)

    cv2.putText(
        image,
        "Y",
        tuple(np.round(y_top - np.array([26.0, 10.0])).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "X",
        tuple(np.round(x_right + np.array([8.0, 10.0])).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_text_panel(image, lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.78
    thickness = 2
    padding = 16
    line_height = 34
    text_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    panel_w = min(text_width + padding * 2, image.shape[1] - 40)
    panel_h = padding * 2 + line_height * len(lines)
    x1, y1 = 20, 20
    x2, y2 = x1 + panel_w, y1 + panel_h
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x1 + padding, y1 + padding + 24 + idx * line_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_grounded_sam_debug(image, boxes, scores, fruit_masks, fruit_prior, output_path: Path, stalk_polygon=None):
    debug = image.copy()
    palette = [
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 160, 255),
        (160, 255, 0),
        (255, 120, 0),
    ]

    if stalk_polygon is not None and len(stalk_polygon) >= 3:
        stalk_contour = stalk_polygon.astype(np.int32).reshape(-1, 1, 2)
        overlay = debug.copy()
        cv2.fillPoly(overlay, [stalk_contour], (0, 180, 80))
        debug = cv2.addWeighted(overlay, 0.30, debug, 0.70, 0)
        cv2.polylines(debug, [stalk_contour], True, (0, 255, 80), 3, cv2.LINE_AA)
        sx, sy, _, _ = cv2.boundingRect(stalk_contour)
        cv2.putText(debug, "fruit stalk mask", (sx, max(28, sy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 80), 2, cv2.LINE_AA)

    for idx, mask_points in enumerate(fruit_masks or []):
        if len(mask_points) < 3:
            continue
        color = palette[idx % len(palette)]
        contour = mask_points.astype(np.int32).reshape(-1, 1, 2)
        overlay = debug.copy()
        cv2.fillPoly(overlay, [contour], color)
        debug = cv2.addWeighted(overlay, 0.22, debug, 0.78, 0)
        cv2.polylines(debug, [contour], True, color, 2, cv2.LINE_AA)

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = np.round(box).astype(int)
        color = palette[idx % len(palette)]
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        score = scores[idx] if idx < len(scores) else 0.0
        cv2.putText(debug, f"banana {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if fruit_prior is not None and "center" in fruit_prior:
        center = np.asarray(fruit_prior["center"], dtype=np.float64)
        direction = np.asarray(fruit_prior["direction"], dtype=np.float64)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
        end = center + direction * 260.0
        cv2.circle(debug, tuple(np.round(center).astype(int)), 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            debug,
            tuple(np.round(center).astype(int)),
            tuple(np.round(end).astype(int)),
            (255, 255, 255),
            4,
            cv2.LINE_AA,
            tipLength=0.16,
        )
        cv2.putText(debug, "fruit prior", tuple(np.round(end + np.array([8.0, 0.0])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)


def connected_full_axis_curve(whole_axis, stalk_p1, stalk_p2, stalk_curve_points=None):
    if whole_axis is None:
        return None

    main_curve = whole_axis.get("curve_points")
    if main_curve is None or len(main_curve) < 2:
        main_curve = np.asarray([whole_axis["p1"], whole_axis["p2"]], dtype=np.float64)
    else:
        main_curve = np.asarray(main_curve, dtype=np.float64)

    stalk_p1 = np.asarray(stalk_p1, dtype=np.float64)
    stalk_p2 = np.asarray(stalk_p2, dtype=np.float64)

    # Fit the complete biological fruit stalk as one smooth curve. The visible upper
    # stalk supplies the cutting-site tangent, while the bunch centerline supplies
    # the lower rachis direction. Avoid stitching two independent polylines.
    start = main_curve[-1].copy()
    end = stalk_p2.copy()

    if stalk_curve_points is not None and len(stalk_curve_points) >= 2:
        local_curve = np.asarray(stalk_curve_points, dtype=np.float64)
        if np.linalg.norm(local_curve[-1] - stalk_p2) <= np.linalg.norm(local_curve[0] - stalk_p2):
            end = local_curve[-1].copy()
            local_dir = local_curve[-1] - local_curve[-2]
        else:
            end = local_curve[0].copy()
            local_dir = local_curve[0] - local_curve[1]
    else:
        local_dir = stalk_p2 - stalk_p1
    local_dir = local_dir / max(float(np.linalg.norm(local_dir)), 1e-9)

    if len(main_curve) >= 2:
        main_dir = main_curve[-2] - main_curve[-1]
    else:
        main_dir = end - start
    main_dir = main_dir / max(float(np.linalg.norm(main_dir)), 1e-9)

    curve_len = max(float(np.linalg.norm(end - start)), 1.0)
    tangent_scale = curve_len * 0.52
    control1 = start + main_dir * tangent_scale
    control2 = end - local_dir * tangent_scale

    t = np.linspace(0.0, 1.0, 160)[:, None]
    curve = (
        (1.0 - t) ** 3 * start
        + 3.0 * (1.0 - t) ** 2 * t * control1
        + 3.0 * (1.0 - t) * t**2 * control2
        + t**3 * end
    )
    return curve.astype(np.float64)


def operation_segment_from_full_curve(full_curve: np.ndarray, fraction=0.32, min_points=18):
    if full_curve is None or len(full_curve) < 2:
        return None
    count = max(min_points, int(len(full_curve) * fraction))
    count = min(count, len(full_curve))
    return np.asarray(full_curve[-count:], dtype=np.float64)


def draw_result(
    image,
    polygon,
    p1,
    p2,
    angle_horizontal,
    angle_vertical,
    signed_angle,
    confidence,
    label,
    reconstructed_rect=None,
    curve_points=None,
    whole_axis=None,
):
    overlay = image.copy()
    contour = polygon.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(overlay, [contour], (0, 180, 80))
    image = cv2.addWeighted(overlay, 0.28, image, 0.72, 0)
    cv2.polylines(image, [contour], True, (0, 210, 80), 2, cv2.LINE_AA)

    if reconstructed_rect is not None:
        rect_overlay = image.copy()
        rect_contour = reconstructed_rect.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(rect_overlay, [rect_contour], (255, 0, 0))
        image = cv2.addWeighted(rect_overlay, 0.10, image, 0.90, 0)
        cv2.polylines(image, [rect_contour], True, (255, 0, 0), 2, cv2.LINE_AA)

    if curve_points is not None and len(curve_points) >= 2:
        axis_vector = curve_points[-1] - curve_points[-2]
    else:
        axis_vector = p2 - p1
    axis_length = max(float(np.linalg.norm(p2 - p1)), 1.0)
    axis_unit = axis_vector / max(float(np.linalg.norm(axis_vector)), 1.0)
    center = p2
    ref_len = min(axis_length * 0.55, image.shape[0] * 0.28)
    axis_ref = center - axis_unit * ref_len
    arc_radius = int(max(82, min(210, axis_length * 0.24)))

    measure_overlay = image.copy()
    x_ref = center + np.array([ref_len * 0.52, 0.0])
    wedge = np.array([center, x_ref, axis_ref], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(measure_overlay, [wedge], (255, 60, 0))
    image = cv2.addWeighted(measure_overlay, 0.08, image, 0.92, 0)

    draw_local_coordinate(image, center, ref_len)
    cv2.line(
        image,
        tuple(np.round(center).astype(int)),
        tuple(np.round(axis_ref).astype(int)),
        (255, 0, 0),
        3,
        cv2.LINE_AA,
    )
    draw_angle_arc(image, center, arc_radius, signed_angle, angle_horizontal)

    start = tuple(np.round(p1).astype(int))
    end = tuple(np.round(p2).astype(int))
    if curve_points is not None and len(curve_points) >= 2:
        cv2.polylines(image, [curve_points.astype(np.int32).reshape(-1, 1, 2)], False, (255, 0, 0), 4, cv2.LINE_AA)
        cv2.arrowedLine(
            image,
            tuple(np.round(curve_points[-2]).astype(int)),
            tuple(np.round(curve_points[-1]).astype(int)),
            (255, 0, 0),
            4,
            cv2.LINE_AA,
            tipLength=0.25,
        )
    else:
        cv2.arrowedLine(image, end, start, (255, 0, 0), 4, cv2.LINE_AA, tipLength=0.08)
    cv2.circle(image, start, 8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(image, end, 7, (0, 0, 255), -1, cv2.LINE_AA)

    if whole_axis is not None:
        main_p1 = np.asarray(whole_axis["p1"], dtype=np.float64)
        main_p2 = np.asarray(whole_axis["p2"], dtype=np.float64)
        main_curve = whole_axis.get("curve_points")
        full_curve = connected_full_axis_curve(whole_axis, p1, p2, curve_points)
        full_color = (0, 255, 255)
        yellow = full_color
        if full_curve is not None and len(full_curve) >= 2:
            cv2.polylines(image, [np.asarray(full_curve, dtype=np.int32).reshape(-1, 1, 2)], False, full_color, 7, cv2.LINE_AA)
        elif main_curve is not None and len(main_curve) >= 2:
            cv2.polylines(image, [np.asarray(main_curve, dtype=np.int32).reshape(-1, 1, 2)], False, yellow, 5, cv2.LINE_AA)
        else:
            cv2.line(
                image,
                tuple(np.round(main_p1).astype(int)),
                tuple(np.round(main_p2).astype(int)),
                full_color,
                5,
                cv2.LINE_AA,
            )
        if full_curve is not None and len(full_curve) >= 2:
            cv2.circle(image, tuple(np.round(full_curve[0]).astype(int)), 9, full_color, 2, cv2.LINE_AA)
            cv2.circle(image, tuple(np.round(full_curve[-1]).astype(int)), 9, full_color, 2, cv2.LINE_AA)
            label_point = np.asarray(full_curve[len(full_curve) // 2], dtype=np.float64) + np.array([16.0, 0.0])
        else:
            cv2.circle(image, tuple(np.round(main_p1).astype(int)), 9, full_color, 2, cv2.LINE_AA)
            cv2.circle(image, tuple(np.round(main_p2).astype(int)), 9, full_color, 2, cv2.LINE_AA)
            label_point = main_p1 * 0.62 + main_p2 * 0.38 + np.array([16.0, 0.0])
        cv2.putText(
            image,
            "full stalk curve",
            tuple(np.round(label_point).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            full_color,
            3,
            cv2.LINE_AA,
        )
        if curve_points is not None and len(curve_points) >= 2:
            cv2.polylines(image, [curve_points.astype(np.int32).reshape(-1, 1, 2)], False, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.arrowedLine(
                image,
                tuple(np.round(curve_points[-2]).astype(int)),
                tuple(np.round(curve_points[-1]).astype(int)),
                (255, 0, 0),
                4,
                cv2.LINE_AA,
                tipLength=0.25,
            )

    normal = np.array([-axis_unit[1], axis_unit[0]])
    top_label = p1 + normal * 34
    bottom_label = p2 + normal * 34
    cv2.putText(image, "top", tuple(np.round(top_label).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, "tip", tuple(np.round(bottom_label).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3, cv2.LINE_AA)

    side = "right" if signed_angle >= 0 else "left"
    panel_lines = [
        f"{label}  conf {confidence:.2f}",
        f"X-axis angle: {angle_horizontal:.1f} deg",
        f"Y-axis angle: {signed_angle:+.1f} deg ({side})",
        "blue band: reconstructed curved stalk",
    ]
    if whole_axis is not None:
        panel_lines.append("yellow curve: complete stalk")
    panel_lines.extend(
        [
            "red axes: local coordinates",
            "blue arrow: fitted stalk axis",
        ]
    )
    draw_text_panel(
        image,
        panel_lines,
    )
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate fruit stalk growth angle from YOLO segmentation masks.")
    parser.add_argument("--model", default="weights/Banana-Gpose-best.pt")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--output", default="demo/results_legacy")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--class-name", default="Fruit stalk")
    parser.add_argument(
        "--axis-method",
        choices=["growth-auto", "obb-spine", "fruit-prior", "curve", "capsule", "centerline", "rect", "line"],
        default="growth-auto",
        help="Use rotated-box spine, curved stalk prior, capsule template fitting, mask medial centerline, reconstructed rectangle, or direct mask boundary line fitting.",
    )
    parser.add_argument(
        "--use-grounding-dino",
        action="store_true",
        help="Use GroundingDINO open-vocabulary banana detections as fruit direction prior.",
    )
    parser.add_argument("--grounding-threshold", type=float, default=0.18)
    parser.add_argument(
        "--use-sam-fruit-masks",
        action="store_true",
        help="Segment GroundingDINO banana boxes with SAM/SAM2 before estimating fruit direction.",
    )
    parser.add_argument(
        "--debug-sam",
        action="store_true",
        help="Save a GroundingDINO/SAM fruit-mask debug image next to the angle result.",
    )
    parser.add_argument(
        "--allow-large-sam-prior",
        action="store_true",
        help="Allow large Grounded-SAM bunch masks to guide the occluded stalk growth direction.",
    )
    parser.add_argument(
        "--fit-whole-axis",
        action="store_true",
        help="Also fit and draw the banana bunch rachis/main axis using Grounded-SAM/box regions.",
    )
    parser.add_argument("--sam-model", default="sam2.1_t.pt")
    parser.add_argument(
        "--extend-ratio",
        type=float,
        default=0.30,
        help="How much to extend the reconstructed stalk long axis beyond the visible mask.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    images = iter_images(source)
    if not images:
        raise SystemExit(f"No images found: {source}")

    model = YOLO(args.model)
    rows = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            rows.append({"image": str(image_path), "status": "image_read_failed"})
            continue

        result = model.predict(str(image_path), conf=args.conf, device=args.device, verbose=False)[0]
        if result.masks is None or result.boxes is None or len(result.masks.xy) == 0:
            rows.append({"image": str(image_path), "status": "no_detection"})
            continue

        polygons = [np.asarray(poly, dtype=np.float32) for poly in result.masks.xy]
        areas = [polygon_area(poly) for poly in polygons]
        best_index = int(np.argmax(areas))
        polygon = polygons[best_index]
        fruit_prior = None
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        fruit_masks = []
        if args.axis_method in ("growth-auto", "fruit-prior") or args.use_grounding_dino or args.fit_whole_axis:
            boxes, scores = detect_banana_regions(image, args.device, args.grounding_threshold)
            if args.use_sam_fruit_masks or args.fit_whole_axis:
                fruit_masks = segment_fruit_boxes_with_sam(image_path, boxes, args.sam_model, args.device)
                fruit_prior = fruit_prior_from_masks(polygon, fruit_masks, image.shape, args.allow_large_sam_prior)
            if fruit_prior is None and not args.use_sam_fruit_masks:
                fruit_prior = fruit_prior_from_boxes(polygon, boxes, scores)
            if args.debug_sam:
                debug_image = output / f"{image_path.stem}_grounded_sam_debug.jpg"
                draw_grounded_sam_debug(image, boxes, scores, fruit_masks, fruit_prior, debug_image, polygon)
        axis = fit_stalk_axis(polygon, image.shape, args.axis_method, args.extend_ratio, fruit_prior)
        if axis is None:
            rows.append({"image": str(image_path), "status": "axis_fit_failed"})
            continue

        p1, p2, angle_horizontal, angle_vertical, signed_angle, length_px, reconstructed_rect, curve_points = axis
        whole_axis = fit_bunch_main_axis(polygon, image.shape, boxes, scores, fruit_masks) if args.fit_whole_axis else None
        if whole_axis is not None:
            full_curve = connected_full_axis_curve(whole_axis, p1, p2, curve_points)
            operation_curve = operation_segment_from_full_curve(full_curve)
            if operation_curve is not None and len(operation_curve) >= 2:
                curve_points = operation_curve
                p1 = operation_curve[0]
                p2 = operation_curve[-1]
                tangent = operation_curve[-1] - operation_curve[max(0, len(operation_curve) - 4)]
                tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
                pseudo_p1 = p2 - tangent * max(float(np.linalg.norm(p2 - p1)), 1.0)
                angle_horizontal, angle_vertical, signed_angle, length_px = axis_metrics(pseudo_p1, p2)
        confidence = float(result.boxes.conf[best_index].item())
        class_id = int(result.boxes.cls[best_index].item())
        label = result.names.get(class_id, args.class_name)

        visualized = draw_result(
            image,
            polygon,
            p1,
            p2,
            angle_horizontal,
            angle_vertical,
            signed_angle,
            confidence,
            label,
            reconstructed_rect,
            curve_points,
            whole_axis,
        )
        out_image = output / f"{image_path.stem}_angle.jpg"
        cv2.imwrite(str(out_image), visualized)

        rows.append(
            {
                "image": str(image_path),
                "status": "ok",
                "class": label,
                "confidence": f"{confidence:.6f}",
                "mask_area_px2": f"{areas[best_index]:.2f}",
                "axis_length_px": f"{length_px:.2f}",
                "angle_from_horizontal_deg": f"{angle_horizontal:.3f}",
                "angle_from_vertical_deg": f"{angle_vertical:.3f}",
                "signed_angle_from_vertical_deg": f"{signed_angle:.3f}",
                "growth_angle_y_signed_deg": f"{signed_angle:.3f}",
                "growth_angle_y_abs_deg": f"{abs(signed_angle):.3f}",
                "growth_side": "right" if signed_angle >= 0 else "left",
                "axis_method": args.axis_method,
                "fruit_prior": "yes" if fruit_prior is not None else "no",
                "whole_axis": "yes" if whole_axis is not None else "no",
                "whole_axis_source": whole_axis["source"] if whole_axis is not None else "",
                "whole_axis_angle_from_horizontal_deg": f"{whole_axis['angle_from_horizontal_deg']:.3f}" if whole_axis is not None else "",
                "whole_axis_angle_from_vertical_deg": f"{whole_axis['angle_from_vertical_deg']:.3f}" if whole_axis is not None else "",
                "x1": f"{p1[0]:.2f}",
                "y1": f"{p1[1]:.2f}",
                "x2": f"{p2[0]:.2f}",
                "y2": f"{p2[1]:.2f}",
                "output_image": str(out_image),
            }
        )

    fieldnames = [
        "image",
        "status",
        "class",
        "confidence",
        "mask_area_px2",
        "axis_length_px",
        "angle_from_horizontal_deg",
        "angle_from_vertical_deg",
        "signed_angle_from_vertical_deg",
        "growth_angle_y_signed_deg",
        "growth_angle_y_abs_deg",
        "growth_side",
        "axis_method",
        "fruit_prior",
        "whole_axis",
        "whole_axis_source",
        "whole_axis_angle_from_horizontal_deg",
        "whole_axis_angle_from_vertical_deg",
        "x1",
        "y1",
        "x2",
        "y2",
        "output_image",
    ]
    csv_path = output / "angles.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        if row["status"] == "ok":
            print(
                f"{Path(row['image']).name}: "
                f"{row['growth_angle_y_signed_deg']} deg from vertical Y, "
                f"{row['angle_from_horizontal_deg']} deg from horizontal, "
                f"conf={row['confidence']}"
            )
        else:
            print(f"{Path(row['image']).name}: {row['status']}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
