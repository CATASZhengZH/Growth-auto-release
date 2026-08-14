"""Banana-Gpose loading and fruit-stalk instance selection."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


CUSTOM_MODULE = "ultralytics.nn.extra_modules.codex_blocks"


def _install_checkpoint_compatibility() -> None:
    """Register the frozen model classes referenced by the released checkpoint.

    The training environment added a YOLO26-compatible segmentation head and
    C2PSA followed by parameter-free SimAM to Ultralytics 8.3.226. Registering
    those original class names lets PyTorch deserialize the frozen checkpoint
    without modifying the installed Ultralytics package.
    """

    try:
        importlib.import_module(CUSTOM_MODULE)
    except ImportError:
        from ultralytics.nn.modules import C2PSA

        class CodexSimAM(nn.Module):
            def __init__(self, e_lambda: float = 1e-4):
                super().__init__()
                self.e_lambda = e_lambda
                self.act = nn.Sigmoid()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n = x.shape[2] * x.shape[3] - 1
                d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
                v = d.sum(dim=(2, 3), keepdim=True) / max(n, 1)
                return x * self.act(d / (4 * (v + self.e_lambda)) + 0.5)

        class CodexC2PSASimAM(C2PSA):
            def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
                super().__init__(c1, c2, n, e)
                self.attn = CodexSimAM()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.attn(super().forward(x))

        CodexSimAM.__module__ = CUSTOM_MODULE
        CodexC2PSASimAM.__module__ = CUSTOM_MODULE

        extra_name = "ultralytics.nn.extra_modules"
        extra = sys.modules.get(extra_name)
        if extra is None:
            extra = types.ModuleType(extra_name)
            extra.__path__ = []
            sys.modules[extra_name] = extra

        module = types.ModuleType(CUSTOM_MODULE)
        module.CodexSimAM = CodexSimAM
        module.CodexC2PSASimAM = CodexC2PSASimAM
        sys.modules[CUSTOM_MODULE] = module
        setattr(extra, "codex_blocks", module)

    head = importlib.import_module("ultralytics.nn.modules.head")
    if hasattr(head, "Segment26") and hasattr(head, "Proto26"):
        return

    from ultralytics.nn.modules.block import Proto
    from ultralytics.nn.modules.head import Segment

    class Segment26TrainOutput(dict):
        def __len__(self):
            return 3

        def __iter__(self):
            return iter(
                (
                    super().__getitem__("feats"),
                    super().__getitem__("mask_coefficient"),
                    super().__getitem__("proto"),
                )
            )

        def __getitem__(self, key):
            if key in (-1, 2):
                return super().__getitem__("proto")
            if key == 0:
                return super().__getitem__("feats")
            if key == 1:
                return (
                    super().__getitem__("feats"),
                    super().__getitem__("mask_coefficient"),
                    super().__getitem__("proto"),
                )
            return super().__getitem__(key)

    class Proto26(Proto):
        def __init__(self, ch=(), c_=256, c2=32, nc=80):
            c1 = ch[0] if isinstance(ch, (list, tuple)) else ch
            super().__init__(c1, c_, c2)

        def forward(self, x, return_semantic=True):
            return super().forward(x[0] if isinstance(x, (list, tuple)) else x)

    class Segment26(Segment):
        def __init__(self, nc=80, nm=32, npr=256, reg_max=16, end2end=False, ch=()):
            super().__init__(nc, nm, npr, ch)
            self.end2end = False
            self.proto = Proto26(ch, self.npr, self.nm, nc)

        def forward(self, x):
            out = super().forward(x)
            if not self.training:
                return out
            det_outputs, mask_coefficients, proto = out
            bs = proto.shape[0]
            reg = self.reg_max * 4
            boxes = torch.cat([xi[:, :reg].view(bs, reg, -1) for xi in det_outputs], 2)
            scores = torch.cat(
                [
                    xi[:, reg : reg + self.nc].view(bs, self.nc, -1)
                    for xi in det_outputs
                ],
                2,
            )
            return Segment26TrainOutput(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "feats": det_outputs,
                    "mask_coefficient": mask_coefficients,
                    "proto": proto,
                }
            )

    for cls in (Segment26TrainOutput, Proto26, Segment26):
        cls.__module__ = head.__name__
        setattr(head, cls.__name__, cls)


@dataclass(frozen=True)
class SegmentationPrediction:
    polygon: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    area_px2: float


class BananaGposeSegmenter:
    """Run Banana-Gpose and select the largest predicted stalk polygon."""

    def __init__(self, weights: str | Path, device: str = "0", confidence: float = 0.25):
        _install_checkpoint_compatibility()
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"Banana-Gpose weights not found: {self.weights}")
        self.device = str(device)
        self.confidence = float(confidence)
        self.model = YOLO(str(self.weights))

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)

    def predict(self, image_path: str | Path) -> SegmentationPrediction | None:
        result = self.model.predict(
            str(image_path), conf=self.confidence, device=self.device, verbose=False
        )[0]
        if result.masks is None or result.boxes is None or len(result.masks.xy) == 0:
            return None

        polygons = [np.asarray(poly, dtype=np.float32) for poly in result.masks.xy]
        areas = [self._polygon_area(poly) for poly in polygons]
        best_index = int(np.argmax(areas))
        class_id = int(result.boxes.cls[best_index].item())
        return SegmentationPrediction(
            polygon=polygons[best_index],
            confidence=float(result.boxes.conf[best_index].item()),
            class_id=class_id,
            class_name=result.names.get(class_id, "Fruit stalk"),
            area_px2=float(areas[best_index]),
        )
