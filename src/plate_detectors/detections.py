"""Representación común de ground truth y predicciones (bbox u OBB) como polígonos en píxeles."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plate_detectors.labels import bbox_to_polygon, find_label_path, read_labels


@dataclass
class Detections:
    classes: np.ndarray  # (N,) int
    polygons: np.ndarray  # (N, 4, 2) en píxeles
    confidences: np.ndarray | None = None  # (N,) sólo para predicciones

    def __len__(self) -> int:
        return len(self.classes)

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.empty(0, dtype=int), np.empty((0, 4, 2)))

    @classmethod
    def from_label_file(cls, label_path: Path | None, img_w: int, img_h: int) -> "Detections":
        if label_path is None:
            return cls.empty()
        classes, polygons = read_labels(label_path)
        return cls(classes, polygons * np.array([img_w, img_h]))

    @classmethod
    def ground_truth_for(cls, image_path: Path, img_w: int, img_h: int) -> "Detections":
        return cls.from_label_file(find_label_path(image_path), img_w, img_h)

    @classmethod
    def from_result(cls, result) -> "Detections":
        """Convierte un `ultralytics.engine.results.Results` de un modelo detect u obb.

        Los modelos OBB devuelven `result.obb` y dejan `result.boxes` en None;
        los de detección hacen lo contrario.
        """
        if result.obb is not None:
            data = result.obb
            polygons = data.xyxyxyxy.cpu().numpy().astype(np.float64)
        elif result.boxes is not None:
            data = result.boxes
            xyxy = data.xyxy.cpu().numpy().astype(np.float64)
            polygons = np.stack([_xyxy_to_polygon(b) for b in xyxy]) if len(xyxy) else np.empty((0, 4, 2))
        else:
            return cls.empty()
        return cls(
            data.cls.cpu().numpy().astype(int),
            polygons,
            data.conf.cpu().numpy(),
        )


def _xyxy_to_polygon(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return bbox_to_polygon((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
