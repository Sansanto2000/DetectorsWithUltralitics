"""IoU entre polígonos convexos y score por imagen. Válido para bbox y OBB por igual."""

import cv2
import numpy as np

from plate_detectors.detections import Detections


def _convex(polygon: np.ndarray) -> np.ndarray:
    # convexHull ordena los vértices: las etiquetas OBB no garantizan un orden cíclico.
    return cv2.convexHull(np.asarray(polygon, dtype=np.float32))


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU exacto entre dos cuadriláteros convexos (4, 2)."""
    ha, hb = _convex(a), _convex(b)
    inter, _ = cv2.intersectConvexConvex(ha, hb)
    union = cv2.contourArea(ha) + cv2.contourArea(hb) - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz (N, M) de IoU entre polígonos (N, 4, 2) y (M, 4, 2)."""
    return np.array([[polygon_iou(pa, pb) for pb in b] for pa in a]).reshape(len(a), len(b))


def match(gt: Detections, pred: Detections, class_aware: bool = True) -> list[tuple[int, int, float]]:
    """Emparejamiento greedy 1-a-1 por IoU descendente.

    Cada GT y cada predicción se usan a lo sumo una vez. Con `class_aware`
    sólo se emparejan detecciones de la misma clase.

    Returns:
        Lista de (índice_gt, índice_pred, iou) con iou > 0.
    """
    iou = iou_matrix(gt.polygons, pred.polygons)
    if class_aware:
        iou = np.where(gt.classes[:, None] == pred.classes[None, :], iou, 0.0)
    pairs = []
    used_gt, used_pred = set(), set()
    for flat in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(flat, iou.shape)
        if iou[i, j] <= 0:
            break
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        pairs.append((int(i), int(j), float(iou[i, j])))
    return pairs


def image_score(gt: Detections, pred: Detections, class_aware: bool = True) -> float:
    """Score por imagen en [0, 1]: suma de IoU emparejados / max(n_gt, n_pred).

    Dividir por el máximo penaliza tanto falsos negativos como falsos positivos.
    Una imagen sin GT ni predicciones vale 1.
    """
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0
    matched = sum(iou for _, _, iou in match(gt, pred, class_aware))
    return matched / max(n_gt, n_pred)
