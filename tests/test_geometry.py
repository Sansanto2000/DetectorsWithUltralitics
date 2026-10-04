import numpy as np
import pytest

from plate_detectors.detections import Detections
from plate_detectors.geometry import image_score, match, polygon_iou
from plate_detectors.labels import bbox_to_polygon


def box(x1, y1, x2, y2):
    return bbox_to_polygon((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def dets(*polygons, classes=None):
    polygons = np.stack(polygons) if polygons else np.empty((0, 4, 2))
    classes = np.zeros(len(polygons), dtype=int) if classes is None else np.array(classes)
    return Detections(classes, polygons)


def rotate(polygon, angle):
    c, s = np.cos(angle), np.sin(angle)
    return polygon @ np.array([[c, -s], [s, c]]).T


def test_iou_axis_aligned():
    assert polygon_iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert polygon_iou(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0


def test_iou_rotated_square_matches_analytic_value():
    square = box(-1, -1, 1, 1)
    octagon_area = 8 * (np.sqrt(2) - 1)
    expected = octagon_area / (8 - octagon_area)
    assert polygon_iou(square, rotate(square, np.pi / 4)) == pytest.approx(expected, abs=1e-6)


def test_iou_ignores_vertex_order():
    square = box(0, 0, 10, 10)
    assert polygon_iou(square, square[[0, 2, 1, 3]]) == pytest.approx(1.0)


def test_score_perfect_and_empty():
    assert image_score(dets(box(0, 0, 10, 10)), dets(box(0, 0, 10, 10))) == pytest.approx(1.0)
    assert image_score(dets(), dets()) == 1.0
    assert image_score(dets(box(0, 0, 10, 10)), dets()) == 0.0
    assert image_score(dets(), dets(box(0, 0, 10, 10))) == 0.0


def test_score_matching_is_one_to_one():
    # 2 GT idénticos, 1 predicción correcta + 1 falso positivo lejano.
    # La versión original emparejaba ambos GT con la misma predicción y daba 1.0.
    gt = dets(box(0, 0, 10, 10), box(0, 0, 10, 10))
    pred = dets(box(0, 0, 10, 10), box(100, 100, 110, 110))
    assert image_score(gt, pred) == pytest.approx(0.5)


def test_greedy_match_prefers_highest_iou():
    gt = dets(box(0, 0, 10, 10))
    pred = dets(box(2, 0, 12, 10), box(0, 0, 10, 10))
    assert match(gt, pred) == [(0, 1, pytest.approx(1.0))]


def test_class_aware_matching():
    gt = dets(box(0, 0, 10, 10), classes=[0])
    pred = dets(box(0, 0, 10, 10), classes=[1])
    assert image_score(gt, pred) == 0.0
    assert image_score(gt, pred, class_aware=False) == pytest.approx(1.0)
