import random

import cv2
import numpy as np
import pytest

from plate_detectors.dataset.move_contents import move_contents
from plate_detectors.dataset.rotate90 import rotate_coords_90cw, rotate_dataset
from plate_detectors.dataset.split import copy_split, split_files
from plate_detectors.labels import read_labels


def test_rotate_bbox_formula():
    assert rotate_coords_90cw([0.2, 0.3, 0.1, 0.4]) == pytest.approx([0.7, 0.2, 0.4, 0.1])


@pytest.mark.parametrize("coords", [[0.2, 0.3, 0.1, 0.4], [0.1, 0.2, 0.4, 0.25, 0.35, 0.6, 0.05, 0.55]])
def test_four_rotations_are_identity(coords):
    rotated = coords
    for _ in range(4):
        rotated = rotate_coords_90cw(rotated)
    assert rotated == pytest.approx(coords)


def fill_mask(shape, polygon_px):
    """Rasteriza con precisión subpíxel. Los vértices de fillPoly son centros de píxel,
    por eso se resta 0.5: así un punto continuo x cae en el píxel floor(x)."""
    mask = np.zeros(shape, np.uint8)
    pts = np.round((np.asarray(polygon_px) - 0.5) * 16).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255, shift=4)
    return mask


def make_sample(root, split, name, polygon_norm, size=(100, 200)):
    """Imagen negra con el polígono relleno en blanco, su etiqueta OBB y una bbox extra."""
    h, w = size
    img = cv2.cvtColor(fill_mask((h, w), np.asarray(polygon_norm) * [w, h]), cv2.COLOR_GRAY2BGR)
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "images" / split / f"{name}.png"), img)
    coords = " ".join(str(v) for v in np.asarray(polygon_norm).ravel())
    (root / "labels" / split / f"{name}.txt").write_text(f"1 {coords}\n0 0.5 0.5 0.1 0.1\n")


def test_rotate_dataset_keeps_obb_labels_aligned_with_pixels(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    h, w = 100, 200
    tilted = cv2.boxPoints(((w / 2, h / 2), (90, 30), 20)) / [w, h]  # rectángulo rotado 20°
    make_sample(src, "train", "a", tilted)
    make_sample(src, "val", "b", tilted)

    rotate_dataset(src, dst)

    for split, name in (("train", "a"), ("val", "b")):
        rotated = cv2.imread(str(dst / "images" / split / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127
        assert rotated.shape == (w, h)  # alto y ancho intercambiados
        classes, polygons = read_labels(dst / "labels" / split / f"{name}.txt")
        assert classes.tolist() == [1, 0]  # se conservan las líneas OBB y bbox
        label_mask = fill_mask(rotated.shape, polygons[0] * [h, w]) > 127
        pixel_iou = (label_mask & rotated).sum() / (label_mask | rotated).sum()
        assert pixel_iou > 0.999


def test_split_matches_original_algorithm_and_does_not_mutate_input():
    files = [f"img_{i:03d}.png" for i in range(57)]
    original = list(files)
    train, val = split_files(files, 0.8, seed=42)

    random.seed(42)  # implementación original (formatDatasetForUltralytics.py)
    expected = list(files)
    random.shuffle(expected)
    assert (train, val) == (expected[:45], expected[45:])
    assert files == original


def test_copy_split_creates_empty_label_for_background_images(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir()
    (dst / "images" / "train").mkdir(parents=True)
    (dst / "labels" / "train").mkdir(parents=True)
    for name in ("a", "b"):
        (src / "images" / f"{name}.png").write_bytes(b"x")
    (src / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1")

    copy_split([src / "images" / "a.png", src / "images" / "b.png"], src, dst, "train")

    assert (dst / "labels" / "train" / "a.txt").read_text() == "0 0.5 0.5 0.1 0.1"
    assert (dst / "labels" / "train" / "b.txt").read_text() == ""


def test_move_contents_skips_existing(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a").write_text("nuevo")
    (src / "b").write_text("nuevo")
    (dst / "b").write_text("viejo")

    assert move_contents(src, dst) == (1, 1)
    assert (dst / "a").read_text() == "nuevo"
    assert (dst / "b").read_text() == "viejo"
