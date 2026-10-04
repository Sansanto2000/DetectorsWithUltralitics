from pathlib import Path

import numpy as np
import pytest

from plate_detectors.detections import Detections
from plate_detectors.labels import find_label_path, label_path_for, parse_label_line, read_labels


def test_label_path_replaces_last_images_component():
    assert label_path_for(Path("/d/images/x/images/val/a.png")) == Path("/d/images/x/labels/val/a.txt")


def test_label_path_without_images_dir_is_sibling():
    assert label_path_for(Path("/d/a.png")) == Path("/d/a.txt")


def test_find_label_path_returns_none_when_missing(tmp_path):
    assert find_label_path(tmp_path / "images" / "a.png") is None


def test_parse_bbox_and_obb():
    assert parse_label_line("0 0.5 0.5 0.2 0.1") == (0, [0.5, 0.5, 0.2, 0.1])
    assert parse_label_line("1 0 0 1 0 1 1 0 1\r") == (1, [0, 0, 1, 0, 1, 1, 0, 1])
    assert parse_label_line("   ") is None


@pytest.mark.parametrize("line", ["0", "0 0.5 0.5", "0 1 2 3 4 5 6"])
def test_parse_rejects_unknown_formats(line):
    with pytest.raises(ValueError):
        parse_label_line(line)


def test_bbox_and_equivalent_obb_give_same_polygon(tmp_path):
    # rectángulo x∈[0.1, 0.4], y∈[0.2, 0.6]
    label = tmp_path / "a.txt"
    label.write_text("0 0.25 0.4 0.3 0.4\n0 0.1 0.2 0.4 0.2 0.4 0.6 0.1 0.6\n")
    classes, polygons = read_labels(label)
    assert classes.tolist() == [0, 0]
    np.testing.assert_allclose(polygons[0], polygons[1])


def test_read_labels_skips_invalid_lines(tmp_path, capsys):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.2 0.1\nbasura\n\n")
    classes, _ = read_labels(label)
    assert len(classes) == 1
    assert "línea inválida" in capsys.readouterr().out


def test_detections_scale_to_pixels(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.1 0.2 0.4 0.2 0.4 0.6 0.1 0.6\n")
    gt = Detections.from_label_file(label, img_w=200, img_h=100)
    np.testing.assert_allclose(gt.polygons[0].min(axis=0), [20, 20])
    np.testing.assert_allclose(gt.polygons[0].max(axis=0), [80, 60])
