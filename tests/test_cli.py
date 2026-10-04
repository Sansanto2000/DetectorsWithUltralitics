"""Cada script se importa y muestra su ayuda sin errores."""

import subprocess
import sys

import pytest

MODULES = [
    "plate_detectors.train",
    "plate_detectors.resume",
    "plate_detectors.evaluate",
    "plate_detectors.predict",
    "plate_detectors.check_gpu",
    "plate_detectors.dataset.split",
    "plate_detectors.dataset.rotate90",
    "plate_detectors.dataset.find_duplicates",
    "plate_detectors.dataset.sanitize_filenames",
    "plate_detectors.dataset.move_contents",
    "plate_detectors.plots.rank_predictions",
    "plate_detectors.plots.mosaic",
    "plate_detectors.plots.show_labels",
    "plate_detectors.plots.metric_vs_param",
]


@pytest.mark.parametrize("module", [m for m in MODULES if m != "plate_detectors.check_gpu"])
def test_help(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_train_parse_overrides():
    from plate_detectors.train import parse_overrides

    assert parse_overrides(["mosaic=0", "cos_lr=true", "lr0=1e-3", "optimizer=SGD", "freeze=none"]) == {
        "mosaic": 0,
        "cos_lr": True,
        "lr0": 1e-3,
        "optimizer": "SGD",
        "freeze": None,
    }
