"""Utilidades para preparar datasets en formato YOLO (Ultralytics).

Estructura esperada de un dataset listo para entrenar:

    dataset/
    ├── images/{train,val}/
    └── labels/{train,val}/
"""

from pathlib import Path

SPLITS = ("train", "val")


def make_yolo_dirs(root: Path, splits: tuple[str, ...] = SPLITS) -> None:
    for kind in ("images", "labels"):
        for split in splits:
            (Path(root) / kind / split).mkdir(parents=True, exist_ok=True)
