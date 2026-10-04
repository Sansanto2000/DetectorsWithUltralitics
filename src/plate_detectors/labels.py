"""Lectura y escritura de etiquetas en formato YOLO.

Cada línea de un `.txt` de etiquetas es una de:
    bbox: `cls cx cy w h`                  (4 coordenadas normalizadas)
    OBB:  `cls x1 y1 x2 y2 x3 y3 x4 y4`    (4 vértices normalizados)

Internamente ambas se representan como polígonos de 4 vértices, de modo que
el resto del código (IoU, dibujo, rotación) no distingue entre formatos.
"""

from pathlib import Path

import numpy as np

BBOX_NCOORDS = 4
OBB_NCOORDS = 8


def label_path_for(image_path: Path) -> Path:
    """Ruta esperada de la etiqueta de una imagen, siguiendo la convención de Ultralytics:
    se reemplaza el último componente `images` por `labels` y la extensión por `.txt`.
    Si la ruta no contiene `images`, la etiqueta se busca junto a la imagen.
    """
    parts = list(Path(image_path).parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def find_label_path(image_path: Path) -> Path | None:
    """Como `label_path_for`, pero devuelve None si el archivo no existe."""
    path = label_path_for(image_path)
    return path if path.exists() else None


def parse_label_line(line: str) -> tuple[int, list[float]] | None:
    """Parsea una línea de etiqueta. Devuelve None si la línea está vacía.

    Raises:
        ValueError: si la línea no tiene formato bbox ni OBB.
    """
    parts = line.split()
    if not parts:
        return None
    coords = [float(v) for v in parts[1:]]
    if len(coords) not in (BBOX_NCOORDS, OBB_NCOORDS):
        raise ValueError(f"se esperaban {BBOX_NCOORDS} u {OBB_NCOORDS} coordenadas, hay {len(coords)}")
    return int(float(parts[0])), coords


def format_label_line(cls: int, coords: list[float]) -> str:
    return " ".join([str(cls), *(str(v) for v in coords)])


def bbox_to_polygon(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    """Bbox centro/tamaño -> 4 vértices (sentido horario desde arriba a la izquierda)."""
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def coords_to_polygon(coords: list[float]) -> np.ndarray:
    if len(coords) == BBOX_NCOORDS:
        return bbox_to_polygon(*coords)
    return np.asarray(coords, dtype=np.float64).reshape(4, 2)


def read_labels(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lee un archivo de etiquetas.

    Returns:
        classes (N,) int y polygons (N, 4, 2) en coordenadas normalizadas.
        Las líneas mal formadas se reportan y se omiten.
    """
    classes, polygons = [], []
    for i, line in enumerate(Path(label_path).read_text().splitlines(), start=1):
        try:
            parsed = parse_label_line(line)
        except ValueError as e:
            print(f"[WARN] {label_path}:{i} línea inválida ({e}): {line.strip()!r}")
            continue
        if parsed is None:
            continue
        cls, coords = parsed
        classes.append(cls)
        polygons.append(coords_to_polygon(coords))
    if not classes:
        return np.empty(0, dtype=int), np.empty((0, 4, 2))
    return np.array(classes, dtype=int), np.stack(polygons)
