"""Dibujo de detecciones y armado de collages."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from plate_detectors.detections import Detections  # noqa: E402

# Colores en BGR (OpenCV).
COLOR_GT = (0, 180, 0)  # verde
COLOR_PRED = (200, 0, 0)  # azul
CLASS_COLORS = [
    (56, 56, 255),
    (56, 255, 56),
    (255, 56, 56),
    (0, 200, 255),
    (255, 200, 0),
    (255, 0, 200),
]

COLLAGE_ROW_HEIGHT = 400


def class_color(cls: int) -> tuple[int, int, int]:
    return CLASS_COLORS[cls % len(CLASS_COLORS)]


def bgr_to_mpl(color: tuple[int, int, int]) -> np.ndarray:
    return np.array(color[::-1]) / 255


def draw_polygons(img: np.ndarray, polygons: np.ndarray, color, thickness: int = 2) -> None:
    pts = [np.round(p).astype(np.int32) for p in polygons]
    cv2.polylines(img, pts, isClosed=True, color=color, thickness=thickness)


def put_caption(img: np.ndarray, text: str) -> None:
    """Texto blanco con borde negro en la esquina superior izquierda."""
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 2)):
        cv2.putText(img, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness)


def render_comparison(image_path: Path, gt: Detections, pred: Detections, caption: str = "") -> np.ndarray:
    """Imagen con GT (verde) y predicciones (azul) superpuestas."""
    img = cv2.imread(str(image_path))
    draw_polygons(img, gt.polygons, COLOR_GT)
    draw_polygons(img, pred.polygons, COLOR_PRED)
    put_caption(img, f"{caption}GT:{len(gt)}  Pred:{len(pred)}")
    return img


COMPARISON_LEGEND = [
    Patch(color=bgr_to_mpl(COLOR_GT), label="Ground Truth"),
    Patch(color=bgr_to_mpl(COLOR_PRED), label="Predicción"),
]


def make_grid(images: list[np.ndarray], cols: int, row_height: int = COLLAGE_ROW_HEIGHT) -> np.ndarray:
    """Une imágenes BGR en una grilla: misma altura, padding negro a la derecha."""
    resized = [cv2.resize(img, (round(img.shape[1] * row_height / img.shape[0]), row_height)) for img in images]
    max_w = max(r.shape[1] for r in resized)
    cells = [cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1], cv2.BORDER_CONSTANT, value=0) for r in resized]
    rows = -(-len(cells) // cols)
    cells += [np.zeros((row_height, max_w, 3), dtype=np.uint8)] * (rows * cols - len(cells))
    return np.vstack([np.hstack(cells[r * cols : (r + 1) * cols]) for r in range(rows)])


def save_collage(
    images: list[np.ndarray],
    output_path: Path,
    title: str,
    cols: int,
    legend: list[Patch] | None = None,
) -> None:
    grid = make_grid(images, cols)
    rows = -(-len(images) // cols)
    fig, ax = plt.subplots(figsize=(cols * 5, rows * 4 + 0.8))
    ax.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    if legend:
        ax.legend(handles=legend, loc="lower right", fontsize=10, framealpha=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")
