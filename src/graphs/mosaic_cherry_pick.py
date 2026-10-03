"""
Arma un mosaico con imágenes específicas y sus predicciones.
Configurar MODEL_PATH e IMAGE_PATHS antes de ejecutar.

Uso:
    python mosaic.py
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from ultralytics import YOLO

# --- Configuración ---
OUTPUT_DIR  = Path("/home/sponte/Repositorios/DetectorsWithUltralitics/src/graphs/show")
GRID_COLS   = 4
CONF_THRESH = 0.25
COLOR_GT    = (0, 180, 0)
COLOR_PRED  = (200, 0, 0)

MODEL_PATH = r"/home/sponte/Repositorios/DetectorsWithUltralitics/runs/detect/0.0.3.m/weights/best.pt"
IMAGE_PATHS = [
    r"/mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics.90/images/val/188.png",
    r"/mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics.90/images/train/E_6840.png",
    r"/mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics.90/images/val/35.png",
    r"/mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics.90/images/train/0.png",
    r"/mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics.90/images/val/99.png",
    r"/mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics.90/images/val/139.png",
    r"/mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics.90/images/train/E5121.png",
    r"/mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics.90/images/val/C_3829.png",
]

def find_label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    try:
        idx = next(i for i, p in enumerate(parts) if p == "images")
    except StopIteration:
        candidate = image_path.with_suffix(".txt")
        return candidate if candidate.exists() else None
    parts[idx] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    return label_path if label_path.exists() else None


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            boxes.append([(cx - w/2)*img_w, (cy - h/2)*img_h,
                           (cx + w/2)*img_w, (cy + h/2)*img_h])
    return np.array(boxes) if boxes else np.empty((0, 4))


def draw_boxes(img, boxes, color, thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def render_annotated(image_path: Path, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    img = cv2.imread(str(image_path))
    draw_boxes(img, gt,   COLOR_GT)
    draw_boxes(img, pred, COLOR_PRED)
    label = f"GT:{len(gt)}  Pred:{len(pred)}"
    cv2.putText(img, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3)
    cv2.putText(img, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    return img


def main():
    model_path  = Path(MODEL_PATH)
    image_paths = [Path(p) for p in IMAGE_PATHS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    h_tgt   = 400
    renders = []

    for img_path in image_paths:
        result    = model.predict(str(img_path), conf=CONF_THRESH, verbose=False)[0]
        pred_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))

        raw  = cv2.imread(str(img_path))
        h, w = raw.shape[:2]
        lbl  = find_label_path(img_path)
        gt   = load_gt_boxes(lbl, w, h) if lbl else np.empty((0, 4))

        renders.append(render_annotated(img_path, gt, pred_xyxy))

    # Resize + pad
    resized = []
    for img in renders:
        h, w = img.shape[:2]
        resized.append(cv2.resize(img, (int(w * h_tgt / h), h_tgt)))

    max_w = max(r.shape[1] for r in resized)
    padded = []
    for r in resized:
        pad = np.zeros((h_tgt, max_w - r.shape[1], 3), dtype=np.uint8)
        padded.append(np.hstack([r, pad]))

    cols  = GRID_COLS
    rows  = (len(padded) + cols - 1) // cols
    blank = np.zeros((h_tgt, max_w, 3), dtype=np.uint8)
    while len(padded) < rows * cols:
        padded.append(blank)

    grid = np.vstack([np.hstack(padded[r*cols:(r+1)*cols]) for r in range(rows)])

    model_name = model_path.parent.parent.name
    fig, ax = plt.subplots(figsize=(cols * 5, rows * 4 + 0.8))
    ax.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(f"Mosaico — {model_name}", fontsize=14, fontweight="bold", pad=10)
    legend = [
        Patch(color=np.array(COLOR_GT[::-1])  / 255, label="Ground Truth"),
        Patch(color=np.array(COLOR_PRED[::-1]) / 255, label="Predicción"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=10, framealpha=0.8)
    plt.tight_layout()

    out = OUTPUT_DIR / f"{model_name}_mosaic.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()