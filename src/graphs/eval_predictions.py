"""
Evalúa visualmente las mejores y peores predicciones de un modelo YOLO (bbox estándar).
Genera dos collages: mejores y peores predicciones sobre el conjunto de test.

    Verde  = Ground Truth
    Azul   = Predicción del modelo

Uso:
    python eval_predictions.py <modelo.pt> <carpeta_imagenes>

Ejemplo:
    python eval_predictions.py runs/detect/0.0.2.r/weights/best.pt /dataset/images/test
"""

import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from ultralytics import YOLO

# --- Config ---
OUTPUT_DIR  = Path("/home/sponte/Repositorios/DetectorsWithUltralitics/src/graphs/show")
N_SHOW      = 6      # Imágenes por collage
GRID_COLS   = 3      # Columnas del grid
CONF_THRESH = 0.5   # Umbral de confianza

COLOR_GT   = (0, 180, 0)    # Verde  - Ground Truth
COLOR_PRED = (200, 0, 0)    # Azul   - Predicción


# ---------------------------------------------------------------------------
# Helpers de labels
# ---------------------------------------------------------------------------

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
    """Carga etiquetas YOLO (cx cy w h normalizados) → xyxy absoluto."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes) if boxes else np.empty((0, 4))


# ---------------------------------------------------------------------------
# Métrica por imagen
# ---------------------------------------------------------------------------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter_x1 = np.maximum(a[:, 0:1], b[:, 0])
    inter_y1 = np.maximum(a[:, 1:2], b[:, 1])
    inter_x2 = np.minimum(a[:, 2:3], b[:, 2])
    inter_y2 = np.minimum(a[:, 3:4], b[:, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_score(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Score por imagen en [0, 1].
    - Cada GT se empareja con la predicción de mayor IoU.
    - Se penaliza dividiendo por max(n_gt, n_pred) para castigar
      tanto falsos negativos como falsos positivos.
    """
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0
    iou = iou_matrix(gt, pred)
    return float(iou.max(axis=1).sum() / max(n_gt, n_pred))


# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------

def draw_boxes(img: np.ndarray, boxes: np.ndarray, color: tuple, thickness: int = 2):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def render_annotated(image_path: Path, gt: np.ndarray, pred: np.ndarray, score: float) -> np.ndarray:
    """Imagen con GT (verde) y predicciones (azul) superpuestas + score."""
    img = cv2.imread(str(image_path))
    draw_boxes(img, gt,   COLOR_GT,   thickness=2)
    draw_boxes(img, pred, COLOR_PRED, thickness=2)
    label = f"Score: {score:.3f}  GT:{len(gt)}  Pred:{len(pred)}"
    cv2.putText(img, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(img, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return img


def make_collage(images: list[np.ndarray], title: str, output_path: Path):
    cols   = GRID_COLS
    rows   = (len(images) + cols - 1) // cols
    h_tgt  = 400

    # Resize + pad al mismo ancho
    resized = []
    for img in images:
        h, w = img.shape[:2]
        new_w = int(w * h_tgt / h)
        resized.append(cv2.resize(img, (new_w, h_tgt)))

    max_w = max(r.shape[1] for r in resized)
    padded = []
    for r in resized:
        pad_w = max_w - r.shape[1]
        pad = np.zeros((h_tgt, pad_w, 3), dtype=np.uint8)
        padded.append(np.hstack([r, pad]))

    # Completar grid
    blank = np.zeros((h_tgt, max_w, 3), dtype=np.uint8)
    while len(padded) < rows * cols:
        padded.append(blank)

    grid_rows = [np.hstack(padded[r * cols:(r + 1) * cols]) for r in range(rows)]
    collage   = np.vstack(grid_rows)

    fig, ax = plt.subplots(figsize=(cols * 5, rows * 4 + 0.8))
    ax.imshow(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    legend = [
        Patch(color=np.array(COLOR_GT[::-1])  / 255, label="Ground Truth"),
        Patch(color=np.array(COLOR_PRED[::-1]) / 255, label="Predicción"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=10, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collage mejores/peores predicciones YOLO")
    parser.add_argument("model",       help="Ruta al modelo .pt")
    parser.add_argument("images_dir",  help="Carpeta con imágenes de test")
    parser.add_argument("--n",    type=int,   default=N_SHOW,      help="Imágenes por collage")
    parser.add_argument("--conf", type=float, default=CONF_THRESH, help="Umbral de confianza")
    args = parser.parse_args()

    model_path = Path(args.model)
    images_dir = Path(args.images_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Modelo:    {model_path}")
    print(f"[INFO] Imágenes:  {images_dir}")

    model = YOLO(str(model_path))

    valid_ext   = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in valid_ext)

    if not image_files:
        print("[ERROR] No se encontraron imágenes.")
        sys.exit(1)

    print(f"[INFO] Procesando {len(image_files)} imágenes...")

    records = []
    for img_path in image_files:
        result    = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        pred_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))

        raw  = cv2.imread(str(img_path))
        h, w = raw.shape[:2]
        lbl  = find_label_path(img_path)
        gt   = load_gt_boxes(lbl, w, h) if lbl else np.empty((0, 4))

        score = compute_score(gt, pred_xyxy)
        records.append({"path": img_path, "pred": pred_xyxy, "gt": gt, "score": score})

    # Ordenar de mejor a peor
    records.sort(key=lambda x: x["score"], reverse=True)

    def to_imgs(subset):
        return [render_annotated(r["path"], r["gt"], r["pred"], r["score"]) for r in subset]

    model_name = model_path.parent.parent.name  # e.g. "0.0.2.r"
    total      = len(records)
    page_size  = args.n
    n_pages    = (total + page_size - 1) // page_size

    print(f"[INFO] Generando {n_pages} collage(s) para {total} imágenes...")

    for page_idx in range(n_pages):
        start  = page_idx * page_size
        end    = min(start + page_size, total)
        subset = records[start:end]

        if page_idx == 0:
            label = "best"
            tag   = "Mejores predicciones"
        elif page_idx == n_pages - 1:
            label = "worst"
            tag   = "Peores predicciones"
        else:
            label = f"mid_{page_idx:02d}"
            tag   = f"Predicciones intermedias (pág. {page_idx + 1}/{n_pages})"

        score_lo    = subset[-1]["score"]
        score_hi    = subset[0]["score"]
        score_range = f"  |  Score: {score_lo:.3f}–{score_hi:.3f}"
        rank_range  = f"  |  #{start + 1}–{end} de {total}"

        make_collage(
            to_imgs(subset),
            title=f"{tag} — {model_name}{score_range}{rank_range}",
            output_path=OUTPUT_DIR / f"{model_name}_{label}.png",
        )


if __name__ == "__main__":
    main()