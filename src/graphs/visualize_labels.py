"""
Visualiza una imagen de un dataset YOLO junto con sus etiquetas OBB.

Uso:
    python visualize_labels.py <ruta_imagen>

Ejemplo:
    python visualize_labels.py /dataset/images/train/img_001.png

La etiqueta se busca automáticamente en la carpeta 'labels' correspondiente.
Funciona con datasets en formato YOLO estándar (bbox) y OBB (8 coordenadas).
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# --- Config visual ---
COLORS = [
    (255, 56,  56 ),
    (56,  255, 56 ),
    (56,  56,  255),
    (255, 200, 0  ),
    (0,   200, 255),
    (200, 0,   255),
]
LINE_THICKNESS = 2
FONT_SCALE = 0.6
SHOW_CLASS_ID = True


def find_label_path(image_path: Path) -> Path | None:
    """Dado el path de una imagen, busca el .txt de etiquetas correspondiente.
    Soporta estructura YOLO estándar (images/.../xxx -> labels/.../xxx).
    """
    parts = list(image_path.parts)
    try:
        idx = next(i for i, p in enumerate(parts) if p == "images")
    except StopIteration:
        # Fallback: mismo directorio, distinta extensión
        return image_path.with_suffix(".txt")

    parts[idx] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    return label_path if label_path.exists() else None


def parse_labels(label_path: Path):
    """Lee el archivo de etiquetas YOLO.

    Returns:
        list of (class_id, coords) donde coords es lista de floats.
        OBB  -> 8 valores  (x1 y1 x2 y2 x3 y3 x4 y4)
        BBOX -> 4 valores  (cx cy w h)
    """
    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            annotations.append((class_id, coords))
    return annotations


def draw_obb(img, coords, color, class_id):
    """Dibuja un polígono OBB de 4 vértices (coordenadas normalizadas)."""
    h, w = img.shape[:2]
    pts = np.array(coords, dtype=np.float32).reshape(4, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = pts.astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=LINE_THICKNESS)
    if SHOW_CLASS_ID:
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(img, str(class_id), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, LINE_THICKNESS)


def draw_bbox(img, coords, color, class_id):
    """Dibuja un bounding box estándar YOLO (cx cy w h normalizados)."""
    h, w = img.shape[:2]
    cx, cy, bw, bh = coords
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, LINE_THICKNESS)
    if SHOW_CLASS_ID:
        cv2.putText(img, str(class_id), (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, LINE_THICKNESS)


def visualize(image_path: str):
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"[ERROR] No existe la imagen: {image_path}")
        sys.exit(1)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] No se pudo leer la imagen: {image_path}")
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    label_path = find_label_path(image_path)

    if label_path is None:
        print("[WARN] No se encontró archivo de etiquetas. Mostrando imagen sin anotaciones.")
        annotations = []
    else:
        print(f"[INFO] Etiquetas: {label_path}")
        annotations = parse_labels(label_path)

    print(f"[INFO] Imagen: {image_path.name} | Anotaciones: {len(annotations)}")

    for class_id, coords in annotations:
        color = COLORS[class_id % len(COLORS)]
        if len(coords) == 8:
            draw_obb(img, coords, color, class_id)
        elif len(coords) == 4:
            draw_bbox(img, coords, color, class_id)
        else:
            print(f"[WARN] Formato desconocido ({len(coords)} valores) en clase {class_id}")

    # --- Mostrar ---
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(image_path.name, fontsize=11, pad=10)

    # Leyenda de clases presentes
    clases_presentes = sorted(set(c for c, _ in annotations))
    handles = [
        mpatches.Patch(color=np.array(COLORS[c % len(COLORS)]) / 255, label=f"clase {c}")
        for c in clases_presentes
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    output_dir = Path("/home/sponte/Repositorios/DetectorsWithUltralitics/src/graphs/show")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_labeled.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Guardado en: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python visualize_labels.py <ruta_imagen>")
        sys.exit(1)
    visualize(sys.argv[1])