"""Dibuja las etiquetas (bbox u OBB) de una o más imágenes de un dataset YOLO.

La etiqueta se busca en la carpeta `labels/` paralela a `images/`. La salida es
`<imagen>_labeled.png` en la carpeta de figuras.

Uso:
    uv run python -m plate_detectors.plots.show_labels /dataset/images/train/img_001.png \\
        --data configs/components.yaml
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch

from plate_detectors.detections import Detections
from plate_detectors.drawing import bgr_to_mpl, class_color, draw_polygons
from plate_detectors.labels import find_label_path
from plate_detectors.paths import FIGURES_DIR


def load_class_names(data_yaml: Path | None) -> dict[int, str]:
    if data_yaml is None:
        return {}
    names = yaml.safe_load(Path(data_yaml).read_text())["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def render_labels(image_path: Path, class_names: dict[int, str], output_dir: Path) -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"no se pudo leer la imagen {image_path}")
    h, w = img.shape[:2]

    label_path = find_label_path(image_path)
    if label_path is None:
        print(f"[WARN] {image_path.name}: sin archivo de etiquetas, se dibuja la imagen sola")
    gt = Detections.from_label_file(label_path, w, h)
    print(f"[INFO] {image_path.name}: {len(gt)} anotaciones ({label_path})")

    for cls, polygon in zip(gt.classes, gt.polygons, strict=True):
        color = class_color(cls)
        draw_polygons(img, polygon[None], color)
        cx, cy = polygon.mean(axis=0).astype(int)
        cv2.putText(img, str(cls), (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(image_path.name, fontsize=11, pad=10)
    handles = [
        Patch(color=bgr_to_mpl(class_color(c)), label=f"{c}: {class_names.get(c, 'clase')}")
        for c in np.unique(gt.classes)
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_labeled.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, nargs="+", help="imágenes a visualizar")
    parser.add_argument("--data", type=Path, help="YAML del dataset, para mostrar nombres de clase")
    parser.add_argument("--out", type=Path, default=FIGURES_DIR, help="carpeta de salida")
    args = parser.parse_args()

    class_names = load_class_names(args.data)
    for image_path in args.images:
        render_labels(image_path, class_names, args.out)


if __name__ == "__main__":
    main()
