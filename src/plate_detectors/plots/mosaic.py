"""Mosaico de imágenes elegidas a mano con GT (verde) y predicciones (azul).

Uso:
    uv run python -m plate_detectors.plots.mosaic \\
        runs/detect/0.0.3.m/weights/best.pt \\
        /dataset/images/val/188.png /dataset/images/val/35.png ...
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from plate_detectors.detections import Detections
from plate_detectors.drawing import COMPARISON_LEGEND, render_comparison, save_collage
from plate_detectors.paths import FIGURES_DIR, run_name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="pesos .pt del modelo")
    parser.add_argument("images", type=Path, nargs="+", help="imágenes a incluir, en orden")
    parser.add_argument("--cols", type=int, default=4, help="columnas de la grilla")
    parser.add_argument("--conf", type=float, default=0.25, help="umbral de confianza")
    parser.add_argument("--out", type=Path, default=FIGURES_DIR, help="carpeta de salida")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    renders = []
    for img_path in args.images:
        result = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        h, w = result.orig_shape
        gt = Detections.ground_truth_for(img_path, w, h)
        renders.append(render_comparison(img_path, gt, Detections.from_result(result)))

    name = run_name(args.model)
    save_collage(renders, args.out / f"{name}_mosaic.png", f"Mosaico — {name}", args.cols, COMPARISON_LEGEND)


if __name__ == "__main__":
    main()
