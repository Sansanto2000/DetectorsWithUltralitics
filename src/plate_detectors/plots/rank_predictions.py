"""Ordena las imágenes de un split por calidad de predicción y genera collages paginados.

    Verde = Ground Truth    Azul = Predicción

El score de cada imagen es la suma de IoU emparejados 1-a-1 dividida por
max(n_gt, n_pred) (ver `plate_detectors.geometry.image_score`). Funciona con
modelos y etiquetas bbox u OBB.

Genera `<modelo>_best.png`, `<modelo>_mid_NN.png` y `<modelo>_worst.png`, de
mejor a peor, y un `<modelo>_scores.csv` con el score de cada imagen.

Uso:
    uv run python -m plate_detectors.plots.rank_predictions \\
        runs/obb/0.0.5.r.pretrained.obb/weights/best.pt /dataset/images/val
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from plate_detectors.detections import Detections
from plate_detectors.drawing import COMPARISON_LEGEND, render_comparison, save_collage
from plate_detectors.geometry import image_score
from plate_detectors.paths import FIGURES_DIR, list_images, run_name


def score_images(model: YOLO, image_files: list[Path], conf: float, class_aware: bool) -> list[dict]:
    records = []
    for img_path in image_files:
        result = model.predict(str(img_path), conf=conf, verbose=False)[0]
        h, w = result.orig_shape
        pred = Detections.from_result(result)
        gt = Detections.ground_truth_for(img_path, w, h)
        records.append(
            {"path": img_path, "gt": gt, "pred": pred, "score": image_score(gt, pred, class_aware)}
        )
    return records


def page_label(page_idx: int, n_pages: int) -> tuple[str, str]:
    if page_idx == 0:
        return "best", "Mejores predicciones"
    if page_idx == n_pages - 1:
        return "worst", "Peores predicciones"
    return f"mid_{page_idx:02d}", f"Predicciones intermedias (pág. {page_idx + 1}/{n_pages})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="pesos .pt del modelo")
    parser.add_argument("images_dir", type=Path, help="carpeta con imágenes (las etiquetas se buscan en labels/)")
    parser.add_argument("--per-page", type=int, default=6, help="imágenes por collage")
    parser.add_argument("--cols", type=int, default=3, help="columnas de la grilla")
    parser.add_argument("--conf", type=float, default=0.5, help="umbral de confianza")
    parser.add_argument("--class-agnostic", action="store_true", help="emparejar GT y predicciones sin mirar la clase")
    parser.add_argument("--out", type=Path, default=FIGURES_DIR, help="carpeta de salida")
    args = parser.parse_args()

    image_files = list_images(args.images_dir)
    if not image_files:
        parser.error(f"no hay imágenes en {args.images_dir}")

    name = run_name(args.model)
    print(f"[INFO] Modelo: {args.model} | Imágenes: {len(image_files)} en {args.images_dir}")
    records = score_images(YOLO(str(args.model)), image_files, args.conf, not args.class_agnostic)
    records.sort(key=lambda r: r["score"], reverse=True)

    args.out.mkdir(parents=True, exist_ok=True)
    scores_path = args.out / f"{name}_scores.csv"
    with open(scores_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "score", "n_gt", "n_pred"])
        for r in records:
            writer.writerow([r["path"].name, f"{r['score']:.4f}", len(r["gt"]), len(r["pred"])])
    mean = sum(r["score"] for r in records) / len(records)
    print(f"[OK] {scores_path} (score medio {mean:.3f})")

    total = len(records)
    n_pages = -(-total // args.per_page)
    print(f"[INFO] Generando {n_pages} collage(s)...")
    for page_idx in range(n_pages):
        start = page_idx * args.per_page
        page = records[start : start + args.per_page]
        label, tag = page_label(page_idx, n_pages)
        title = (
            f"{tag} — {name}  |  Score: {page[-1]['score']:.3f}–{page[0]['score']:.3f}"
            f"  |  #{start + 1}–{start + len(page)} de {total}"
        )
        images = [
            render_comparison(r["path"], r["gt"], r["pred"], caption=f"Score: {r['score']:.3f}  ")
            for r in page
        ]
        save_collage(images, args.out / f"{name}_{label}.png", title, args.cols, COMPARISON_LEGEND)


if __name__ == "__main__":
    main()
