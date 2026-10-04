"""Corre un modelo sobre imágenes y guarda las predicciones dibujadas.

Por defecto usa las imágenes de ejemplo de `images/` y guarda en
`runs/<task>/predict.<run>/`.

Uso:
    uv run python -m plate_detectors.predict runs/obb/0.0.5.r.pretrained.obb/weights/best.pt
    uv run python -m plate_detectors.predict best.pt /dataset/images/val --conf 0.5
"""

import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from plate_detectors.detections import Detections
from plate_detectors.paths import RUNS_DIR, SAMPLE_IMAGES_DIR, list_images, run_name


def expand_sources(sources: list[Path]) -> list[Path]:
    images = []
    for source in sources:
        images += list_images(source) if source.is_dir() else [source]
    return images


def predict_images(weights: Path, sources: list[Path], out_dir: Path, conf: float = 0.25) -> None:
    model = YOLO(str(weights))
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in expand_sources(sources):
        result = model.predict(str(img_path), conf=conf, verbose=False)[0]
        detections = Detections.from_result(result)
        counts = Counter(model.names[c] for c in detections.classes)
        summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items())) or "sin detecciones"
        print(f"  {img_path.name}: {summary}")
        result.save(filename=str(out_dir / f"{img_path.stem}_pred.jpg"))
    print(f"[OK] Predicciones en {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="pesos .pt del modelo")
    parser.add_argument("sources", type=Path, nargs="*", default=[SAMPLE_IMAGES_DIR], help="imágenes o carpetas")
    parser.add_argument("--conf", type=float, default=0.25, help="umbral de confianza")
    parser.add_argument("--out", type=Path, help="carpeta de salida (default: runs/<task>/predict.<run>)")
    args = parser.parse_args()

    out = args.out or RUNS_DIR / YOLO(str(args.model)).task / f"predict.{run_name(args.model)}"
    predict_images(args.model, args.sources, out, args.conf)


if __name__ == "__main__":
    main()
