"""Evalúa un modelo sobre un split de un dataset y guarda curvas y matrices de confusión.

La salida va a `runs/<task>/<name>/`. Convención de nombres: `val.<modelo><dataset>`,
p.ej. `val.mr` = modelo m evaluado sobre el dataset real.

Uso:
    uv run python -m plate_detectors.evaluate runs/detect/0.0.4.m+/weights/best.pt \\
        --data configs/observation.yaml --name val.mm+
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from plate_detectors.paths import run_name, run_project_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="pesos .pt del modelo")
    parser.add_argument("--data", required=True, help="YAML del dataset (ver configs/)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--name", help="nombre de la salida (default: val.<run>)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", help="p.ej. 0 o cpu (default: autodetectar)")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=run_project_dir(model.task),
        name=args.name or f"val.{run_name(args.model)}",
    )
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    print(f"[OK] Resultados en {metrics.save_dir}")


if __name__ == "__main__":
    main()
