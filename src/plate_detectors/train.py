"""Entrena un modelo YOLO y luego corre inferencia sobre las imágenes de ejemplo.

Los defaults reproducen el experimento 0.0.5.r.pretrained.obb: arquitectura
yolo26n-obb inicializada con los pesos preentrenados de `weights/`, 1000 épocas,
batch 32, imgsz 640, seed 42, degrees=3.0 y flipud=0.5.

Cualquier otro argumento de entrenamiento de Ultralytics se pasa como
`clave=valor` al final (https://docs.ultralytics.com/modes/train/#train-settings).

Uso:
    # Desde pesos preentrenados (pipeline OBB actual)
    uv run python -m plate_detectors.train --data configs/components.yaml --name 0.0.6.r.obb

    # Fine-tuning sobre datos reales partiendo de un run anterior (p.ej. sintético -> real)
    uv run python -m plate_detectors.train --data configs/observation.yaml --name 0.0.3.m \\
        --model runs/detect/0.0.1.g/weights/best.pt

    # Overrides de Ultralytics
    uv run python -m plate_detectors.train --data configs/components.yaml --name prueba \\
        mosaic=0 close_mosaic=50 patience=200
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.cfg import smart_value

from plate_detectors.paths import SAMPLE_IMAGES_DIR, WEIGHTS_DIR, run_project_dir
from plate_detectors.predict import predict_images

# Aumentaciones usadas desde 0.0.4 en adelante.
DEFAULT_OVERRIDES = {"degrees": 3.0, "flipud": 0.5}


def parse_overrides(items: list[str]) -> dict:
    """`["mosaic=0", "cos_lr=true"]` -> `{"mosaic": 0, "cos_lr": True}`, con el mismo parseo que el CLI `yolo`."""
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"override inválido {item!r}, se espera clave=valor")
        overrides[key] = smart_value(value)
    return overrides


def default_pretrained_weights(model: str) -> Path | None:
    """Para una arquitectura `.yaml`, sus pesos preentrenados en `weights/` (si existen)."""
    if not model.endswith(".yaml"):
        return None
    weights = WEIGHTS_DIR / Path(model).with_suffix(".pt").name
    return weights if weights.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="YAML del dataset (ver configs/)")
    parser.add_argument("--name", required=True, help="nombre del run, p.ej. 0.0.6.r.obb")
    parser.add_argument(
        "--model",
        default="yolo26n-obb.yaml",
        help="arquitectura .yaml (yolo26n.yaml = bbox, yolo26n-obb.yaml = OBB) o pesos .pt para fine-tuning",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="pesos preentrenados a cargar sobre la arquitectura (default: weights/<model>.pt si existe)",
    )
    parser.add_argument("--from-scratch", action="store_true", help="no cargar pesos preentrenados")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="p.ej. 0, 0,1 o cpu (default: autodetectar)")
    parser.add_argument("--no-samples", action="store_true", help="no correr inferencia sobre images/ al terminar")
    parser.add_argument("overrides", nargs="*", help="argumentos extra de Ultralytics como clave=valor")
    args = parser.parse_args()

    model = YOLO(args.model)
    weights = None if args.from_scratch else (args.weights or default_pretrained_weights(args.model))
    if weights:
        print(f"[INFO] Cargando pesos preentrenados {weights}")
        model = model.load(str(weights))

    train_args = {
        "data": args.data,
        "project": run_project_dir(model.task),
        "name": args.name,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "deterministic": True,
        "device": args.device,
        **DEFAULT_OVERRIDES,
        **parse_overrides(args.overrides),
    }
    model.train(**train_args)

    save_dir = Path(model.trainer.save_dir)
    print(f"[OK] Run guardado en {save_dir}")
    if not args.no_samples:
        predict_images(save_dir / "weights" / "best.pt", [SAMPLE_IMAGES_DIR], save_dir / "samples")


if __name__ == "__main__":
    main()
