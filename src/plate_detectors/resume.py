"""Reanuda un entrenamiento interrumpido desde su `last.pt`.

Ultralytics restaura TODOS los argumentos originales del run. Sólo permite
cambiar algunos (imgsz, batch, device, workers, patience, ...): en particular
`epochs` NO se puede extender al reanudar. Para entrenar más épocas hay que
lanzar un entrenamiento nuevo partiendo de esos pesos:

    uv run python -m plate_detectors.train --model runs/detect/0.0.4.m+/weights/last.pt ...

Uso:
    uv run python -m plate_detectors.resume runs/obb/0.0.5.r.obb
    uv run python -m plate_detectors.resume runs/obb/0.0.5.r.obb/weights/last.pt --batch 16
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from plate_detectors.paths import SAMPLE_IMAGES_DIR
from plate_detectors.predict import predict_images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="carpeta del run o su weights/last.pt")
    parser.add_argument("--device", help="p.ej. 0 o cpu")
    parser.add_argument("--batch", type=int, help="reducir si falta memoria")
    parser.add_argument("--no-samples", action="store_true", help="no correr inferencia sobre images/ al terminar")
    args = parser.parse_args()

    checkpoint = args.checkpoint / "weights" / "last.pt" if args.checkpoint.is_dir() else args.checkpoint
    model = YOLO(str(checkpoint))
    overrides = {k: v for k, v in {"device": args.device, "batch": args.batch}.items() if v is not None}
    model.train(resume=True, **overrides)

    save_dir = Path(model.trainer.save_dir)
    print(f"[OK] Run guardado en {save_dir}")
    if not args.no_samples:
        predict_images(save_dir / "weights" / "best.pt", [SAMPLE_IMAGES_DIR], save_dir / "samples")


if __name__ == "__main__":
    main()
