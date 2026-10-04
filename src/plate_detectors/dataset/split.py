"""Copia un dataset plano (`images/` + `labels/`) a la estructura train/val de YOLO.

    origen/                      destino/
    ├── images/        ──►       ├── images/{train,val}/
    └── labels/                  └── labels/{train,val}/

Las imágenes sin etiqueta se copian con un `.txt` vacío (imagen de fondo).
Los archivos que ya existen en el destino no se sobrescriben, lo que permite
fusionar varios orígenes en un mismo destino.

Uso:
    uv run python -m plate_detectors.dataset.split \\
        /mnt/data3/sponte/datasets/components.obb/otros \\
        /mnt/data3/sponte/datasets/components.obb.merge --train-fraction 0.8 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

from plate_detectors.dataset import make_yolo_dirs
from plate_detectors.paths import list_images


def split_files(files: list[Path], train_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Mezcla con `seed` y separa en (train, val). No modifica `files`."""
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    split_index = int(len(shuffled) * train_fraction)
    return shuffled[:split_index], shuffled[split_index:]


def copy_split(images: list[Path], source: Path, destination: Path, split: str) -> None:
    for src_img in images:
        dst_img = destination / "images" / split / src_img.name
        src_lbl = source / "labels" / f"{src_img.stem}.txt"
        dst_lbl = destination / "labels" / split / f"{src_img.stem}.txt"

        if dst_img.exists():
            print(f"[EXISTE IMG] {dst_img}")
        else:
            shutil.copy2(src_img, dst_img)

        if dst_lbl.exists():
            print(f"[EXISTE LBL] {dst_lbl}")
        elif src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            dst_lbl.touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="dataset con images/ y labels/")
    parser.add_argument("destination", type=Path, help="destino (se crea si no existe)")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="fracción de imágenes para train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_files = list_images(args.source / "images")
    train, val = split_files(image_files, args.train_fraction, args.seed)
    print(f"Total imágenes: {len(image_files)} | Train: {len(train)} | Val: {len(val)}")

    make_yolo_dirs(args.destination)
    copy_split(train, args.source, args.destination, "train")
    copy_split(val, args.source, args.destination, "val")
    print(f"[OK] Dataset en {args.destination}")


if __name__ == "__main__":
    main()
