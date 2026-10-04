"""Copia un dataset YOLO rotando todas sus imágenes y etiquetas 90° en sentido horario.

Soporta etiquetas bbox y OBB. Un punto normalizado (x, y) pasa a (1 - y, x);
en bbox además se intercambian ancho y alto.

Uso:
    uv run python -m plate_detectors.dataset.rotate90 \\
        /mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics \\
        /mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics.90
"""

import argparse
from pathlib import Path

import cv2

from plate_detectors.dataset import make_yolo_dirs
from plate_detectors.labels import BBOX_NCOORDS, format_label_line, parse_label_line
from plate_detectors.paths import list_images


def rotate_coords_90cw(coords: list[float]) -> list[float]:
    if len(coords) == BBOX_NCOORDS:
        x, y, w, h = coords
        return [1 - y, x, h, w]
    rotated = []
    for x, y in zip(coords[0::2], coords[1::2], strict=True):
        rotated += [1 - y, x]
    return rotated


def rotate_label_file(src: Path, dst: Path) -> None:
    lines = []
    for i, line in enumerate(src.read_text().splitlines(), start=1):
        try:
            parsed = parse_label_line(line)
        except ValueError as e:
            print(f"[WARN] {src}:{i} línea inválida ({e}): {line.strip()!r}")
            continue
        if parsed is not None:
            cls, coords = parsed
            lines.append(format_label_line(cls, rotate_coords_90cw(coords)))
    dst.write_text("\n".join(lines))


def rotate_sample(img_path: Path, label_path: Path, out_img: Path, out_label: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] No se pudo leer {img_path}, se omite")
        return
    cv2.imwrite(str(out_img), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    if label_path.exists():
        rotate_label_file(label_path, out_label)
    else:
        print(f"[WARN] {img_path.name} no tiene etiqueta: se copia sólo la imagen rotada")


def rotate_dataset(source: Path, destination: Path) -> None:
    splits = tuple(sorted(p.name for p in (source / "images").iterdir() if p.is_dir()))
    make_yolo_dirs(destination, splits)
    for split in splits:
        images = list_images(source / "images" / split)
        print(f"[INFO] {split}: {len(images)} imágenes")
        for img_path in images:
            rotate_sample(
                img_path,
                source / "labels" / split / f"{img_path.stem}.txt",
                destination / "images" / split / img_path.name,
                destination / "labels" / split / f"{img_path.stem}.txt",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="dataset YOLO con images/<split>/ y labels/<split>/")
    parser.add_argument("destination", type=Path, help="destino (se crea si no existe)")
    args = parser.parse_args()

    rotate_dataset(args.source, args.destination)
    print(f"[OK] Dataset rotado en {args.destination}")


if __name__ == "__main__":
    main()
