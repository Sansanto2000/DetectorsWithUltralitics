"""Mueve todo el contenido de una carpeta a otra, sin sobrescribir lo que ya exista en el destino.

Uso:
    uv run python -m plate_detectors.dataset.move_contents /ruta/origen /ruta/destino
"""

import argparse
import shutil
from pathlib import Path


def move_contents(source: Path, target: Path) -> tuple[int, int]:
    target.mkdir(parents=True, exist_ok=True)
    items = sorted(source.iterdir())
    moved = skipped = 0
    for i, item in enumerate(items, start=1):
        destination = target / item.name
        if destination.exists():
            print(f"[SKIP] Ya existe: {destination}")
            skipped += 1
            continue
        shutil.move(str(item), str(destination))
        moved += 1
        if i % 1000 == 0:
            print(f"[INFO] {i}/{len(items)}")
    return moved, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    moved, skipped = move_contents(args.source, args.target)
    print(f"[OK] {moved} movidos, {skipped} omitidos")


if __name__ == "__main__":
    main()
