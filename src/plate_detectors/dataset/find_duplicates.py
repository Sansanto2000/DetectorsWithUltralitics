"""Busca imágenes casi duplicadas en una carpeta, incluso si están rotadas 90°/180°/270°.

Compara perceptual hashes (pHash) de cada imagen en sus 4 rotaciones y reporta
los pares cuya distancia de Hamming mínima es menor al umbral. Sólo reporta,
no borra nada.

Uso:
    uv run python -m plate_detectors.dataset.find_duplicates \\
        /mnt/data3/sponte/datasets/observaciones-etiquetadas.clean/images --threshold 5
"""

import argparse
from itertools import combinations
from pathlib import Path

import imagehash
from PIL import Image

from plate_detectors.paths import list_images


def rotation_hashes(path: Path) -> list[imagehash.ImageHash]:
    img = Image.open(path).convert("L")  # escala de grises: hash más estable
    return [imagehash.phash(img.rotate(angle, expand=True)) for angle in (0, 90, 180, 270)]


def min_distance(hashes_a: list[imagehash.ImageHash], hashes_b: list[imagehash.ImageHash]) -> int:
    return min(a - b for a in hashes_a for b in hashes_b)


def find_similar(folder: Path, threshold: int) -> list[tuple[str, str, int]]:
    hashes = {p.name: rotation_hashes(p) for p in list_images(folder)}
    print(f"[INFO] {len(hashes)} imágenes en {folder}")
    similar = []
    for a, b in combinations(hashes, 2):
        distance = min_distance(hashes[a], hashes[b])
        if distance < threshold:
            similar.append((a, b, distance))
    return similar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="carpeta con imágenes")
    parser.add_argument("--threshold", type=int, default=5, help="distancia de Hamming máxima (exclusiva)")
    args = parser.parse_args()

    similar = find_similar(args.folder, args.threshold)
    for a, b, distance in similar:
        print(f"{a} ~ {b} (diff={distance})")
    print(f"[OK] {len(similar)} pares similares")


if __name__ == "__main__":
    main()
