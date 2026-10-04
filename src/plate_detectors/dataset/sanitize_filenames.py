"""Reemplaza espacios por `_` en los nombres de todos los archivos y carpetas bajo una raíz.

Si el nombre nuevo ya existe, no se renombra y se avisa.

Uso:
    uv run python -m plate_detectors.dataset.sanitize_filenames \\
        /mnt/data3/sponte/datasets/ReTrHO.Observaciones.19.05.26
"""

import argparse
from pathlib import Path


def sanitize_tree(root: Path) -> int:
    renamed = 0
    for path in root.rglob("*"):
        if " " not in path.name:
            continue
        new_path = path.with_name(path.name.replace(" ", "_"))
        if new_path.exists():
            print(f"[SKIP] ya existe {new_path}")
            continue
        path.rename(new_path)
        renamed += 1
        print(f"[RENAMED] {path} -> {new_path}")
    return renamed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="carpeta raíz a recorrer")
    args = parser.parse_args()
    print(f"[OK] {sanitize_tree(args.root)} elementos renombrados")


if __name__ == "__main__":
    main()
