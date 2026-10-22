"""Genera un dataset sintético con GASP, listo para entrenar con un YAML de `configs/`.

Pasos:
    1. Corre `gsssp.save_yolo_dataset` en el entorno de GASP (`uv run --project <GASP>`):
       GASP depende de TensorFlow y de otra versión de numpy, así que no puede
       importarse desde este repo.
    2. Remapea las clases de GASP a las del YAML destino, por nombre
       (`observacion` -> `observation`, `science`, `lamp`): GASP usa índices fijos
       (observacion=0, science=1, lamp=2) que no coinciden con los de `configs/`.
    3. Separa train/val y escribe `data.yaml` con las clases del YAML destino.
    4. Valida cada imagen con el mismo chequeo que Ultralytics usa para descartar.

Por defecto genera placas de 500-1400 px por lado (los mismos rangos que
`gsssp.save_yolo_dataset`) llevadas con letterbox a 1280x1280: conserva la proporción,
como Ultralytics con las imágenes reales, y ocupa ~0.29 MB por imagen. Para placas con
la proporción de las reales: `--height-range 900 2000 --width-range 2800 4500`.

Uso:
    # Observaciones con etiquetas OBB
    uv run python -m plate_detectors.dataset.generate_synthetic \\
        /mnt/data3/sponte/datasets/gasp.observation.obb --images 96000

    # Componentes (lamp/science)
    uv run python -m plate_detectors.dataset.generate_synthetic \\
        /mnt/data3/sponte/datasets/gasp.components.obb --data configs/components.yaml
"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import yaml
from ultralytics.data.utils import verify_image_label

from plate_detectors.dataset import make_yolo_dirs
from plate_detectors.dataset.split import split_files
from plate_detectors.paths import CONFIGS_DIR, REPO_ROOT, list_images

# Nombres de clase de GASP que difieren de los de `configs/`.
GASP_NAME_ALIASES = {"observacion": "observation"}

# Se ejecuta dentro del entorno de GASP; recibe la configuración como JSON en argv[1].
GASP_DRIVER = """
import json, sys
from gsssp import LabelClass, LabelFormat, ResizeMode, save_yolo_dataset
cfg = json.loads(sys.argv[1])
by_name = {c.yolo_name: c for c in LabelClass}
extra = {"resize_shape": tuple(cfg["resize_shape"])} if cfg["resize_shape"] else {}
save_yolo_dataset(
    cfg["output_dir"],
    batch_count=cfg["batch_count"],
    batch_size=cfg["batch_size"],
    seed=cfg["seed"],
    label_format=LabelFormat[cfg["label_format"]],
    label_classes=tuple(by_name[n] for n in cfg["label_classes"]),
    resize_mode=ResizeMode[cfg["resize_mode"]],
    height_range=tuple(cfg["height_range"]),
    width_range=tuple(cfg["width_range"]),
    **extra,
)
"""


def load_names(data_yaml: Path) -> dict[int, str]:
    names = yaml.safe_load(Path(data_yaml).read_text())["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def to_target_name(gasp_name: str) -> str:
    return GASP_NAME_ALIASES.get(gasp_name, gasp_name)


def to_gasp_name(target_name: str) -> str:
    inverse = {v: k for k, v in GASP_NAME_ALIASES.items()}
    return inverse.get(target_name, target_name)


def class_mapping(gasp_names: dict[int, str], target_names: dict[int, str]) -> dict[int, int]:
    """Índice de GASP -> índice destino, para las clases presentes en el destino."""
    target_index = {name: idx for idx, name in target_names.items()}
    return {idx: target_index[to_target_name(name)] for idx, name in gasp_names.items()
            if to_target_name(name) in target_index}


def remap_label_file(path: Path, mapping: dict[int, int]) -> None:
    lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cls, rest = line.split(maxsplit=1)
        if int(cls) not in mapping:
            raise ValueError(f"{path}: clase {cls} sin equivalente en el YAML destino")
        lines.append(f"{mapping[int(cls)]} {rest}")
    path.write_text("\n".join(lines))


def run_gasp(gasp_dir: Path, config: dict) -> None:
    cmd = ["uv", "run", "--project", str(gasp_dir), "python", "-c", GASP_DRIVER, json.dumps(config)]
    print(f"[INFO] Generando con GASP ({gasp_dir})...")
    subprocess.run(cmd, check=True)


def split_into(raw: Path, output: Path, val_fraction: float, seed: int) -> dict[str, int]:
    """Mueve imágenes y etiquetas de `raw/` a `output/{images,labels}/{train,val}/`."""
    make_yolo_dirs(output)
    train, val = split_files(list_images(raw / "images"), 1 - val_fraction, seed)
    for split, images in (("train", train), ("val", val)):
        for img in images:
            shutil.move(img, output / "images" / split / img.name)
            shutil.move(raw / "labels" / f"{img.stem}.txt", output / "labels" / split / f"{img.stem}.txt")
    return {"train": len(train), "val": len(val)}


def write_data_yaml(output: Path, names: dict[int, str]) -> Path:
    data = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": names}
    path = output / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return path


def validate(output: Path, num_classes: int) -> tuple[int, dict[int, int], list[str]]:
    """Corre `verify_image_label` de Ultralytics sobre todo el dataset.

    Returns:
        (imágenes, cajas por clase, mensajes de las imágenes que Ultralytics descartaría)
    """
    n_images, per_class, rejected = 0, {}, []
    for split in ("train", "val"):
        for img in list_images(output / "images" / split):
            label = output / "labels" / split / f"{img.stem}.txt"
            result = verify_image_label((str(img), str(label), "", False, num_classes, 0, 0, False))
            n_images += 1
            if result[0] is None:
                rejected.append(result[-1].strip())
                continue
            for cls in result[1][:, 0].astype(int):
                per_class[cls] = per_class.get(cls, 0) + 1
    return n_images, per_class, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="carpeta del dataset (no debe existir o estar vacía)")
    parser.add_argument("--data", type=Path, default=CONFIGS_DIR / "observation.yaml",
                        help="YAML destino: define qué clases se etiquetan y con qué índices")
    parser.add_argument("--images", type=int, default=96000, help="cantidad de imágenes (se redondea al lote)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0, help="semilla de GASP y del split")
    parser.add_argument("--label-format", choices=["obb", "aabb"], default="obb")
    parser.add_argument("--resize-mode", choices=["letterbox", "none"], default="letterbox",
                        help="letterbox conserva la proporción; none deja la placa a tamaño original (~1.5 MB/img)")
    parser.add_argument("--imgsz", type=int, default=1280, help="lado del letterbox")
    parser.add_argument("--height-range", type=int, nargs=2, default=[500, 1400], metavar=("MIN", "MAX"),
                        help="rango en px (default 500 1400)")
    parser.add_argument("--width-range", type=int, nargs=2, default=[500, 1400], metavar=("MIN", "MAX"),
                        help="rango en px (default 500 1400)")
    parser.add_argument("--gasp-dir", type=Path, default=REPO_ROOT.parent / "GSSSP",
                        help="clon local de GASP (default: carpeta hermana GSSSP)")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"{args.output} ya existe y no está vacía")
    if not (args.gasp_dir / "pyproject.toml").exists():
        parser.error(f"no se encontró GASP en {args.gasp_dir}; indicarlo con --gasp-dir")

    target_names = load_names(args.data)
    raw = args.output / "_gasp"
    batch_count = math.ceil(args.images / args.batch_size)
    run_gasp(args.gasp_dir, {
        "output_dir": str(raw.resolve()),
        "batch_count": batch_count,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "label_format": args.label_format.upper(),
        "label_classes": [to_gasp_name(n) for n in target_names.values()],
        "resize_mode": args.resize_mode.upper(),
        "resize_shape": [args.imgsz, args.imgsz] if args.resize_mode == "letterbox" else None,
        "height_range": args.height_range,
        "width_range": args.width_range,
    })

    mapping = class_mapping(load_names(raw / "data.yaml"), target_names)
    print(f"[INFO] Remapeo de clases GASP -> destino: {mapping}")
    for label in (raw / "labels").glob("*.txt"):
        remap_label_file(label, mapping)

    counts = split_into(raw, args.output, args.val_fraction, args.seed)
    shutil.rmtree(raw)
    data_yaml = write_data_yaml(args.output, target_names)
    print(f"[INFO] Train: {counts['train']} | Val: {counts['val']}")

    n_images, per_class, rejected = validate(args.output, len(target_names))
    summary = ", ".join(f"{target_names[c]}={n}" for c, n in sorted(per_class.items()))
    print(f"[INFO] Cajas por clase: {summary}")
    if rejected:
        print(f"[WARN] Ultralytics descartaría {len(rejected)} de {n_images} imágenes:")
        for msg in rejected[:10]:
            print(f"  {msg}")
    print(f"[OK] {n_images} imágenes en {args.output} ({data_yaml})")


if __name__ == "__main__":
    main()
