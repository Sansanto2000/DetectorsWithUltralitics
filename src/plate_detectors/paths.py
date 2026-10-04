"""Rutas del repositorio. Todas se resuelven relativas a la raíz, nunca absolutas."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = REPO_ROOT / "configs"
FIGURES_DIR = REPO_ROOT / "figures"
RUNS_DIR = REPO_ROOT / "runs"
SAMPLE_IMAGES_DIR = REPO_ROOT / "images"
WEIGHTS_DIR = REPO_ROOT / "weights"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(folder: Path) -> list[Path]:
    """Imágenes de una carpeta (no recursivo), ordenadas por nombre."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and is_image(p))


def run_project_dir(task: str) -> str:
    """Valor de `project` para Ultralytics: guarda los runs en `runs/<task>/` del repo.

    Sin esto Ultralytics usa `runs_dir` de su archivo de settings global, que
    depende de dónde se ejecutó por primera vez en la máquina.
    """
    return str(RUNS_DIR / task)


def run_name(weights: Path) -> str:
    """Nombre del run a partir de sus pesos: `runs/obb/0.0.5.r.obb/weights/best.pt` -> `0.0.5.r.obb`."""
    weights = Path(weights)
    return weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
