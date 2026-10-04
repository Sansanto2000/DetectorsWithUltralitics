"""Grafica una métrica de validación en función de un hiperparámetro, a lo largo de varios runs.

El valor del hiperparámetro se lee de `args.yaml` y la métrica de `results.csv`
de cada run (máximo a lo largo de las épocas, o la última época con `--at last`).

Uso (efecto del tamaño de imagen en el fine-tuning 0.0.4.m+):
    uv run python -m plate_detectors.plots.metric_vs_param \\
        runs/detect/0.0.4.m+ runs/detect/0.0.4.m+.1024 runs/detect/0.0.4.m+.2048 \\
        --param imgsz --out figures/imgsz_vs_map50-95.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from plate_detectors.paths import FIGURES_DIR  # noqa: E402

DEFAULT_METRIC = "metrics/mAP50-95(B)"


def read_run(run_dir: Path, param: str, metric: str, at: str) -> tuple[float, float]:
    args = yaml.safe_load((run_dir / "args.yaml").read_text())
    with open(run_dir / "results.csv") as f:
        values = [float(row[metric]) for row in csv.DictReader(f)]
    return float(args[param]), (max(values) if at == "best" else values[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="carpetas de runs de Ultralytics")
    parser.add_argument("--param", default="imgsz", help="clave de args.yaml para el eje x")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="columna de results.csv para el eje y")
    parser.add_argument("--at", choices=["best", "last"], default="best", help="máximo o última época")
    parser.add_argument("--title", default="Rendimiento según hiperparámetro")
    parser.add_argument("--out", type=Path, default=FIGURES_DIR / "metric_vs_param.png")
    args = parser.parse_args()

    points = sorted((*read_run(r, args.param, args.metric, args.at), r.name) for r in args.runs)
    for x, y, name in points:
        print(f"  {name}: {args.param}={x:g}  {args.metric}={y:.4f}")
    xs, ys, _ = zip(*points)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    for x, y, _ in points:
        ax.annotate(f"({x:g}, {y:.3f})", (x, y), textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel(args.param)
    ax.set_ylabel(f"{args.metric} ({args.at})")
    ax.set_title(args.title)
    ax.grid(True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()
