# CLAUDE.md

Guía para agentes que trabajan en este repositorio.

## Qué es

Entrenamiento y evaluación de detectores YOLO (Ultralytics) para detectar
observaciones en escaneos de placas espectroscópicas. Pipeline:

1. Preentrenar con muchas imágenes sintéticas etiquetadas generadas con
   [GASP](https://github.com/Sansanto2000/GASP) (librería propia del autor).
2. Fine-tuning sobre escaneos reales (hay pocos).

Hay dos tareas, cada una con su YAML en `configs/`:

| Config | Clases | Formato | Notas |
|---|---|---|---|
| `observation.yaml` | `observation` | bbox (runs 0.0.1–0.0.4) | observación completa |
| `components.yaml` | `lamp`, `science` | **OBB** (0.0.5, actual) | espectros de lámpara de comparación y de ciencia |

## Entorno

- Se desarrolla en una Mac. **Se entrena en un servidor con GPU**: el repo vive en
  `/home/sponte/Repositorios/DetectorsWithUltralitics` y los datasets en
  `/mnt/data3/sponte/datasets/` (las rutas `path:` de `configs/*.yaml` apuntan ahí).
  Los datasets **no** están en la Mac: para probar localmente, armar un mini
  dataset sintético (ver `tests/test_dataset.py`) o usar las imágenes de `images/`.
- Python 3.10, gestionado con `uv`. `numpy==1.26.4` está fijado por compatibilidad
  con torch 2.1.2, que se instala a mano en el servidor (ver README); ahí se corre
  con `uv run --no-sync`.

## Comandos

```bash
uv sync                                   # instala el paquete en modo editable + dev deps
uv run pytest                             # tests (~10 s, sin GPU ni datasets)
uv run python -m plate_detectors.<módulo> --help
```

## Estructura

```
configs/                 YAML de datasets para Ultralytics
src/plate_detectors/
  paths.py               rutas del repo (REPO_ROOT, RUNS_DIR, FIGURES_DIR, ...) y run_name()
  labels.py              formato de etiquetas YOLO (bbox 5 col / OBB 9 col) -> polígonos
  detections.py          Detections: GT y predicciones como polígonos (N,4,2) en píxeles
  geometry.py            IoU de polígonos convexos, matching 1-a-1, score por imagen
  drawing.py             dibujo de polígonos y collages
  train.py resume.py evaluate.py predict.py check_gpu.py
  dataset/               split, rotate90, find_duplicates, sanitize_filenames, move_contents
  plots/                 rank_predictions, mosaic, show_labels, metric_vs_param
tests/                   pytest
runs/{detect,obb}/       salidas de Ultralytics (versionadas, ver abajo)
figures/                 figuras generadas (versionadas)
weights/                 pesos base: yolo26n.pt (COCO, bbox), yolo26n-obb.pt (DOTA, OBB)
images/                  3 escaneos de ejemplo para inferencia rápida
```

## Convenciones

- **Nada de rutas absolutas en el código.** Rutas del repo vía `plate_detectors.paths`;
  rutas de datos por argumentos de CLI. Cada script es un módulo con `argparse` y un
  docstring con ejemplo de uso (`description=__doc__`).
- **bbox y OBB se tratan igual**: todo se convierte a polígonos de 4 vértices
  (`labels.coords_to_polygon`, `Detections.from_result`). Código nuevo de evaluación o
  dibujo debe trabajar sobre `Detections`, no sobre `result.boxes`.
- Al entrenar/evaluar pasar `project=paths.run_project_dir(model.task)` para que la
  salida caiga en `runs/<task>/` del repo.
- Identificadores en inglés; docstrings, comentarios y mensajes en español.
  Prefijos de log: `[INFO]`, `[WARN]`, `[OK]`.
- `runs/`, `figures/` y los `.pt` **se versionan a propósito** (decisión del autor).
  No agregarlos a `.gitignore`. Los runs de prueba sí se borran antes de commitear.
- No commitear ni pushear sin confirmación del autor.
- Antes de afirmar algo sobre el comportamiento del código, ejecutar algo que lo
  demuestre y mostrar los números.

## Nomenclatura de runs

`<versión>.<datos>[.<variante>]`, con `<datos>`:

- `g`: sólo sintético (GASP)
- `r`: sólo real
- `m`: fine-tuning sobre real partiendo de `g`
- `+`: nuevo dataset real (ReTrHO)

Las evaluaciones son `val.<modelo><dataset>`; por ejemplo, `val.gr` es el modelo `g` evaluado
sobre el dataset real.

| Run | Parte de | Datos | imgsz | mejor mAP50-95 (val) |
|---|---|---|---|---|
| detect/0.0.1.g | yolo26n.yaml | sintético | 640 | 0.990 |
| detect/0.0.2.r | yolo26n.yaml | real | 640 | 0.894 |
| detect/0.0.3.m | 0.0.1.g best | real | 640 | 0.949 |
| detect/0.0.4.m+ | 0.0.3.m best | real ReTrHO | 640 | 0.840 |
| detect/0.0.4.m+.1024 | 0.0.4.m+ last | real ReTrHO | 1024 | 0.843 |
| detect/0.0.4.m+.2048 | 0.0.4.m+ last | real ReTrHO | 2048 | 0.813 |
| obb/0.0.5.r.obb | yolo26n-obb.yaml (sin preentrenar) | components | 640 | 0.753 |
| obb/0.0.5.r.pretrained.obb | yolo26n-obb.pt (DOTA) | components | 640 | 0.783 |

Los `args.yaml` de runs viejos referencian rutas anteriores al refactor
(`src/configurationFiles/*.yaml`, `yolo26n-obb.pt` en la raíz). Hoy esos archivos están en `configs/` y `weights/`.

## Trampas conocidas (verificadas)

- **`resume` no puede extender `epochs`.** Ultralytics restaura los args originales
  y sólo permite cambiar `imgsz, batch, device, workers, patience, time, ...`
  (`check_resume` en `ultralytics/engine/trainer.py`). Para entrenar más, lanzar
  `train --model runs/.../last.pt`.
- **Modelos OBB devuelven `result.obb` y `result.boxes is None`**; los de detección, al revés.
  Usar `Detections.from_result`.
- **Ultralytics descarta en silencio imágenes con etiquetas fuera de [0, 1]**
  (`ignoring corrupt image/label: non-normalized or out of bounds coordinates`).
  Sólo deja un warning. Pasa fácil con OBB que tocan el borde.
- Sin `project` explícito, Ultralytics guarda en `runs_dir` de su settings global
  (`~/Library/Application Support/Ultralytics/settings.json` o `~/.config/Ultralytics/`),
  que depende de dónde se usó por primera vez. En la Mac del autor apunta a otro
  repositorio (`ExportWithTFJS/runs`).
- Los overrides `clave=valor` de `train` se parsean con `ultralytics.cfg.smart_value`
  (igual que el CLI `yolo`); `yaml.safe_load` leería `1e-3` como string.
- `rank_predictions` puntúa con matching 1-a-1 y por clase (`--class-agnostic` para
  desactivarlo). La versión anterior permitía que una predicción cubriera varios GT,
  y eso inflaba el score.
- En tests de raster: `cv2.fillPoly` toma los vértices enteros como centros de píxel.
  Restar 0.5 al rasterizar coordenadas continuas (ver `fill_mask` en `tests/test_dataset.py`).
- `figures/imgsz_vs_map50-95.png` se generó con valores tipeados a mano (0.835/0.833/0.812)
  que no coinciden con ningún `results.csv`. `plots.metric_vs_param` lo regenera desde los runs
  (0.840/0.843/0.813, mejor época).
