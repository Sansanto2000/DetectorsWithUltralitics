# DetectorsWithUltralitics

Entrenamiento y evaluación de detectores YOLO (Ultralytics) para encontrar
observaciones en escaneos de placas espectroscópicas.

El pipeline preentrena con imágenes sintéticas generadas con
[GASP](https://github.com/Sansanto2000/GASP) y hace fine-tuning sobre escaneos
reales. La versión actual usa cajas orientadas (OBB).

## Instalación

```
uv sync
```

## Uso

Todos los scripts son módulos del paquete `plate_detectors` y aceptan `--help`.

```
# Entrenar (defaults = experimento 0.0.5.r.pretrained.obb)
uv run python -m plate_detectors.train --data configs/components.yaml --name 0.0.6.r.obb

# Fine-tuning desde un run anterior, con overrides de Ultralytics
uv run python -m plate_detectors.train --data configs/observation.yaml --name 0.0.3.m \
    --model runs/detect/0.0.1.g/weights/best.pt mosaic=0

# Reanudar un entrenamiento interrumpido
uv run python -m plate_detectors.resume runs/obb/0.0.5.r.obb

# Evaluar
uv run python -m plate_detectors.evaluate runs/obb/0.0.5.r.obb/weights/best.pt \
    --data configs/components.yaml --name val.rr

# Inferencia sobre imágenes
uv run python -m plate_detectors.predict runs/obb/0.0.5.r.obb/weights/best.pt images/
```

Preparación de datasets (`plate_detectors.dataset.*`): `split`, `rotate90`,
`find_duplicates`, `sanitize_filenames`, `move_contents`.

Figuras (`plate_detectors.plots.*`, salida en `figures/`): `rank_predictions`,
`mosaic`, `show_labels`, `metric_vs_param`.

Tests: `uv run pytest`.

## Problemas de drivers GPU
### Imposibilidad de actualizar
Si actualizar los drivers no es una opcion reducir la version de torch con el siguiente comando.
```
uv pip install torch==2.1.2 torchvision==0.16.2   --index-url https://download.pytorch.org/whl/cu118
```
A la hora de ejecutar usar el flag `--no-sync` para evitar sobreescritura de las versiones de torch.
```
uv run --no-sync python -m plate_detectors.check_gpu
```

## Exportar

### ONNX
```
uv run yolo export model=runs/detect/0.0.3.m/weights/best.pt format=onnx
```

### TFJS
```
tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```
