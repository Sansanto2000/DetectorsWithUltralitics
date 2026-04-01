# DetectorsWithUltralitics
Repositorio preparado para realizar entrenamientos de modelos de detección de objetos basados en YOLO.

## Problemas de drivers GPU
### Imposibilidad de actualizar
Si actualizar los drivers no es una opcion reducir la version de torch con el siguiente comando.
```
uv pip install torch==2.1.2 torchvision==0.16.2   --index-url https://download.pytorch.org/whl/cu118
```
A la hora de ejecutar usar el flag `--no-sync` para evitar sobreescritura de las versiones de torch.
```
uv run --no-sync src/test.py
```

## Otros formatos
Para exportar el modelo otros formatos puede usar las siguientes instrucciones.

### TFJS
Ejecute el siguiete comando en la consola.
```
tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```
