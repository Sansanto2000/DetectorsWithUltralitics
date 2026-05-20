from pathlib import Path
import shutil
import tqdm

SOURCE = "/ruta/origen"
TARGET = "/ruta/destino"

# Carpetas
source_dir = Path(SOURCE)
target_dir = Path(TARGET)

# Crear destino si no existe
target_dir.mkdir(parents=True, exist_ok=True)

# Mover todos los archivos y carpetas
items = list(source_dir.iterdir())
for item in tqdm(items, desc="Moviendo archivos"):
    destination = target_dir / item.name
    if destination.exists():
        print(f"[SKIP] Ya existe: {destination}")
        continue
    shutil.move(str(item), str(target_dir / item.name))

print("Contenido movido correctamente.")