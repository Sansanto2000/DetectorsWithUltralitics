from pathlib import Path

# Carpeta raíz a recorrer
ROOT_PATH = Path(r"/mnt/data3/sponte/datasets/ReTrHO.Observaciones.19.05.26")

# Recorre toda la jerarquía
for path in ROOT_PATH.rglob("*"):
    
    # Solo archivos y carpetas con espacios
    if " " in path.name:
        new_name = path.name.replace(" ", "_")
        new_path = path.with_name(new_name)

        # Evitar sobrescribir
        if new_path.exists():
            continue

        path.rename(new_path)

        print(f"[RENAMED] {path} -> {new_path}")

print("Proceso finalizado.")