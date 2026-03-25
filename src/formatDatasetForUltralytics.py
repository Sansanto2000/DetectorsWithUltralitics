'''
Genera una copia de un dataset respetando la estructura de dataset requerida por YOLO.
'''
import os
import random
import shutil
from pathlib import Path

"""
Dataset a formatear se espera distribucion tipo:
dataset |- images   
        |- labels
"""
SOURCE_PATH = '/mnt/data3/sponte/datasets/conGSSSP.large.3'

"""
Ubicacion destino. El dataset formateado sigue la taxonomia:
dataset -|- images -|- train
         |          |- val
         |
         |- labels -|- train
                    |- val
Si la carpeta no existe la crea.
"""
DESTINY_PATH = '/mnt/data3/sponte/datasets/conGSSSP.large.3.ultralytics'

"""
Porcentajen de datos para train, los demas van a val.
"""
SPLIT_SIZE = 0.8 
SEED = 42

def create_dirs(base_path:str):
    """Dado un PATH crea una taxonomia de carpetas tipo YOLO
    para problmeas de deteccion.

    Args:
        base_path (string): ubicacion destino.
    """
    paths = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]
    for p in paths:
        os.makedirs(os.path.join(base_path, p), exist_ok=True)
        
def get_image_files(images_path:str) -> list[Path]:
    """Recupera el listado de archivos de imagen contenidos en una carpeta.

    Args:
        images_path (str): carpeta con imagenes.

    Returns:
        list[Path]: listado de archivos de tipo imagen.
    """
    
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    return [
        f for f in os.listdir(images_path)
        if f.lower().endswith(valid_ext)
    ]

def split_dataset(files:list[str], split_size:float, seed:int) -> tuple[list[str], list[str]]:
    """Divide el conjunto de archivos recibido en 2 conjuntos acorde a una proporcion.

    Args:
        files (list[str]): conjunto de archivos.
        split_size (float): proporcion de la divicion. Valor de 0.0 a 1.0 indica la
            proporcion de archivos que se quedan en el conjunto 1. Los restantes van al 
            conjunto 2.
        seed (int): semilla aleatoria.

    Returns:
        tuple[list[str], list[str]]: arreglos separados 
    """
    
    random.seed(seed)
    random.shuffle(files)

    split_index = int(len(files) * split_size)
    train_files = files[:split_index]
    val_files = files[split_index:]

    return train_files, val_files


def copy_files(file_list:list[str], subset:str):
    """Busca un listado de archivos en una carpeta y los copia la direccion especificada por
    variables de entorno. Dentro de un subconjunto.

    Args:
        file_list (list[str]): listado de archivos a buscar.
        subset (str): nombre del subconjunto.
    """

    for file_name in file_list:
        # Paths imagen
        src_img = os.path.join(SOURCE_PATH, "images", file_name)
        dst_img = os.path.join(DESTINY_PATH, "images", subset, file_name)

        # Nombre base
        base_name = os.path.splitext(file_name)[0]

        # Paths label
        src_lbl = os.path.join(SOURCE_PATH, "labels", base_name + ".txt")
        dst_lbl = os.path.join(DESTINY_PATH, "labels", subset, base_name + ".txt")

        # Copiar imagen
        shutil.copy2(src_img, dst_img)

        # Copiar label (si no existe, crear vacío)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            open(dst_lbl, "w").close()  # archivo vacío

def main():
    """Ejecuta codigo principal acorde a lo configurado en las variables de entorno.
    Dataset de SOURCE_PATH es duplicado en DESTINY_PATH respetando el formato YOLO.
    SEED es la semilla aleatoria. SPLIT_SIZE la razon de divicion.
    """
    images_path = os.path.join(SOURCE_PATH, "images")

    print("📂 Leyendo imágenes...")
    image_files = get_image_files(images_path)

    print(f"Total imágenes: {len(image_files)}")

    print("🔀 Dividiendo dataset...")
    train_files, val_files = split_dataset(image_files, SPLIT_SIZE, SEED)

    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    print("📁 Creando estructura...")
    create_dirs(DESTINY_PATH)

    print("📋 Copiando TRAIN...")
    copy_files(train_files, "train")

    print("📋 Copiando VAL...")
    copy_files(val_files, "val")

    print("✅ Dataset formateado correctamente.")


if __name__ == "__main__":
    main()