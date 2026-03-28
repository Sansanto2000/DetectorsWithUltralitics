'''
Dado un dataset en formato YOLO rota todas sus imagenes 90 grados a la derecha.
Tambien rota las etiquetas sus etiquetas.
'''
import os
import re
import cv2
from pathlib import Path

"""
Dataset a rotar 90 grados se espera distribucion tipo:
dataset -|- images -|- train
         |          |- val
         |
         |- labels -|- train
                    |- val
"""
SOURCE_PATH = '/mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics'
"""
Ubicacion destino. 
Si la carpeta no existe la crea.
"""
DESTINY_PATH = '/mnt/data3/sponte/datasets/observaciones-etiquetadas.ultralytics.90'

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

def rotate_image_and_labels(
        img_path:Path, label_path:Path, 
        save_img_path:Path, save_label_path:Path):
    """Recive la ubicacion de una imagen y su etiqueta. Realiza las 
    operaciones correspondientes y guarda sus copias rotadas en las 
    ubicaciones especificadas.

    Args:
        img_path (Path): ubicacion de la imagen.
        label_path (Path): ubicacion de la etiqueta.
        save_img_path (Path): ubicacion destino de la imagen.
        save_label_path (Path): ubicacion destino de la etiqueta.
    """
    # Leer imagen
    img = cv2.imread(img_path)

    # Rotar imagen 90° clockwise
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Guardar imagen rotada
    cv2.imwrite(save_img_path, rotated)

    # Procesar etiquetas
    if not os.path.exists(label_path):
        print(f"⚠️  No se encontró etiqueta para {img_path}. Se omite rotación de esta imagen.")
        return

    new_lines = []
    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Limpiar caracteres raros tipo \r, \t, etc.
        clean_line = line.replace("\r", "").strip()

        # Separar por cualquier whitespace
        parts = re.split(r"\s+", clean_line)

        if len(parts) == 1:
            continue

        if len(parts) != 5:
            print(f"⚠️ [{label_path}] línea {i} inválida: '{line}'")
            continue

        try:
            cls, x, y, bw, bh = map(float, parts)
        except ValueError:
            print(f"⚠️ [{label_path}] error parseando línea {i}: '{line.strip()}'")
            continue

        # Transformación
        new_x = 1 - y
        new_y = x
        new_w = bh
        new_h = bw

        new_lines.append(f"{int(cls)} {new_x} {new_y} {new_w} {new_h}")

    # Guardar nuevas etiquetas
    with open(save_label_path, "w") as f:
        f.write("\n".join(new_lines))

def rotate_folder(source_path:str, destiny_path:str):
    """Realiza operacion de rotacion de todos los archivos en un carpeta en formato YOLO.

    Args:
        source_path (str): ubicacion de la carpeta con imagenes y etiquetas.
        destiny_path (str): ubicacion destino de las imagenes y etiquetas rotadas.
    """

    sub_folders = ['val', 'train']
    for sub_folder in sub_folders:
        image_path = os.path.join(source_path, "images", sub_folder)
        labels_path = os.path.join(source_path,"labels", sub_folder)
        out_images = os.path.join(destiny_path,"images", sub_folder)
        out_labels = os.path.join(destiny_path,"labels", sub_folder)
        for file in os.listdir(image_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(image_path, file)
                label_path = os.path.join(labels_path, file.replace(".jpg", ".txt").replace(".png", ".txt"))
                save_img_path = os.path.join(out_images, file)
                save_label_path = os.path.join(out_labels, file.replace(".jpg", ".txt").replace(".png", ".txt"))

                rotate_image_and_labels(img_path, label_path, save_img_path, save_label_path)

def main():
    """Ejecuta codigo principal acorde a lo configurado en las variables de entorno.
    Dataset de SOURCE_PATH es duplicado en DESTINY_PATH respetando el formato YOLO y 
    alterando cada entrada para que este rotada 90 grados en direccion de las agujas del reloj.
    """
    
    print("📁 Creando estructura destino...")
    create_dirs(DESTINY_PATH)

    print("📁 Rotando archivos...")
    rotate_folder(SOURCE_PATH, DESTINY_PATH)

    print("✅ Dataset rotado correctamente.")


if __name__ == "__main__":
    main()