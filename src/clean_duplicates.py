'''
Dada una carpeta revisar mirar la similiritud de todas las imagenes 
que contine para detectar duplicados.
'''

from PIL import Image
import imagehash
import os

DATASET_PATH = '/mnt/data3/sponte/datasets/observaciones-etiquetadas.clean/images'


def get_hash_with_rotations(img):
    hashes = []
    for angle in [0, 90, 180, 270]:
        rotated = img.rotate(angle, expand=True)
        hashes.append(imagehash.phash(rotated))
    return hashes


def get_hashes(folder):
    hashes = {}
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, file)
            img = Image.open(path).convert("L")  # escala de grises mejora estabilidad
            hashes[file] = get_hash_with_rotations(img)
    return hashes


def min_hash_diff(hash_list1, hash_list2):
    min_diff = float("inf")
    for h1 in hash_list1:
        for h2 in hash_list2:
            diff = h1 - h2
            if diff < min_diff:
                min_diff = diff
    return min_diff


def find_similar(hashes, threshold=5):
    files = list(hashes.keys())
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            diff = min_hash_diff(hashes[files[i]], hashes[files[j]])
            if diff < threshold:
                print(f"{files[i]} ~ {files[j]} (diff={diff})")


hashes = get_hashes(DATASET_PATH)
find_similar(hashes)