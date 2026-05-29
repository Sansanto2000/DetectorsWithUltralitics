import cv2
import numpy as np

img = cv2.imread(
    "/mnt/data3/sponte/datasets/components.obb.merge/images/train/1b2d7255-mu_Cen_Hu_133_mu_Cen.png"
)

label_path = (
    "/mnt/data3/sponte/datasets/components.obb.merge/labels/train/"
    "1b2d7255-mu_Cen_Hu_133_mu_Cen.txt"
)

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
}

with open(label_path, "r") as f:
    lines = f.readlines()

h, w = img.shape[:2]

for line in lines:
    values = list(map(float, line.strip().split()))

    cls = int(values[0])
    coords = values[1:]

    pts = np.array(coords, dtype=np.float32).reshape(4, 2)

    # Coordenadas normalizadas -> píxeles
    pts[:, 0] *= w
    pts[:, 1] *= h

    pts = pts.astype(np.int32)

    color = CLASS_COLORS.get(cls, (255, 255, 255))

    cv2.polylines(img, [pts], True, color, 2)

    x, y = pts[0]

    cv2.putText(
        img,
        str(cls),
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

cv2.imwrite(
    "/home/sponte/Repositorios/DetectorsWithUltralitics/src/graphs/show/ground_truth.jpg",
    img
)