from ultralytics import YOLO

model = YOLO("yolo11n-obb.pt")

results = model("imagen.jpg", show=True)