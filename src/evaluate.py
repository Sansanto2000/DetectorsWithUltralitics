from ultralytics import YOLO

MODEL_PATH = "/home/sponte/Repositorios/DetectorsWithUltralitics/runs/detect/0.0.1.g/weights/best.pt"
CONFIGURATION_FILE = '/home/sponte/Repositorios/DetectorsWithUltralitics/src/configurationFiles/observation.yaml'

# Load model
model = YOLO(MODEL_PATH)

# Evaluate over val
metrics = model.val(
    data=CONFIGURATION_FILE,
    split="val",   # usa el conjunto de validación
    imgsz=640,
    batch=32,
    device=0,
    name="val.gr"
)