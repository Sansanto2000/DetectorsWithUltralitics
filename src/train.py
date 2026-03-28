from ultralytics import YOLO

CONFIGURATION_FILE = '/home/sponte/Repositorios/DetectorsWithUltralitics/src/configurationFiles/observation.yaml'
EPOCHS = 500
BATCH_SIZE = 32

# Load a COCO-pretrained YOLO26n model
#model = YOLO("yolo26n.yaml")
model = YOLO("/home/sponte/Repositorios/DetectorsWithUltralitics/runs/detect/0.0.1.g/weights/best.pt")

# Train the model
results = model.train(
    data=CONFIGURATION_FILE, 
    batch=BATCH_SIZE,
    epochs=EPOCHS, 
    imgsz=640,
    name="0.0.3.m",
    device=0)

# Run inference with the YOLO26n model on the 'placa_ejemplo.jpg' image
results = model("/home/sponte/Repositorios/DetectorsWithUltralitics/images/placa_ejemplo.png")