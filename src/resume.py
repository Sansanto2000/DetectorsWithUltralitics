from ultralytics import YOLO

EPOCHS = 1000 # Nueva epoco final
SEED = 42
MODEL_PATH = "/home/sponte/Repositorios/DetectorsWithUltralitics/runs/detect/0.0.4.m+/weights/last.pt"

# Load a COCO-pretrained YOLO26n model
#model = YOLO("yolo26n.yaml")
model = YOLO(MODEL_PATH)

# Train the model
results = model.train(
    resume=True,
    epochs=EPOCHS, 
)

# Run inference with the YOLO26n model on the 'placa_ejemplo.jpg' image
results = model("/home/sponte/Repositorios/DetectorsWithUltralitics/images/placa_ejemplo.png")