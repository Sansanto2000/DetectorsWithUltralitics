from ultralytics import YOLO

CONFIGURATION_FILE = '/home/sponte/Repositorios/DetectorsWithUltralitics/src/configurationFiles/observation.yaml'

# Load a COCO-pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=CONFIGURATION_FILE, epochs=1, imgsz=640)

# Run inference with the YOLO26n model on the 'bus.jpg' image
results = model("path/to/bus.jpg")