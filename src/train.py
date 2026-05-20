from ultralytics import YOLO

CONFIGURATION_FILE = '/home/sponte/Repositorios/DetectorsWithUltralitics/src/configurationFiles/observation.yaml'
EPOCHS = 1000
BATCH_SIZE = 32
SEED = 42

# Load a COCO-pretrained YOLO26n model
#model = YOLO("yolo26n.yaml")
model = YOLO("/home/sponte/Repositorios/DetectorsWithUltralitics/runs/detect/0.0.4.m+/weights/last.pt")

# Train the model
results = model.train(
    data=CONFIGURATION_FILE, 
    batch=BATCH_SIZE,
    epochs=EPOCHS, 
    imgsz=1024,
    name="0.0.4.m+.1024",
    device=0,
    seed=SEED,
    # augmentation
    degrees=3.0,
    flipud=0.5,
    close_mosaic=50,
    # translate=0.05,
    # scale=0.3,
    # mosaic=0.,
    # mixup=0.5 # Mezcla de imagenes via transparencia
)

# Run inference with the YOLO26n model on the 'placa_ejemplo.jpg' image
results = model("/home/sponte/Repositorios/DetectorsWithUltralitics/images/placa_ejemplo.png")