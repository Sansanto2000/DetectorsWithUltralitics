from ultralytics import YOLO

# CONFIGURATION_FILE = '/home/sponte/Repositorios/DetectorsWithUltralitics/src/configurationFiles/observation.yaml'
CONFIGURATION_FILE = '/home/sponte/Repositorios/DetectorsWithUltralitics/src/configurationFiles/components.yaml'
EPOCHS = 1000
BATCH_SIZE = 32
SEED = 42
SAVE_NAME = "0.0.5.r.pretrained.obb"
IMAGE_SIZE = 640
PRETRAINED = True

# Load a COCO-pretrained YOLO26n model
# model = YOLO("yolo26n.yaml")       # Para Bounding Boxes
model = YOLO("yolo26n-obb.yaml")    # Para bounding boxes orientados
# model = YOLO("/home/sponte/Repositorios/" \
#             "DetectorsWithUltralitics/runs/" \
#             "detect/0.0.4.m+/weights/last.pt")    # Retomar entrenameiento anterior

# Preentrenar
if PRETRAINED:
    model = model.load("yolo26n-obb.pt")

# Train the model
results = model.train(
    data=CONFIGURATION_FILE, 
    deterministic=True,
    batch=BATCH_SIZE,
    epochs=EPOCHS, 
    imgsz=IMAGE_SIZE,
    name=SAVE_NAME,
    device=0,
    seed=SEED,
    # augmentation
    degrees=3.0,
    flipud=0.5,
    # close_mosaic=50,
    # translate=0.05,
    # scale=0.3,
    # mosaic=0.,
    # mixup=0.5 # Mezcla de imagenes via transparencia
)


best_model = YOLO(
    f"runs/obb/{SAVE_NAME}/weights/best.pt"
)
print("--------------------------------------------------------------")
# Run inference with the YOLO26n model on the '4-observaciones.png' image
print("👁️‍🗨️Running inference on the trained model over 4-observaciones.png")
results = best_model("/home/sponte/Repositorios/DetectorsWithUltralitics/images/4-observaciones.png")
print("Inference results:", results)
print("--------------------------------------------------------------")
# Run inference with the YOLO26n model on the '3-inclinadas.png' image
print("👁️‍🗨️Running inference on the trained model over 3-inclinadas.png")
results = best_model("/home/sponte/Repositorios/DetectorsWithUltralitics/images/3-inclinadas.png")
print("Inference results:", results)
print("--------------------------------------------------------------")
# Run inference with the YOLO26n model on the '1-fideo.png' image
print("👁️‍🗨️Running inference on the trained model over 1-fideo.png")
results = best_model("/home/sponte/Repositorios/DetectorsWithUltralitics/images/1-fideo.png")
print("Inference results:", results)
print("--------------------------------------------------------------")