from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26n.pt")

# Export the model to TF.js format
model.export(format="tfjs")  # creates '/yolo26n_web_model'

# Load the exported TF.js model
tfjs_model = YOLO("./yolo26n_web_model")

# Run inference
results = tfjs_model("https://ultralytics.com/images/bus.jpg")