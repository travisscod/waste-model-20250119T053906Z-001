from ultralytics import YOLO

# Load a pretrained YOLOv8 classification model
model = YOLO('yolov8n-cls.pt')

# Train the model
results = model.train(
    data='dataset_split',
    epochs=100,
    imgsz=224,
    batch=32,
    device='0',
    project='waste_classification',
    name='yolov8_waste_classifier',
    exist_ok=True  # Allow overwriting existing experiment
)