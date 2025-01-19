from ultralytics import YOLO

# Load your trained model
model = YOLO('waste_classification/yolov8_waste_classifier/weights/best.pt')

# Export to TFLite format
model.export(format='tflite', int8=True)  # int8 quantization for EdgeTPU compatibility