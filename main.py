import torch
from yolov5 import YOLOv5

# Load the YOLOv8 model
model = YOLOv5('yolov8n.pt')

# Function to run inference on an image
def run_inference(image_path):
    # Load the image
    img = torch.load(image_path)
    
    # Run inference
    results = model(img)
    
    # Print results
    results.print()  # Print results to console
    results.show()   # Display results
    results.save()   # Save results to file

# Example usage
if __name__ == "__main__":
    image_path = 'path/to/your/image.jpg'
    run_inference(image_path)