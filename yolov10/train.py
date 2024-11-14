import os
from ultralytics import YOLO

# Define paths
dataset_yaml = 'dataset.yaml'
model_path = 'yolov10s.pt'  # YOLOv10-S model checkpoint

# Load the model
model = YOLO(model_path)

# Train the model
model.train(data=dataset_yaml, epochs=50, imgsz=400, batch=16, device="cpu")

# Save the trained model
model.save('yolov10s_custom_trained.pt')

# Evaluate the model on the test dataset
results = model.val(data=dataset_yaml, split='test')  # 'split' specifies the test dataset
print(results)