import os
import cv2
import numpy as np
import torch
import time
import psutil  # For CPU/Memory monitoring
import pandas as pd  # For saving metrics to a CSV
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates  # For GPU monitoring
from src.preprocessing.image_processor import ImageProcessor  # Import your ImageProcessor class

# Initialize GPU monitoring if using GPU
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)

# Define image dimensions
img_width, img_height = 400, 224
kernel_size = (11, 11)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='C:\\Users\\bellv\\Desktop\\Uni\\5toA2doC\\Tesis\\EchoFlow\\yolov5\\runs\\train\\my_yolov5_model\\weights\\best.pt')
model.eval()

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 24)

# Instantiate ImageProcessor
#image_processor = ImageProcessor()
#image_processor.add_processing_step(ImageProcessor.convert_to_grayscale)
#image_processor.add_processing_step(
#    lambda img: ImageProcessor.resize_image(img, (img_width, img_height)))
#image_processor.add_processing_step(ImageProcessor.normalize_image_255)

image_processor = ImageProcessor()
image_processor.add_processing_step(ImageProcessor.convert_to_grayscale)
image_processor.add_processing_step(lambda img: ImageProcessor.denoise_image(img, kernel_size=kernel_size))
image_processor.add_processing_step(lambda img: ImageProcessor.resize_image(img, (img_width, img_height)))
image_processor.add_processing_step(ImageProcessor.edge_detection)
image_processor.add_processing_step(ImageProcessor.normalize_image_255)
image_processor.add_processing_step(image_processor.extract_sift_features)
image_processor.add_processing_step(image_processor.suppress_non_roi)
image_processor.add_processing_step(ImageProcessor.normalize_image_255)

# Define a function to preprocess the frame
def preprocess_frame(frame):
    return image_processor.process_image(frame)

# Define a function to detect objects
def detect(frame):
    results = model(frame)
    return results.xyxy[0].cpu().numpy()  # xmin, ymin, xmax, ymax, confidence, class

# Initialize metrics
metrics_data = {
    "frame": [],
    "preprocess_time": [],
    "inference_time": [],
    "fps": [],
    "cpu_usage": [],
    "gpu_usage": []
}

frame_count = 0
start_time = time.time()
total_inference_time = 0

# Run the loop for 20 seconds
time_limit = 20  # Time window in seconds

while cap.isOpened() and (time.time() - start_time) < time_limit:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Preprocess the frame
    preprocess_start = time.time()
    processed_frame = preprocess_frame(frame)
    preprocess_end = time.time()

    # Perform detection
    inference_start = time.time()
    detections = detect(frame)
    inference_end = time.time()

    # Calculate metrics
    inference_time = inference_end - inference_start
    total_inference_time += inference_time
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # System resource monitoring
    cpu_usage = psutil.cpu_percent()
    gpu_utilization = nvmlDeviceGetUtilizationRates(gpu_handle).gpu

    # Collect data for the current frame
    metrics_data["frame"].append(frame_count)
    metrics_data["preprocess_time"].append(preprocess_end - preprocess_start)
    metrics_data["inference_time"].append(inference_time)
    metrics_data["fps"].append(fps)
    metrics_data["cpu_usage"].append(cpu_usage)
    metrics_data["gpu_usage"].append(gpu_utilization)

    # Display metrics in the console
    print(f"Frame: {frame_count}")
    print(f"Preprocessing Time: {preprocess_end - preprocess_start:.4f}s")
    print(f"Inference Time: {inference_time:.4f}s")
    print(f"FPS: {fps:.2f}")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"GPU Usage: {gpu_utilization}%\n")

    # Draw bounding boxes
    for *box, conf, cls in detections:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Concatenate and display frames
    processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    combined_frame = np.hstack((frame, processed_frame))
    cv2.imshow('Performance Metrics', combined_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate average metrics
average_inference_time = total_inference_time / frame_count
print(f"Average Inference Time: {average_inference_time:.4f}s")
print(f"Average FPS: {frame_count / elapsed_time:.2f}")

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('model_ev5_metrics.csv', index=False)
print("Metrics saved to 'model_metrics.csv'")

# Cleanup
cap.release()
cv2.destroyAllWindows()
