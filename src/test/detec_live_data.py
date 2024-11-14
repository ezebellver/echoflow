import os
import cv2
import numpy as np
import torch
import random
import time
from src.preprocessing.image_processor import ImageProcessor  # Import your ImageProcessor class

# Define image dimensions
img_width, img_height = 400, 224

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='C:\\Users\\bellv\\OneDrive\\Escritorio\\Uni\\5toA2doC\\Tesis\\EchoFlow\\yolov5\\runs\\train\\my_yolov5_model\\weights\\best.pt')  # change 'best.pt' to the path of your weights file

# Set the model to evaluation mode
model.eval()

# Folder containing random images
image_folder = 'C:\\Users\\bellv\\EchoFlowDataset\\DatasetYOLO\\annotated\\train\\images'

# Instantiate ImageProcessor and add preprocessing steps
image_processor = ImageProcessor()

kernel_size = (7, 7)

# Add preprocessing steps
image_processor.add_processing_step(ImageProcessor.convert_to_grayscale)
image_processor.add_processing_step(
    lambda img: ImageProcessor.denoise_image(img, kernel_size=kernel_size))
image_processor.add_processing_step(
    lambda img: ImageProcessor.resize_image(img, (img_width, img_height)))
image_processor.add_processing_step(ImageProcessor.edge_detection)
image_processor.add_processing_step(ImageProcessor.normalize_image_255)
image_processor.add_processing_step(image_processor.extract_sift_features)
image_processor.add_processing_step(image_processor.suppress_non_roi)
image_processor.add_processing_step(ImageProcessor.normalize_image_255)

# Function to load and resize a random image from the folder
def load_random_image(image_folder):
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    random_image_path = os.path.join(image_folder, random.choice(image_files))
    random_image = cv2.imread(random_image_path)
    if random_image is not None:
        random_image = cv2.resize(random_image, (img_width, img_height))
    return random_image

# Function to perform detection on an image using YOLOv5
def detect_objects(image):
    # Perform inference on the image
    results = model(image)

    # Extract detections
    detections = results.xyxy[0].cpu().numpy()  # xyxy format: xmin, ymin, xmax, ymax, confidence, class
    return detections

# Function to display a random image from the folder with detected objects
def display_random_image(image_folder):
    random_image = load_random_image(image_folder)
    if random_image is not None:
        # Perform detection on the random image
        detections = detect_objects(random_image)

        # Draw bounding boxes and labels on the random image
        for *box, conf, cls in detections:
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(random_image, pt1, pt2, (0, 255, 0), 2)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.putText(random_image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the random image with detections for 2 seconds
        cv2.imshow('Random Image with Detection', random_image)
        cv2.waitKey(2000)
        cv2.destroyWindow('Random Image with Detection')

# Main loop to capture and process video frames
cap = cv2.VideoCapture(0)

# Set camera properties for better quality (adjust these as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 24)

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    if current_time - start_time >= 10:
        display_random_image(image_folder)
        start_time = current_time

    else:
        processed_frame = image_processor.process_image(frame)
        detections = detect_objects(frame)

        for *box, conf, cls in detections:
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        combined_frame = np.hstack((frame, processed_frame))

        cv2.imshow('YOLOv5 Object Detection - Original and Processed', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
