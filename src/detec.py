import os
import cv2
import numpy as np
import torch
from src.preprocessing.image_processor import ImageProcessor  # Import your ImageProcessor class

# Define image dimensions
img_width, img_height = 400, 224

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='C:\\Users\\bellv\\OneDrive\\Escritorio\\Uni\\5toA2doC\\Tesis\\EchoFlow\\runs\\train\\my_yolov5_model4\\weights\\best.pt')  # change 'best.pt' to the path of your weights file

# Set the model to evaluation mode
model.eval()

# Initialize the video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Set camera properties for better quality (adjust these as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 24)

# Instantiate ImageProcessor and add preprocessing steps
image_processor = ImageProcessor()

kernel_size = (7, 7)

# Add preprocessing steps
image_processor.add_processing_step(ImageProcessor.convert_to_grayscale)
# image_processor.add_processing_step(
#     lambda img: ImageProcessor.denoise_image(img, kernel_size=kernel_size))
image_processor.add_processing_step(
    lambda img: ImageProcessor.resize_image(img, (img_width, img_height)))
# image_processor.add_processing_step(ImageProcessor.edge_detection)
# image_processor.add_processing_step(ImageProcessor.normalize_image_255)
# image_processor.add_processing_step(image_processor.extract_sift_features)
# image_processor.add_processing_step(image_processor.suppress_non_roi)
image_processor.add_processing_step(ImageProcessor.normalize_image_255)

# Define a function to preprocess the frame
def preprocess_frame(frame):
    # Process the frame
    processed_frame = image_processor.process_image(frame)

    return processed_frame

# Define a function to process the frame and perform detection
def detect(frame):
    # Perform inference on the frame
    results = model(frame)

    # Extract detections
    detections = results.xyxy[0].cpu().numpy()  # xyxy format: xmin, ymin, xmax, ymax, confidence, class
    return detections

# Main loop to capture and process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Perform detection on the original frame
    detections = detect(frame)

    # Draw bounding boxes and labels on the original frame
    for *box, conf, cls in detections:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resize the processed frame to match the original frame size
    processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))

    # Convert the processed frame to 3 channels
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

    # Concatenate original and processed frames for comparison
    combined_frame = np.hstack((frame, processed_frame))

    # Display the combined frame
    cv2.imshow('YOLOv5 Object Detection - Original and Processed', combined_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
