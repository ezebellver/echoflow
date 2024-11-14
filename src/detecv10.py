import cv2
import numpy as np
from ultralytics import YOLO
from src.preprocessing.image_processor import ImageProcessor  # Import your ImageProcessor class

# Define image dimensions
img_width, img_height = 400, 224

# Load the YOLOv8 model
model = YOLO('C:\\Users\\bellv\\OneDrive\\Escritorio\\Uni\\5toA2doC\\Tesis\\EchoFlow\\runs\\detect\\train4\\weights\\best.pt')

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
image_processor.add_processing_step(
    lambda img: ImageProcessor.resize_image(img, (img_width, img_height)))
image_processor.add_processing_step(ImageProcessor.normalize_image_255)

# Define a function to preprocess the frame
def preprocess_frame(frame):
    # Process the frame
    processed_frame = image_processor.process_image(frame)
    return processed_frame

# Define a function to process the frame and perform detection
def detect(frame):
    # Perform inference on the frame
    results = model.predict(frame, imgsz=(img_width, img_height))
    print("results: " + str(results))

    # Extract detections
    if results and results[0].boxes.xyxy is not None:
        detections = results[0].boxes.data.cpu().numpy()  # xyxy format: xmin, ymin, xmax, ymax, confidence, class
        return detections
    else:
        return []

# Main loop to capture and process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (for internal processing, but not displaying)
    processed_frame = preprocess_frame(frame)

    # Perform detection on the original frame
    detections = detect(frame)

    # Debug print to check the structure of detections
    print("Detections:", detections)

    # Draw bounding boxes and labels on the original frame
    for det in detections:
        if len(det) == 6:
            *box, conf, cls = det
            cls = int(cls)
            if cls in model.names:
                label = f'{model.names[cls]} {conf:.2f}'
            else:
                label = f'Class {cls} {conf:.2f}'
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display only the original frame with detections
    cv2.imshow('YOLOv10 Object Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
