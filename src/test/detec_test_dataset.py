import os
import cv2
import numpy as np
import torch
from src.preprocessing.image_processor import ImageProcessor  # Import your ImageProcessor class

# Define image dimensions
img_width, img_height = 400, 224

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='C:\\Users\\bellv\\OneDrive\\Escritorio\\Uni\\5toA2doC\\Tesis\\EchoFlow\\yolov5\\runs\\train\\my_yolov5_model\\weights\\best.pt')  # change 'best.pt' to the path of your weights file

# Set the model to evaluation mode
model.eval()

# Define the path to the image dataset
# dataset_path = 'C:\\Users\\bellv\\EchoFlowDataset\\DatasetYOLO\\annotated\\valid\\images'
dataset_path = 'C:\\Users\\bellv\\EchoFlowDataset\\DatasetYOLO\\Ezi AI\\B'

# Define a function to preprocess the frame
def preprocess_frame(frame):
    kernel_size = (19, 19)

    # Instantiate ImageProcessor and add preprocessing steps
    image_processor = ImageProcessor()

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
#
    ## Process the frame
    processed_frame = image_processor.process_image(frame)
#
    #return processed_frame

    mean = 180
    std_dev = 10
    inc = 5
    flag1, flag2 = False, False

    while True:
        # print(len(image_processor.keypoints))
        if flag1 and flag2:
            std_dev += inc
            flag1, flag2 = False, False
        if len(image_processor.keypoints) < mean - std_dev and kernel_size != (1, 1):
            kernel_size = tuple(np.subtract(kernel_size, (2, 2)))
            processed_frame = image_processor.process_image(frame)
            flag1 = True
        elif len(image_processor.keypoints) > mean + std_dev:
            kernel_size = tuple(np.add(kernel_size, (2, 2)))
            processed_frame = image_processor.process_image(frame)
            flag2 = True
        else:
            break

        return processed_frame


# Define a function to process the frame and perform detection
def detect(frame):
    # Perform inference on the frame
    results = model(frame)

    # Extract detections
    detections = results.xyxy[0].cpu().numpy()  # xyxy format: xmin, ymin, xmax, ymax, confidence, class
    return detections

# Loop through each image in the dataset directory
for image_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_name)

    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        continue

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
    # combined_frame = np.hstack((frame, processed_frame))
    combined_frame = processed_frame

    # Display the combined frame
    cv2.imshow('YOLOv5 Object Detection - Original and Processed', combined_frame)

    # Wait for a key press to display each image
    cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
