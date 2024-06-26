import cv2
import keras
import numpy as np
import datetime
from src.preprocessing.image_processor import ImageProcessor

# Load the saved model
model = keras.models.load_model('../models/bsl_fingerspelling_cnn_model.keras', compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Define image dimensions
img_width, img_height = 224, 224

image_processor = ImageProcessor()

# Add preprocessing steps
image_processor.add_processing_step(ImageProcessor.convert_to_grayscale)
image_processor.add_processing_step(lambda img: ImageProcessor.denoise_image(img, kernel_size=(15, 15)))
image_processor.add_processing_step(lambda img: ImageProcessor.resize_image(img, (img_width, img_height)))
image_processor.add_processing_step(ImageProcessor.normalize_image)
image_processor.add_processing_step(ImageProcessor.edge_detection)
image_processor.add_processing_step(ImageProcessor.contour_detection)
image_processor.add_processing_step(ImageProcessor.normalize_image)
image_processor.add_processing_step(image_processor.extract_sift_features)
image_processor.add_processing_step(image_processor.suppress_non_roi)
image_processor.add_processing_step(ImageProcessor.normalize_image)


# Function to predict and log video frames
def predict_video(video_path, model, image_processor, threshold=0.3, output_log_above_threshold='predictions_above_threshold.log', output_log_all_predictions='all_predictions.log'):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    predictions = []
    prev_prediction = None
    prev_certainty = 0
    prev_time = 0

    with open(output_log_above_threshold, 'w') as log_file_above_threshold, open(output_log_all_predictions, 'w') as log_file_all_predictions:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the timestamp of the current frame
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # in seconds

            # Process the frame
            processed_frame = image_processor.process_image(frame)
            processed_frame = np.expand_dims(processed_frame, axis=0)

            # Make predictions
            preds = model.predict(processed_frame)
            predicted_class = np.argmax(preds)
            certainty = np.max(preds)

            # Log the prediction if it changes and if certainty is above threshold
            if predicted_class != prev_prediction:
                log_file_all_predictions.write(f"[{predicted_class}, Certainty: {certainty:.2f}, Time: {str(datetime.timedelta(seconds=timestamp))}]\n")
                prev_prediction = predicted_class

                if certainty > threshold:
                    log_file_above_threshold.write(f"[{predicted_class}, Certainty: {certainty:.2f}, Time: {str(datetime.timedelta(seconds=timestamp))}]\n")

    cap.release()


video_path = "C:\\Users\\bellv\\EchoFlowDataset\\fingerspelling\\Finger-spelling.mp4"
predict_video(video_path, model, image_processor, threshold=0.3, output_log_above_threshold='predictions_above_threshold.log', output_log_all_predictions='all_predictions.log')
