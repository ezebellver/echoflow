import cv2
from src.preprocessing.video_processor import VideoProcessor

# Create an instance of ImageProcessor
processor = VideoProcessor()

target_size = (244, 244)

# Add processing steps
processor.add_processing_step(lambda img: processor.resize_image(img, target_size))
processor.add_processing_step(processor.convert_to_grayscale)
processor.add_processing_step(processor.denoise_image)
processor.add_processing_step(processor.edge_detection)
processor.add_processing_step(processor.contour_detection)
processor.add_processing_step(processor.edge_subtraction)
processor.add_processing_step(processor.normalize_image)


"""
    Preprocess input video stream by resizing, converting to grayscale, denoising,
    applying contour detection, and normalizing using custom ImageProcessor.

    Parameters:
    - video_stream (cv2.VideoCapture): Video stream object.
    """
def preprocess_video_stream(video_stream):

    while True:
        # Read frame from the video stream
        ret, frame = video_stream.read()
        if not ret:
            break

        # Preprocess the frame
        preprocessed_frame = processor(frame)

        # Show the preprocessed frame
        cv2.imshow('Preprocessed Video Stream', preprocessed_frame)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close all OpenCV windows
    video_stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    # Define target size for preprocessing
    target_size = (244, 244)  # Adjust the target size as needed

    # Create a video stream object (e.g., from webcam)
    video_stream = cv2.VideoCapture(0)  # Assuming camera index 0 is available

    # Preprocess video stream
    preprocess_video_stream(video_stream)
