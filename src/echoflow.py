import cv2
import numpy as np
from image_processor import preprocess_image
from feature_extraction import extract_sift_features, suppress_non_roi
from src.video_preprocessing import VideoProcessor


def mark_keypoints(image, keypoints):
    """
    Mark keypoints on the input image with red circles.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.
    - keypoints (list): List of keypoints detected in the image.

    Returns:
    - marked_image (numpy.ndarray): Image with keypoints marked by red circles.
    """
    # Check if the image is already in BGR format (3 channels)
    if len(image.shape) == 2:  # If the image is grayscale
        # Convert grayscale image to color
        colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:  # If the image has one channel
        # Convert single-channel grayscale image to color
        colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # Image is already in BGR format, no need for conversion
        colored_image = image

    # Make a copy of the colored image
    marked_image = colored_image.copy()

    # Draw red circles at the locations of keypoints
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(marked_image, (int(x), int(y)), 5, (0, 0, 255), 1)

    return marked_image



def echoflow(image_path):
    """
    Run preprocessing, feature extraction, and ROI suppression on the input image.

    Parameters:
    - image_path (str): Path to the input image file.

    Returns:
    - keypoints (list): List of keypoints detected in the ROI.
    - descriptors (numpy.ndarray): Descriptors corresponding to the detected keypoints.
    - roi_image (numpy.ndarray): Image with only the ROI visible.
    - marked_image (numpy.ndarray): Image with keypoints marked by red circles.
    """
    # Load input image
    input_image = cv2.imread(image_path)

    # Preprocess input image
    preprocessed_image = preprocess_image(input_image, target_size=(224, 224))

    # Extract SIFT features
    keypoints, descriptors = extract_sift_features(preprocessed_image)

    # Suppress everything but the ROIs
    roi_image = suppress_non_roi(preprocessed_image, keypoints)

    # Mark keypoints on the preprocessed image
    marked_image = mark_keypoints(preprocessed_image, keypoints)

    return keypoints, descriptors, roi_image, marked_image


def echoflow_video(video_path):
    """
    Run preprocessing, feature extraction, and ROI suppression on each frame of the input video.

    Parameters:
    - video_path (str): Path to the input video file.

    Returns:
    - all_keypoints (list): List of keypoints detected in each frame.
    - all_descriptors (list): List of descriptors corresponding to the detected keypoints in each frame.
    """
    # Create an instance of VideoProcessor with the input video path
    video_processor = VideoProcessor(video_path)

    # Process video to extract keypoints and descriptors for each frame
    all_keypoints = []
    all_descriptors = []
    for frame in video_processor.process_video():
        # Preprocess frame
        preprocessed_frame = frame#preprocess_image(frame, target_size=(224, 224))

        # Extract SIFT features
        keypoints, descriptors = extract_sift_features(preprocessed_frame)

        # Suppress everything but the ROIs
        roi_image = suppress_non_roi(preprocessed_frame, keypoints)

        # Mark keypoints on the preprocessed frame
        marked_image = mark_keypoints(preprocessed_frame, keypoints)

        # Append keypoints and descriptors to lists
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)

        # Display the ROI image
        cv2.imshow("ROI Image", roi_image)
        cv2.waitKey(1)

        # Display the marked image with keypoints
        cv2.imshow("Marked Image with Keypoints", marked_image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return all_keypoints, all_descriptors

if __name__ == "__main__":
    # Example usage
    video_path = "..\\data\\BF10l.mov"
    all_keypoints, all_descriptors = echoflow_video(video_path)

    # Print the number of frames processed
    print("Number of frames processed:", len(all_keypoints))


"""if __name__ == "__main__":
    # Example usage
    image_path = "..\\data\\sign_example.jpg"
    keypoints, descriptors, roi_image, marked_image = echoflow(image_path)

    # Print the number of keypoints detected
    print("Number of keypoints:", len(keypoints))

    # Display the ROI image
    cv2.imshow("ROI Image", roi_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display the marked image with keypoints
    cv2.imshow("Marked Image with Keypoints", marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
