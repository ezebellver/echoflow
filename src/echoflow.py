import cv2
import numpy as np
from preprocessing import preprocess_image
from feature_extraction import extract_sift_features, suppress_non_roi

def mark_keypoints(image, keypoints):
    """
    Mark keypoints on the input image with red circles.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.
    - keypoints (list): List of keypoints detected in the image.

    Returns:
    - marked_image (numpy.ndarray): Image with keypoints marked by red circles.
    """
    # Convert grayscale image to color
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

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


if __name__ == "__main__":
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
    cv2.destroyAllWindows()
