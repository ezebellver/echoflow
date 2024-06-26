import cv2
import numpy as np


def extract_sift_features(image):
    """
    Extract SIFT (Scale-Invariant Feature Transform) features from the input image.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.

    Returns:
    - keypoints (list): List of keypoints detected in the image.
    - descriptors (numpy.ndarray): Descriptors corresponding to the detected keypoints.
    """
    if image is None:
        raise ValueError("Input image is None")

    # Ensure the input image has the correct data type (np.uint8)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Scale the input image to the range [0, 255]
    scaled_image = (image * 255).astype(np.uint8)

    # Convert the input image to grayscale if it has more than one channel
    if len(scaled_image.shape) > 2 and scaled_image.shape[2] != 1:
        gray_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = scaled_image

    # Initialize SIFT detector with adjusted parameters
    sift = cv2.SIFT_create(
        nfeatures=0,
        # The number of best features to retain. Default is 0, which means all detected features are retained.
        nOctaveLayers=5,  # The number of layers in each octave of the Gaussian pyramid. Default is 3.
        contrastThreshold=0.02,
        # The contrast threshold used to filter out weak features in low-contrast regions. Default is 0.04.
        edgeThreshold=10,  # The edge threshold used to filter out weak features in edge regions. Default is 10.
        sigma=1.6  # The standard deviation of the Gaussian blur applied to the input image. Default is 1.6.
    )

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    return keypoints, descriptors


def suppress_non_roi(image, keypoints, roi_radius_factor=2):
    """
    Suppress everything but the regions of interest (ROIs) in the input image.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.
    - keypoints (list): List of keypoints representing the ROIs.
    - roi_radius_factor (float): Factor by which to multiply the keypoint size to determine the ROI radius. Default is 1.5.

    Returns:
    - roi_image (numpy.ndarray): Image with only the ROIs visible.
    """
    # Create an empty mask with the same dimensions as the input image
    mask = np.zeros_like(image, dtype=np.uint8)

    # Draw filled circles at the locations of keypoints (ROIs) on the mask
    for kp in keypoints:
        x, y = np.round(kp.pt).astype(int)
        radius = int(np.round(kp.size / 2 * roi_radius_factor))  # Increase the radius based on the factor
        cv2.circle(mask, (x, y), radius, 255, -1)  # Use 255 as the color for grayscale

    # Convert image to uint8
    image_uint8 = (image * 255).astype(np.uint8)

    # Perform bitwise AND operation to keep only the ROIs in the image
    roi_image = cv2.bitwise_and(image_uint8, mask)

    return roi_image


#################################################### VIDEO #####################################################


def extract_sift_features_video(video_path):
    """
    Extract SIFT features from each frame of the input video.

    Parameters:
    - video_path (str): Path to the input video file.

    Returns:
    - all_keypoints (list): List of keypoints detected in each frame of the video.
    - all_descriptors (list): List of descriptors corresponding to the detected keypoints in each frame.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    all_keypoints = []
    all_descriptors = []

    while (cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract SIFT features from the grayscale frame
        keypoints, descriptors = extract_sift_features(gray_frame)

        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)

    cap.release()

    return all_keypoints, all_descriptors


def suppress_non_roi_video(video_frames, keypoints_list, roi_radius_factor=2):
    """
    Suppress everything but the regions of interest (ROIs) in each frame of the input video.

    Parameters:
    - video_frames (list): List of frames from the input video as NumPy arrays.
    - keypoints_list (list): List of lists, where each sublist contains keypoints representing the ROIs in the corresponding frame.
    - roi_radius_factor (float): Factor by which to multiply the keypoint size to determine the ROI radius. Default is 1.5.

    Returns:
    - roi_frames (list): List of frames with only the ROIs visible.
    """
    roi_frames = []

    for frame, keypoints in zip(video_frames, keypoints_list):
        # Create an empty mask with the same dimensions as the frame
        mask = np.zeros_like(frame, dtype=np.uint8)

        # Draw filled circles at the locations of keypoints (ROIs) on the mask
        for kp in keypoints:
            x, y = np.round(kp.pt).astype(int)
            radius = int(np.round(kp.size / 2 * roi_radius_factor))  # Increase the radius based on the factor
            cv2.circle(mask, (x, y), radius, 255, -1)  # Use 255 as the color for grayscale

        # Perform bitwise AND operation to keep only the ROIs in the frame
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

        roi_frames.append(roi_frame)

    return roi_frames
