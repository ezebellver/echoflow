import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageProcessor:
    def __init__(self):
        self.processing_steps = []
        self.keypoints = None

    def add_processing_step(self, method):
        """
        Add a processing step to the image processor.

        Parameters:
        - method (callable): The method to be added as a processing step.
        """
        self.processing_steps.append(method)

    def process_image(self, image):
        """
        Apply all processing steps to the image.

        Parameters:
        - image (numpy.ndarray): Input image as a NumPy array.

        Returns:
        - processed_image (numpy.ndarray): Processed image as a NumPy array.
        """
        processed_image = image.copy()
        for step in self.processing_steps:
            processed_image = step(processed_image)
        return processed_image

    def __call__(self, image):
        """
        Apply all processing steps to the image using the callable interface.

        Parameters:
        - image (numpy.ndarray): Input image as a NumPy array.

        Returns:
        - processed_image (numpy.ndarray): Processed image as a NumPy array.
        """
        return self.process_image(image)

    # Decorator for adding processing steps
    def add_step(self, method):
        """
        Decorator to add a processing step to the image processor.

        Parameters:
        - method (callable): The method to be added as a processing step.
        """
        self.add_processing_step(method)
        return method

    # Existing image processing methods
    @staticmethod
    def resize_image(image, target_size):
        return cv2.resize(image, target_size)

    @staticmethod
    def convert_to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def normalize_image(image):
        # Check if the image values are already within the desired range
#        image = np.reshape(image, (224, 224, 1))
        if np.max(image) <= 1.0:
            return image.astype(np.float32)
        else:
            return image.astype(np.float32) / 255.0

    @staticmethod
    def normalize_image_255(image):
        # Normalizes the image to the range [0, 255]
        if np.max(image) <= 1.0:
            return (image * 255).astype(np.uint8)
        else:
            return image.astype(np.uint8)

    @staticmethod
    def denoise_image(image, kernel_size=(5, 5)):
        return cv2.GaussianBlur(image, kernel_size, 0)

    @staticmethod
    def edge_detection(image, threshold1=20, threshold2=80):
        # Convert image to uint8 if not already in that format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return cv2.Canny(image, threshold1, threshold2)

    @staticmethod
    def contour_detection(image):
        contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = np.zeros_like(image)
        image_with_contours = cv2.drawContours(image_with_contours, contours, -1, (255, 255, 255), 1)
        return image_with_contours

    @staticmethod
    def median_filter(image, kernel_size=3):
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def open_operation(image, kernel_size=(3, 3)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # New methods for SIFT features and ROI suppression
    def extract_sift_features(self, image):
        """
        Extract SIFT (Scale-Invariant Feature Transform) features from the input image.

        Parameters:
        - image (numpy.ndarray): Input image as a NumPy array.

        Returns:
        - image (numpy.ndarray): The same input image.
        """
        if image is None:
            raise ValueError("Input image is None")

        # Initialize SIFT detector with adjusted parameters
        sift = cv2.SIFT_create(
            nfeatures=0,
            nOctaveLayers=5,
            contrastThreshold=0.02,
            edgeThreshold=10,
            sigma=1.6
        )

        # Detect keypoints and compute descriptors
        self.keypoints, descriptors = sift.detectAndCompute(image, None)

        return image

    def suppress_non_roi(self, image, roi_radius_factor=2):
        """
        Suppress everything but the regions of interest (ROIs) in the input image.

        Parameters:
        - image (numpy.ndarray): Input image as a NumPy array.
        - roi_radius_factor (float): Factor by which to multiply the keypoint size to determine the ROI radius. Default is 1.5.

        Returns:
        - roi_image (numpy.ndarray): Image with only the ROIs visible.
        """
        if self.keypoints is None:
            raise ValueError("Keypoints have not been extracted. Run extract_sift_features first.")

        # Create an empty mask with the same dimensions as the input image
        mask = np.zeros_like(image, dtype=np.uint8)

        # Draw filled circles at the locations of keypoints (ROIs) on the mask
        for kp in self.keypoints:
            x, y = np.round(kp.pt).astype(int)
            radius = int(np.round(kp.size / 2 * roi_radius_factor))  # Increase the radius based on the factor
            cv2.circle(mask, (x, y), radius, 255, -1)  # Use 255 as the color for grayscale

        # Convert image to uint8
        image_uint8 = (image * 255).astype(np.uint8)

        # Perform bitwise AND operation to keep only the ROIs in the image
        roi_image = cv2.bitwise_and(image_uint8, mask)

        return roi_image

    def adaptive_sift_features(self, image, target_keypoints=300, std_dev=10):
        """
        Extract SIFT features with adaptive parameter adjustment to find a desired number of keypoints.

        Parameters:
        - image (numpy.ndarray): Input image as a NumPy array.
        - target_keypoints (int): Desired number of keypoints to find.
        - std_dev (int): Acceptable standard deviation around the target number of keypoints.

        Returns:
        - image (numpy.ndarray): The same input image.
        """
        if image is None:
            raise ValueError("Input image is None")

        # Initialize default SIFT parameters
        contrast_threshold = 0.02
        edge_threshold = 10

        while True:
            # Initialize SIFT detector with current parameters
            sift = cv2.SIFT_create(
                nfeatures=0,
                nOctaveLayers=5,
                contrastThreshold=contrast_threshold,
                edgeThreshold=edge_threshold,
                sigma=1.6
            )

            # Detect keypoints and compute descriptors
            self.keypoints, descriptors = sift.detectAndCompute(image, None)

            # Check if the number of keypoints is within the desired range
            num_keypoints = len(self.keypoints)
            print("Number of keypoints: ", num_keypoints)
            if abs(num_keypoints - target_keypoints) <= std_dev:
                break

            # Adjust parameters if the number of keypoints is not within the desired range
            if num_keypoints < target_keypoints:
                contrast_threshold *= 0.8
                edge_threshold *= 0.8
            else:
                contrast_threshold *= 1.2
                edge_threshold *= 1.2

            # Avoid infinite loop by setting reasonable limits on parameters
            if contrast_threshold < 0.0001 or edge_threshold < 1:
                break

        return image
