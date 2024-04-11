import cv2
import numpy as np


class ImageProcessor:
    def __init__(self):
        self.processing_steps = []

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
        return image.astype(np.float32) / 255.0

    @staticmethod
    def denoise_image(image, kernel_size=(5, 5)):
        return cv2.GaussianBlur(image, kernel_size, 0)

    @staticmethod
    def edge_detection(image, threshold1=20, threshold2=80):
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
