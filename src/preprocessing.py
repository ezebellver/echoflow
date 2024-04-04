import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def resize_image(self, target_size):
        """
        Resize input image to the target size.

        Parameters:
        - target_size (tuple): Target size of the output image in the format (width, height).

        Returns:
        - resized_image (numpy.ndarray): Resized image as a NumPy array.
        """
        self.image = cv2.resize(self.image, target_size)
        return self.image

    def convert_to_grayscale(self):
        """
        Convert input image to grayscale.

        Returns:
        - grayscale_image (numpy.ndarray): Grayscale image as a NumPy array.
        """
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image

    def normalize_image(self):
        """
        Normalize pixel values of the input image.

        Returns:
        - normalized_image (numpy.ndarray): Normalized image as a NumPy array.
        """
        self.image = self.image.astype(np.float32) / 255.0
        return self.image

    def denoise_image(self, kernel_size=(5, 5)):
        """
        Apply Gaussian denoising filter to the input image.

        Parameters:
        - kernel_size (tuple): Size of the Gaussian kernel in the format (width, height).
                               Default is (5, 5).

        Returns:
        - denoised_image (numpy.ndarray): Denoised image as a NumPy array.
        """
        self.image = cv2.GaussianBlur(self.image, kernel_size, 0)
        return self.image

    def edge_detection(self):
        """
        Apply non-maximum suppression to thin the edges in the input image.

        Returns:
        - suppressed_image (numpy.ndarray): Image with edges thinned using non-maximum suppression.
        """
        self.image = cv2.Canny(self.image, 50, 150)
        return self.image


def preprocess_image(image, target_size, visualize_steps=False):
    """
    Preprocess input image by resizing, converting to grayscale, denoising,
    applying edge detection, and normalizing.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.
    - target_size (tuple): Target size of the output image in the format (width, height).
    - visualize_steps (bool): Whether to visualize intermediate steps. Default is False.

    Returns:
    - preprocessed_image (numpy.ndarray): Preprocessed image as a NumPy array.
    """
    # Create an instance of ImageProcessor with input image
    image_processor = ImageProcessor(image)

    # Convert image to grayscale
    image_processor.convert_to_grayscale()

    # Visualize grayscale image
    if visualize_steps:
        plt.imshow(image_processor.image, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        plt.show()

    # Resize the image
    image_processor.resize_image(target_size)

    # Visualize resized image
    if visualize_steps:
        plt.imshow(image_processor.image, cmap='gray')
        plt.title("Resized Image")
        plt.axis('off')
        plt.show()

    # Denoise the image
    image_processor.denoise_image()

    # Visualize denoised image
    if visualize_steps:
        plt.imshow(image_processor.image, cmap='gray')
        plt.title("Denoised Image")
        plt.axis('off')
        plt.show()

    # Apply edge detection
    image_processor.edge_detection()

    # Visualize edge image
    if visualize_steps:
        plt.imshow(image_processor.image, cmap='gray')
        plt.title("Edge Image")
        plt.axis('off')
        plt.show()

    # Normalize pixel values
    image_processor.normalize_image()

    # Visualize preprocessed image
    if visualize_steps:
        plt.imshow(image_processor.image, cmap='gray')
        plt.title("Preprocessed Image")
        plt.axis('off')
        plt.show()

    return image_processor.image


if __name__ == "__main__":
    # Example usage
    # Load input image
    input_image = cv2.imread("..\\data\\sign_example.jpg")

    # Define target size for preprocessing
    target_size = (224, 224)

    # Preprocess input image
    preprocessed_image = preprocess_image(input_image, target_size, visualize_steps=True)
