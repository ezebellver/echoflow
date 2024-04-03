import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image(image, target_size):
    """
    Resize input image to the target size.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.
    - target_size (tuple): Target size of the output image in the format (width, height).

    Returns:
    - resized_image (numpy.ndarray): Resized image as a NumPy array.
    """
    # Resize the image using OpenCV
    resized_image = cv2.resize(image, target_size)

    return resized_image


def convert_to_grayscale(image):
    """
    Convert input image to grayscale.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.

    Returns:
    - grayscale_image (numpy.ndarray): Grayscale image as a NumPy array.
    """
    # Convert the input image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grayscale_image


def normalize_image(image):
    """
    Normalize pixel values of the input image.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.

    Returns:
    - normalized_image (numpy.ndarray): Normalized image as a NumPy array.
    """
    # Convert image to floating point format
    normalized_image = image.astype(np.float32)

    # Normalize pixel values to the range [0, 1]
    normalized_image /= 255.0

    return normalized_image


def denoise_image(image, kernel_size=(5, 5)):
    """
    Apply Gaussian denoising filter to the input image.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.
    - kernel_size (tuple): Size of the Gaussian kernel in the format (width, height).
                           Default is (5, 5).

    Returns:
    - denoised_image (numpy.ndarray): Denoised image as a NumPy array.
    """
    # Apply Gaussian blur to denoise the image
    denoised_image = cv2.GaussianBlur(image, kernel_size, 0)

    return denoised_image


def edge_detection(image):
    """
    Apply non-maximum suppression to thin the edges in the input image.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.

    Returns:
    - suppressed_image (numpy.ndarray): Image with edges thinned using non-maximum suppression.
    """
    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    return edges


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
    # Convert image to grayscale
    grayscale_image = convert_to_grayscale(image)

    # Visualize grayscale image
    if visualize_steps:
        plt.imshow(grayscale_image, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        plt.show()

    # Resize the image
    resized_image = resize_image(grayscale_image, target_size)

    # Visualize resized image
    if visualize_steps:
        plt.imshow(resized_image, cmap='gray')
        plt.title("Resized Image")
        plt.axis('off')
        plt.show()

    # Denoise the image
    denoised_image = denoise_image(resized_image)

    # Visualize denoised image
    if visualize_steps:
        plt.imshow(denoised_image, cmap='gray')
        plt.title("Denoised Image")
        plt.axis('off')
        plt.show()

    # Apply edge detection
    edge_image = edge_detection(denoised_image)

    # Visualize edge image
    if visualize_steps:
        plt.imshow(edge_image, cmap='gray')
        plt.title("Edge Image")
        plt.axis('off')
        plt.show()

    # Normalize pixel values
    preprocessed_image = normalize_image(edge_image)

    # Visualize preprocessed image
    if visualize_steps:
        plt.imshow(preprocessed_image, cmap='gray')
        plt.title("Preprocessed Image")
        plt.axis('off')
        plt.show()

    return preprocessed_image


if __name__ == "__main__":
    # Example usage
    # Load input image
    input_image = cv2.imread("..\\data\\sign_example.jpg")

    # Define target size for preprocessing
    target_size = (224, 224)

    # Preprocess input image
    preprocessed_image = preprocess_image(input_image, target_size, visualize_steps=True)
