import os
import cv2
import matplotlib.pyplot as plt


def show_image(image_path):
    # Read the image from the file
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded. Check the path and try again.")

    # Check if the image is normalized (assuming normalized images are in range [0, 1])
    if image.max() <= 1.0:
        image = (image * 255).astype('uint8')

    # Convert the image from BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image_rgb, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')
    plt.title('Processed Image')
    plt.show()


# Example usage:
# Replace 'processed_image_path' with the path to the processed image
show_image('C:\\Users\\bellv\\EchoFlowDataset\\DatasetYOLO\\Preprocesed\\N\\N_10.jpg')
