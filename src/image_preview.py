import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.image_processor import ImageProcessor  # Import your ImageProcessor class

# Define image dimensions
img_width, img_height = 400, 224
kernel_size = (19, 19)

# Provide the path to the image you want to process
# image_path = 'C:\\Users\\bellv\\EchoFlowDataset\\fingerspelling\\photos\\F\\F.png'
# image_path = ('C:\\Users\\bellv\\EchoFlowDataset\\asl\\fingerspelling5\\dataset5\\A_color\\g\\color_6_0002.png')
image_path = 'C:\\Users\\bellv\\EchoFlowDataset\\DatasetPropio\\3ra ronda\\O\\IMG_1819 Grande.jpeg'

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Image at path {image_path} could not be loaded. Check the path and try again.")

# Instantiate ImageProcessor and add preprocessing steps
image_processor = ImageProcessor()

# Add preprocessing steps
image_processor.add_processing_step(ImageProcessor.convert_to_grayscale)
image_processor.add_processing_step(lambda img: ImageProcessor.denoise_image(img, kernel_size=kernel_size))
image_processor.add_processing_step(lambda img: ImageProcessor.resize_image(img, (img_width, img_height)))
# image_processor.add_processing_step(ImageProcessor.normalize_image)
image_processor.add_processing_step(ImageProcessor.edge_detection)
# image_processor.add_processing_step(ImageProcessor.contour_detection)
image_processor.add_processing_step(ImageProcessor.normalize_image_255)
image_processor.add_processing_step(image_processor.extract_sift_features)
image_processor.add_processing_step(image_processor.suppress_non_roi)
image_processor.add_processing_step(ImageProcessor.normalize_image_255)

# Process the image
processed_image = image_processor.process_image(image)

while True:
    if len(image_processor.keypoints) < 165:
        kernel_size = tuple(np.subtract(kernel_size, (2, 2)))
        processed_image = image_processor.process_image(image)
    elif len(image_processor.keypoints) > 190:
        kernel_size = tuple(np.add(kernel_size, (2, 2)))
        processed_image = image_processor.process_image(image)
    else:
        break



# Display original, preprocessed, and ROI images side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image', fontsize=14)
axes[0].axis('off')
axes[1].imshow(processed_image, cmap='gray')  # Display preprocessed image in grayscale
axes[1].set_title('Preprocessed Image', fontsize=14)
axes[1].axis('off')

# Turn off axis ticks and tick labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for better appearance
plt.subplots_adjust(wspace=0.05, top=0.85)  # Adjust spacing between subplots and top margin
plt.show()

# Save the image as preprocessed_image.png
# fig.savefig('preprocessed_image.png', bbox_inches='tight')
