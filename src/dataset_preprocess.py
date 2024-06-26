import os
import cv2
import numpy as np

from src.preprocessing.image_processor import ImageProcessor  # Import your ImageProcessor class

# Define image dimensions
img_width, img_height = 400, 224


def preprocess_and_save_images(src_dir, dst_dir, start_number=1):
    # Check if the destination directory exists, if not, create it
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Iterate over each folder in the source directory
    for folder_name in os.listdir(src_dir):
        src_folder_path = os.path.join(src_dir, folder_name)
        dst_folder_path = os.path.join(dst_dir, folder_name)

        # Only process directories
        if os.path.isdir(src_folder_path):
            # Create destination folder if it doesn't exist
            if not os.path.exists(dst_folder_path):
                os.makedirs(dst_folder_path)

            # Initialize the numbering for the current folder
            image_number = start_number

            # Iterate over each image in the folder
            for image_name in os.listdir(src_folder_path):
                image_path = os.path.join(src_folder_path, image_name)
                if os.path.isfile(image_path):
                    # Load the image
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Image at path {image_path} could not be loaded. Skipping...")
                        continue

                    kernel_size = (19, 19)

                    # Instantiate ImageProcessor and add preprocessing steps
                    image_processor = ImageProcessor()

                    # Add preprocessing steps
                    image_processor.add_processing_step(ImageProcessor.convert_to_grayscale)
                    image_processor.add_processing_step(
                        lambda img: ImageProcessor.denoise_image(img, kernel_size=kernel_size))
                    image_processor.add_processing_step(
                        lambda img: ImageProcessor.resize_image(img, (img_width, img_height)))
                    # image_processor.add_processing_step(ImageProcessor.normalize_image)
                    image_processor.add_processing_step(ImageProcessor.edge_detection)
                    # image_processor.add_processing_step(ImageProcessor.contour_detection)
                    image_processor.add_processing_step(ImageProcessor.normalize_image_255)
                    image_processor.add_processing_step(image_processor.extract_sift_features)
                    image_processor.add_processing_step(image_processor.suppress_non_roi)
                    image_processor.add_processing_step(ImageProcessor.normalize_image_255)

                    # Process the image
                    processed_image = image_processor.process_image(image)

                    mean = 180
                    std_dev = 10
                    inc = 5
                    flag1, flag2 = False, False

                    while True:
                        # print(len(image_processor.keypoints))
                        if flag1 and flag2:
                            std_dev += inc
                            flag1, flag2 = False, False
                        if len(image_processor.keypoints) < mean - std_dev and kernel_size != (1, 1):
                            kernel_size = tuple(np.subtract(kernel_size, (2, 2)))
                            processed_image = image_processor.process_image(image)
                            flag1 = True
                        elif len(image_processor.keypoints) > mean + std_dev:
                            kernel_size = tuple(np.add(kernel_size, (2, 2)))
                            processed_image = image_processor.process_image(image)
                            flag2 = True
                        else:
                            break

                    # Create new image name
                    new_image_name = f"{folder_name}_{image_number}.jpg"
                    processed_image_path = os.path.join(dst_folder_path, new_image_name)

                    # Save the processed image to the destination folder
                    cv2.imwrite(processed_image_path, processed_image)
                    print(f"Processed image saved at: {processed_image_path}")

                    # Increment the image number for the next image
                    image_number += 1


# Example usage:
# Replace 'src_directory_path' with the path to the parent source directory
# Replace 'dst_directory_path' with the path to the parent destination directory
preprocess_and_save_images('C:\\Users\\bellv\\EchoFlowDataset\\DatasetPropio\\3ra ronda', 'C:\\Users\\bellv\\EchoFlowDataset\\DatasetYOLO\\preprocesed')