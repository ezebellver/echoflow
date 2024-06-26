import os
import zipfile

def extract_first_30_files(zip_folder_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the zip folder
    with zipfile.ZipFile(zip_folder_path, 'r') as zip_ref:
        # List all files in the zip folder
        zip_file_list = zip_ref.namelist()

        # Extract the first 30 files from each subfolder inside the train folder
        extracted_count = 0
        print("Extracting")
        for file_path in zip_file_list:
            # Extract only files (ignore directories)
            print(file_path)
            if not file_path.endswith('/'):
                # Check if the file is within the train folder
                if file_path.startswith('Dataset/train/') and '/' in file_path[len('train/'):]:
                    print(file_path)
                    # Extract subfolder name
                    subfolder = os.path.dirname(file_path[len('train/'):])
                    # Create a subfolder in the output folder if it doesn't exist
                    subfolder_output_path = os.path.join(output_folder, subfolder)
                    if not os.path.exists(subfolder_output_path):
                        os.makedirs(subfolder_output_path)
                    # Extract only the first 30 files from each subfolder
                    if subfolder not in extracted_folders or extracted_folders[subfolder] < 30:
                        print(subfolder)
                        extracted_folders.setdefault(subfolder, 0)
                        extracted_count += 1
                        extracted_folders[subfolder] += 1
                        # Extract the file to the subfolder in the output folder
                        zip_ref.extract(file_path, subfolder_output_path)

# Input zip folder path
zip_folder_path = '..\\data\\dataset\\number_alphabet\\Dataset.zip'

# Output folder path
output_folder = '..\\data\\dataset\\number_alphabet\\Dataset'

# Dictionary to store the number of extracted files from each subfolder
extracted_folders = {}

# Extract the first 30 files from each subfolder inside the train folder
extract_first_30_files(zip_folder_path, output_folder)
