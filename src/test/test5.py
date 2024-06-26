import os
import string


def create_folders_in_directory(directory):
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Iterate over each letter in the alphabet and create a folder for each
    for letter in string.ascii_uppercase:
        folder_path = os.path.join(directory, letter)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")


# Example usage:
# Replace 'your_directory_path' with the path where you want to create the folders
create_folders_in_directory('C:\\Users\\bellv\\EchoFlowDataset\\DatasetYOLO\\class\\blank')
