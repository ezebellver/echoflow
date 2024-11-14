import os


def rename_files_in_folders(base_path):
    # Get all folder names in the base path
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    for folder in folders:
        folder_path = os.path.join(base_path, folder)

        # Get all files in the folder
        files = os.listdir(folder_path)

        # Rename each file
        for i, file_name in enumerate(files):
            # Get the file extension
            file_extension = os.path.splitext(file_name)[1]

            # Create the new file name
            new_name = f"{folder}_{i + 1}{file_extension}"

            # Get the full path to the old and new files
            old_file = os.path.join(folder_path, file_name)
            new_file = os.path.join(folder_path, new_name)

            # Rename the file
            os.rename(old_file, new_file)


# Set the base path to your dataset directory
base_path = "C:\\Users\\bellv\\EchoFlowDataset\\DatasetPropio\\Nueva carpeta"

# Call the function to rename the files
rename_files_in_folders(base_path)
