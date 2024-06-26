import os
import shutil

# Define the source and destination directories
source_dir = 'C:\\Users\\bellv\\EchoFlowDataset\\asl\\fingerspelling5\\dataset5\\A'
destination_dir = 'C:\\Users\\bellv\\EchoFlowDataset\\asl\\fingerspelling5\\dataset5\\A_color'

# Function to copy files
def copy_files(source, destination):
    for root, dirs, files in os.walk(source):
        # Construct the corresponding destination directory
        relative_path = os.path.relpath(root, source)
        dest_dir = os.path.join(destination, relative_path)
        os.makedirs(dest_dir, exist_ok=True)

        # Copy files that don't start with 'depth'
        for file in files:
            if not file.startswith('depth'):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(dest_dir, file)
                shutil.copy2(source_file, destination_file)

# Run the function
copy_files(source_dir, destination_dir)
