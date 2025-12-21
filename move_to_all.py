
import os
import shutil

# Source and destination directories
data_collect_dir = "DataCollect"
dataset_all_dir = "Dataset/all"

# Ensure the destination directory exists
os.makedirs(dataset_all_dir, exist_ok=True)

# Iterate through each label directory in DataCollect
for label_dir in os.listdir(data_collect_dir):
    label_path = os.path.join(data_collect_dir, label_dir)

    # Check if it's a directory
    if os.path.isdir(label_path):
        # Iterate through all files in the label directory
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)

            # Check if it's a file
            if os.path.isfile(file_path):
                # Copy the file to the Dataset/all directory
                shutil.copy(file_path, os.path.join(dataset_all_dir, file))

print("All files have been moved to Dataset/all directory.")
