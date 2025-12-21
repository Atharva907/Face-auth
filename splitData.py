
import os
import random
import shutil
from glob import glob

# Define paths
input_folder = "Dataset/all"
output_folder = "Dataset/splitData"

# Create output directories if they don't exist
os.makedirs(os.path.join(output_folder, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "val", "labels"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "test", "images"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "test", "labels"), exist_ok=True)

# Get all image files
image_files = glob(os.path.join(input_folder, "*.jpg"))
image_files.extend(glob(os.path.join(input_folder, "*.png")))
image_files.extend(glob(os.path.join(input_folder, "*.jpeg")))

# Shuffle the files
random.shuffle(image_files)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Calculate split indices
train_count = int(len(image_files) * train_ratio)
val_count = int(len(image_files) * val_ratio)
test_count = len(image_files) - train_count - val_count

# Split the files
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# Function to copy files and their corresponding labels
def copy_files(files, dest_folder):
    for file_path in files:
        # Get the filename
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]

        # Copy the image
        shutil.copy(file_path, os.path.join(output_folder, dest_folder, "images", filename))

        # Copy the corresponding label if it exists
        label_path = os.path.join(input_folder, base_name + ".txt")
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_folder, dest_folder, "labels", base_name + ".txt"))

# Copy files to their respective folders
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

# Create the data.yaml file for YOLO
data_yaml = f"""path: {os.path.abspath(output_folder)}
train: train/images
val: val/images
test: test/images

nc: 2  # number of classes
names: ['fake', 'real']  # class names
"""

with open(os.path.join(output_folder, "data.yaml"), 'w') as f:
    f.write(data_yaml)

print(f"Data split complete:")
print(f"Train: {len(train_files)} images")
print(f"Validation: {len(val_files)} images")
print(f"Test: {len(test_files)} images")
print(f"data.yaml created at {os.path.join(output_folder, 'data.yaml')}")
