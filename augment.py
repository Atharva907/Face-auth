
import os
import cv2
import numpy as np
import random
from glob import glob

# Define the path to the dataset
dataset_path = "Dataset/all"
augmented_path = "Dataset/all"  # We'll save augmented images in the same directory

# Ensure the directory exists
os.makedirs(augmented_path, exist_ok=True)

# Function to apply random brightness
def random_brightness(image, lower=0.7, upper=1.3):
    # Convert to HSV to modify brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(lower, upper)
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Function to apply random rotation
def random_rotation(image, angle_range=(-30, 30)):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    angle = random.uniform(angle_range[0], angle_range[1])
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

# Function to apply horizontal flip
def horizontal_flip(image):
    return cv2.flip(image, 1)

# Function to apply blur
def apply_blur(image):
    kernel_size = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Function to add noise
def add_noise(image):
    # Add Gaussian noise
    mean = 0
    std = random.uniform(5, 20)
    noisy = image.copy()
    h, w, c = noisy.shape
    noise = np.random.normal(mean, std, (h, w, c)).astype(np.uint8)
    noisy = cv2.add(noisy, noise)
    return noisy

# Function to adjust contrast
def adjust_contrast(image, lower=0.7, upper=1.3):
    alpha = random.uniform(lower, upper)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

# Function to adjust color
def adjust_color(image, lower=0.7, upper=1.3):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    # Randomly adjust the saturation
    hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(lower, upper)
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    # Convert back to BGR
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Get all image files
image_files = glob(os.path.join(dataset_path, "*.jpg"))
image_files.extend(glob(os.path.join(dataset_path, "*.png")))
image_files.extend(glob(os.path.join(dataset_path, "*.jpeg")))

# Number of augmentations per image
augmentations_per_image = 3

# Process each image
for img_path in image_files:
    # Read the image
    image = cv2.imread(img_path)
    if image is None:
        continue

    # Get the base filename and extension
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    ext = os.path.splitext(os.path.basename(img_path))[1]

    # Read the corresponding label file if it exists
    label_path = os.path.join(dataset_path, base_name + ".txt")
    label_content = ""
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            label_content = f.read()

    # Create augmented versions
    for i in range(augmentations_per_image):
        # Start with the original image
        augmented_image = image.copy()

        # Apply random transformations
        if random.random() > 0.5:
            augmented_image = random_brightness(augmented_image)

        if random.random() > 0.5:
            augmented_image = random_rotation(augmented_image)

        if random.random() > 0.5:
            augmented_image = horizontal_flip(augmented_image)

        if random.random() > 0.5:
            augmented_image = apply_blur(augmented_image)

        if random.random() > 0.5:
            augmented_image = add_noise(augmented_image)

        if random.random() > 0.5:
            augmented_image = adjust_contrast(augmented_image)

        if random.random() > 0.5:
            augmented_image = adjust_color(augmented_image)

        # Save the augmented image
        aug_img_path = os.path.join(augmented_path, f"{base_name}_aug{i}{ext}")
        cv2.imwrite(aug_img_path, augmented_image)

        # Save the corresponding label file if it exists
        if label_content:
            aug_label_path = os.path.join(augmented_path, f"{base_name}_aug{i}.txt")
            with open(aug_label_path, 'w') as f:
                f.write(label_content)

print(f"Data augmentation complete. Created {len(image_files) * augmentations_per_image} augmented images.")
