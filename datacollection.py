
import cv2
import os
import numpy as np
from datetime import datetime

# Create a directory to store the collected data if it doesn't exist
output_dir = "DataCollect"
os.makedirs(output_dir, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set a counter for the filename
counter = 0

# Define the labels and corresponding keys
labels = {
    'r': 'real',
    'f': 'fake',
    'm': 'mask',
    'g': 'goggles',
    'n': 'no-mask'
}

# Create a directory for each label
for label in labels.values():
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

print("Press the following keys to capture images:")
for key, label in labels.items():
    print(f"'{key}' - {label}")
print("Press 'q' to quit")

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Data Collection', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, break from the loop
    if key == ord('q'):
        break

    # If a label key is pressed, save the image
    if key in labels:
        # If a face is detected, save the image and label
        if len(faces) > 0:
            # Get the first face detected
            x, y, w, h = faces[0]

            # Generate a unique filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"{timestamp}_{counter}.jpg"
            label_filename = f"{timestamp}_{counter}.txt"

            # Save the image
            img_path = os.path.join(output_dir, labels[key], img_filename)
            cv2.imwrite(img_path, frame)

            # Save the YOLO format label
            label_path = os.path.join(output_dir, labels[key], label_filename)

            # Calculate normalized coordinates for YOLO format
            img_height, img_width = frame.shape[:2]
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height

            # Determine the class index (0 for fake, 1 for real)
            class_index = 1 if labels[key] == 'real' else 0

            # Write the label to the file
            with open(label_path, 'w') as f:
                f.write(f"{class_index} {x_center} {y_center} {width} {height}")

            print(f"Saved {img_filename} with label {labels[key]}")
            counter += 1
        else:
            print("No face detected. Please make sure your face is visible in the camera.")

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
