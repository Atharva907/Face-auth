
import cv2
import numpy as np
import os
import pickle
import time
from insightface import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

# Directory containing face embeddings
embeddings_dir = "face_embeddings"

# Load all saved embeddings
embeddings = []
for file in os.listdir(embeddings_dir):
    if file.endswith('.pkl'):
        with open(os.path.join(embeddings_dir, file), 'rb') as f:
            embeddings.append(pickle.load(f))

print(f"Loaded {len(embeddings)} face embeddings")

# Initialize InsightFace for face recognition
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)  # Use CPU, set ctx_id=-1 to use GPU if available

# Load YOLO model for anti-spoofing
# Update this path to where your trained model is saved
model_path = "runs/detect/train/weights/best.pt"
if os.path.exists(model_path):
    spoof_detector = YOLO(model_path)
    print("Loaded anti-spoofing model")
else:
    print("Anti-spoofing model not found. Please train the model first.")
    spoof_detector = None

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Thresholds
recognition_threshold = 0.6
spoof_confidence_threshold = 0.7

print("iPhone-like Face Authentication System")
print("Press 'q' to quit")

# Variables for tracking
last_recognition_time = 0
access_granted = False
access_denied = False

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get faces
    faces = app.get(frame)

    # Process each face
    for face in faces:
        bbox = face.bbox.astype(int)

        # Step 1: Anti-spoofing check
        is_real = True  # Default to real if no model is available
        if spoof_detector:
            # Crop the face region for spoof detection
            face_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Run spoof detection
            results = spoof_detector.predict(face_region, verbose=False)

            # Get the prediction
            if len(results) > 0 and len(results[0].boxes.cls) > 0:
                # Get the class with highest confidence
                cls = int(results[0].boxes.cls[0].item())
                conf = float(results[0].boxes.conf[0].item())

                # Class 1 is real, 0 is fake
                is_real = cls == 1 and conf >= spoof_confidence_threshold

        # Step 2: Face recognition if it's a real face
        is_match = False
        similarity_score = 0

        if is_real:
            # Get the embedding for the current face
            embedding = face.embedding

            # Calculate similarity with all saved embeddings
            similarities = []
            for saved_embedding in embeddings:
                similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
                similarities.append(similarity)

            # Find the maximum similarity
            similarity_score = max(similarities) if similarities else 0

            # Determine if it's a match
            is_match = similarity_score >= recognition_threshold

        # Step 3: Final decision
        current_time = time.time()
        if is_real and is_match:
            access_granted = True
            access_denied = False
            last_recognition_time = current_time
            color = (0, 255, 0)  # Green for access granted
            label = "ACCESS GRANTED"
        elif not is_real:
            access_granted = False
            access_denied = True
            last_recognition_time = current_time
            color = (0, 0, 255)  # Red for access denied (spoof detected)
            label = "SPOOF DETECTED"
        else:
            if current_time - last_recognition_time < 2:  # Keep showing the last status for 2 seconds
                color = (0, 0, 255) if access_denied else (0, 255, 0)
                label = "ACCESS DENIED" if access_denied else "ACCESS GRANTED"
            else:
                access_granted = False
                access_denied = False
                color = (255, 0, 255)  # Purple for no match
                label = "ACCESS DENIED"

        # Draw rectangle and label
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show similarity score if it's a real face
        if is_real:
            cv2.putText(frame, f"Similarity: {similarity_score:.2f}", (bbox[0], bbox[3] + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the frame
    cv2.imshow('Face Authentication', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, break from the loop
    if key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
