
import cv2
import numpy as np
import os
import pickle
from insightface import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Directory containing face embeddings
embeddings_dir = "face_embeddings"

# Load all saved embeddings
embeddings = []
for file in os.listdir(embeddings_dir):
    if file.endswith('.pkl'):
        with open(os.path.join(embeddings_dir, file), 'rb') as f:
            embeddings.append(pickle.load(f))

print(f"Loaded {len(embeddings)} face embeddings")

# Initialize InsightFace
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)  # Use CPU, set ctx_id=-1 to use GPU if available

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Threshold for face recognition (adjust as needed)
recognition_threshold = 0.6

print("Face Recognition System")
print("Press 'q' to quit")

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

        # Get the embedding for the current face
        embedding = face.embedding

        # Calculate similarity with all saved embeddings
        similarities = []
        for saved_embedding in embeddings:
            similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
            similarities.append(similarity)

        # Find the maximum similarity
        max_similarity = max(similarities) if similarities else 0

        # Determine if it's a match
        is_match = max_similarity >= recognition_threshold

        # Draw rectangle and label
        color = (0, 255, 0) if is_match else (0, 0, 255)
        label = "Match" if is_match else "No Match"
        confidence = f"{max_similarity:.2f}"

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f"{label}: {confidence}", (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, break from the loop
    if key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
