
import cv2
import numpy as np
import os
import pickle
from insightface import FaceAnalysis

# Create a directory to store face embeddings if it doesn't exist
embeddings_dir = "face_embeddings"
os.makedirs(embeddings_dir, exist_ok=True)

# Initialize InsightFace
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)  # Use CPU, set ctx_id=-1 to use GPU if available

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Counter for saved embeddings
counter = 0

print("Face Registration System")
print("Press 's' to save face embedding")
print("Press 'q' to quit")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get face embeddings
    faces = app.get(frame)

    # Draw rectangles around faces
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Registration', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, break from the loop
    if key == ord('q'):
        break

    # If 's' is pressed, save the face embedding
    if key == ord('s'):
        if len(faces) > 0:
            # Get the first face
            face = faces[0]

            # Get the embedding
            embedding = face.embedding

            # Save the embedding
            embedding_path = os.path.join(embeddings_dir, f"face_{counter}.pkl")
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding, f)

            print(f"Saved face embedding to {embedding_path}")
            counter += 1
        else:
            print("No face detected. Please make sure your face is visible in the camera.")

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

print(f"Registration complete. {counter} face embeddings saved.")
