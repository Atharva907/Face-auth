
# Development Guide - iPhone-like Face Authentication System

This guide provides detailed technical information about the iPhone-like Face Authentication System for developers who want to understand, modify, or extend the project.

## Architecture Overview

The system consists of several components:

1. **Data Collection Module** (`datacollection.py`)
   - Captures face images with different labels
   - Generates YOLO format annotations
   - Organizes data by categories

2. **Data Processing Module** (`move_to_all.py`, `augment.py`, `splitData.py`)
   - Consolidates collected data
   - Applies data augmentation
   - Splits data into train/validation/test sets

3. **Face Registration Module** (`register_face.py`)
   - Captures face embeddings using InsightFace
   - Saves embeddings for future recognition

4. **Face Recognition Module** (`face_recognition.py`)
   - Detects faces in camera feed
   - Compares with registered embeddings
   - Determines match status

5. **Anti-Spoofing Module** (YOLO model)
   - Trained on real/fake face images
   - Detects spoof attempts using photos/videos

6. **Main Application** (`main.py`)
   - Integrates all components
   - Provides end-to-end authentication flow

7. **GUI Application** (`ui.py`)
   - User-friendly interface using CustomTkinter
   - Simplifies interaction with all components

## Technical Details

### Face Detection and Recognition

We use InsightFace for face detection and recognition:

```python
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Detect faces in an image
faces = app.get(image)

# Extract face embedding
embedding = faces[0].embedding
```

### Anti-Spoofing with YOLO

The anti-spoofing model is trained using YOLOv8:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run inference
results = model.predict(image)

# Get class and confidence
cls = int(results[0].boxes.cls[0].item())
conf = float(results[0].boxes.conf[0].item())

# Class 1 is real, 0 is fake
is_real = cls == 1 and conf >= threshold
```

### Data Augmentation

We apply various augmentations to improve model robustness:

```python
# Random brightness
def random_brightness(image, lower=0.7, upper=1.3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(lower, upper)
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Random rotation
def random_rotation(image, angle_range=(-30, 30)):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    angle = random.uniform(angle_range[0], angle_range[1])
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))
```

### Face Matching

We use cosine similarity for face matching:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity between embeddings
similarity = cosine_similarity([current_embedding], [saved_embedding])[0][0]

# Determine if it's a match
is_match = similarity >= threshold
```

## Customization Guide

### Adjusting Recognition Threshold

The recognition threshold determines how strict the face matching is:

- Higher threshold (0.7-0.9): More secure but less forgiving
- Lower threshold (0.4-0.6): More forgiving but less secure

You can adjust this in the UI or in the code:

```python
recognition_threshold = 0.6  # Default value
```

### Adding New Face Conditions

To add new face conditions (e.g., with sunglasses):

1. Add a new key in `datacollection.py`:
```python
labels = {
    'r': 'real',
    'f': 'fake',
    'm': 'mask',
    'g': 'goggles',
    'n': 'no-mask',
    's': 'sunglasses'  # New condition
}
```

2. Collect data for the new condition
3. Retrain the anti-spoofing model

### Improving Anti-Spoofing

To improve the anti-spoofing model:

1. Collect more diverse fake face samples
2. Try different model architectures
3. Adjust training parameters:
```bash
yolo train model=yolov8s.pt data=Dataset/splitData/data.yaml epochs=200 imgsz=640 batch=16
```

### Enhancing Liveness Detection

To add more liveness detection features:

1. Blink detection:
```python
def detect_blink(landmarks):
    # Calculate eye aspect ratio
    left_eye = landmarks[42:48]
    right_eye = landmarks[36:42]

    # Calculate eye aspect ratio for both eyes
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    # Average the eye aspect ratio
    ear = (left_ear + right_ear) / 2

    # Check if eyes are closed
    return ear < 0.25
```

2. Head movement detection:
```python
def detect_head_movement(prev_landmarks, curr_landmarks):
    # Calculate the Euclidean distance between landmarks
    distance = np.linalg.norm(prev_landmarks - curr_landmarks)

    # Check if there's significant movement
    return distance > threshold
```

## API Reference

### Data Collection Module

`datacollection.py` provides the following functions:

- `collect_face_data()`: Main function to collect face data
- `save_image_and_label(image, label, bbox)`: Saves image and corresponding label

### Face Registration Module

`register_face.py` provides the following functions:

- `register_face()`: Main function to register faces
- `save_embedding(embedding)`: Saves face embedding to file

### Face Recognition Module

`face_recognition.py` provides the following functions:

- `recognize_face(image)`: Recognizes faces in an image
- `compare_embeddings(emb1, emb2)`: Compares two face embeddings

### Main Application

`main.py` provides the following functions:

- `authenticate()`: Main authentication function
- `detect_spoof(image)`: Detects if a face is real or fake
- `recognize_user(image)`: Recognizes registered users

## Performance Optimization

### GPU Acceleration

To use GPU acceleration:

1. Install CUDA-compatible versions of packages
2. Modify the code to use GPU:
```python
# For InsightFace
app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

# For YOLO
model = YOLO('model.pt')
model.to('cuda')
```

### Model Quantization

To reduce model size and improve inference speed:

1. Quantize the YOLO model:
```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx', int8=True)
```

### Multi-threading

To improve performance with multiple cameras:

```python
import threading

class CameraThread(threading.Thread):
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.frame = None
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_id)

        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame = frame

        cap.release()

    def stop(self):
        self.running = False
```

## Security Considerations

### Embedding Storage

Face embeddings should be stored securely:

1. Encrypt embeddings:
```python
import cryptography.fernet

key = cryptography.fernet.Fernet.generate_key()
cipher_suite = cryptography.fernet.Fernet(key)

# Encrypt
encrypted_embedding = cipher_suite.encrypt(embedding.tobytes())

# Decrypt
decrypted_embedding = np.frombuffer(cipher_suite.decrypt(encrypted_embedding))
```

2. Consider using a secure database for storage

### Anti-Spoofing Robustness

To improve anti-spoofing robustness:

1. Use multiple anti-spoofing techniques
2. Implement challenge-response mechanisms
3. Add temporal consistency checks

## Testing

### Unit Tests

Create unit tests for key components:

```python
import unittest
import numpy as np
from face_recognition import compare_embeddings

class TestFaceRecognition(unittest.TestCase):
    def setUp(self):
        # Create test embeddings
        self.emb1 = np.random.rand(512)
        self.emb2 = np.random.rand(512)
        self.emb3 = self.emb1 + np.random.rand(512) * 0.1  # Similar to emb1

    def test_identical_embeddings(self):
        # Test with identical embeddings
        similarity = compare_embeddings(self.emb1, self.emb1)
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_different_embeddings(self):
        # Test with different embeddings
        similarity = compare_embeddings(self.emb1, self.emb2)
        self.assertLess(similarity, 0.5)

    def test_similar_embeddings(self):
        # Test with similar embeddings
        similarity = compare_embeddings(self.emb1, self.emb3)
        self.assertGreater(similarity, 0.9)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

Create integration tests for the complete workflow:

```python
import unittest
import cv2
import os
from main import authenticate

class TestAuthentication(unittest.TestCase):
    def setUp(self):
        # Setup test environment
        self.test_image_path = 'test_images/real_face.jpg'
        self.assertTrue(os.path.exists(self.test_image_path))

    def test_real_face_authentication(self):
        # Test authentication with a real face
        image = cv2.imread(self.test_image_path)
        result = authenticate(image)
        self.assertTrue(result['is_real'])

    def test_fake_face_detection(self):
        # Test detection of a fake face
        image = cv2.imread('test_images/fake_face.jpg')
        result = authenticate(image)
        self.assertFalse(result['is_real'])

if __name__ == '__main__':
    unittest.main()
```

## Deployment

### Docker Deployment

Create a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     libgl1-mesa-glx     libglib2.0-0     libsm6     libxext6     libxrender-dev     libgomp1

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p Dataset/all Dataset/splitData face_embeddings

# Expose port for web interface (if applicable)
EXPOSE 8000

# Run the application
CMD ["python", "ui.py"]
```

### Web Deployment

To create a web interface using Flask:

```python
from flask import Flask, render_template, Response
import cv2
import json

app = Flask(__name__)

camera = None

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process frame for authentication
            result = authenticate(frame)

            # Add result overlay to frame
            cv2.putText(frame, f"Real: {result['is_real']}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if result['is_real'] else (0, 0, 255), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame
'
                   b'Content-Type: image/jpeg

' + frame + b'
')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
```

## Contributing

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE for more information.
