
# iPhone-like Face Authentication System - Workflow Guide

This guide explains the complete workflow of the iPhone-like Face Authentication System, from data collection to deployment.

## Overview

The system consists of two main components:
1. **Anti-Spoofing Model**: Detects whether a face is real or fake
2. **Face Recognition Model**: Identifies registered users

## Step 1: Data Collection

### Purpose
Collect face images with different conditions to train a robust anti-spoofing model.

### Process
1. Run the data collection script:
   ```
   python datacollection.py
   ```

2. Use the following keys to capture images:
   - 'r' - real face (unobstructed)
   - 'f' - fake face (printed photos, screen images)
   - 'm' - face with mask
   - 'g' - face with goggles
   - 'n' - face without mask
   - 'q' - quit

3. The script will save:
   - Image files (.jpg)
   - Label files (.txt) in YOLO format

### Tips
- Collect at least 100 images for each category
- Ensure diverse lighting conditions
- Include different angles and expressions
- For fake faces, use various media (printed photos, phone screens, etc.)

## Step 2: Data Processing

### Move Data to Central Location
1. Run the move script:
   ```
   python move_to_all.py
   ```
   This moves all collected data to `Dataset/all/`

### Augment Data
1. Run the augmentation script:
   ```
   python augment.py
   ```
   This applies random transformations:
   - Brightness adjustment
   - Rotation
   - Flipping
   - Blur
   - Noise
   - Contrast adjustment
   - Color shift

### Split Data for Training
1. Run the split script:
   ```
   python splitData.py
   ```
   This splits data into:
   - Training set (70%)
   - Validation set (20%)
   - Test set (10%)

## Step 3: Model Training

### Train Anti-Spoofing Model
1. Run the YOLO training command:
   ```
   yolo train model=yolov8n.pt data=Dataset/splitData/data.yaml epochs=100 imgsz=320
   ```

2. The trained model will be saved at:
   ```
   runs/detect/train/weights/best.pt
   ```

### Model Evaluation
1. Check training metrics in:
   ```
   runs/detect/train/results.csv
   ```

2. Visualize training curves:
   ```
   runs/detect/train/results.png
   ```

## Step 4: Face Registration

### Purpose
Register authorized users by saving their face embeddings.

### Process
1. Run the registration script:
   ```
   python register_face.py
   ```

2. Position your face in front of the camera

3. Press 's' to save your face embedding

4. Repeat for different angles and lighting conditions

5. Press 'q' to quit

### Tips
- Register at least 5-10 embeddings per user
- Include different expressions and angles
- Ensure good lighting

## Step 5: Face Recognition

### Purpose
Recognize registered users and grant access.

### Process
1. Run the recognition script:
   ```
   python face_recognition.py
   ```

2. The system will:
   - Detect faces in the camera feed
   - Compare with registered embeddings
   - Display match status and similarity score

### Tips
- Adjust the recognition threshold based on your needs
- Higher threshold = more secure but less forgiving
- Lower threshold = more forgiving but less secure

## Step 6: Full Authentication System

### Purpose
Combine anti-spoofing and face recognition for complete authentication.

### Process
1. Run the main script:
   ```
   python main.py
   ```

2. The system will:
   - Detect faces
   - Check if the face is real (anti-spoofing)
   - If real, compare with registered embeddings
   - Grant or deny access based on results

### Status Messages
- "ACCESS GRANTED" - Real face and matches registered user
- "SPOOF DETECTED" - Fake face detected
- "ACCESS DENIED" - Real face but doesn't match registered user

## Step 7: Using the GUI

### Purpose
Provide a user-friendly interface for all operations.

### Process
1. Run the GUI application:
   ```
   python ui.py
   ```

2. Use the buttons to:
   - Register Face: Open registration window
   - Authenticate: Start authentication process
   - Stop: Stop camera and authentication
   - Clear Saved Data: Remove all registered embeddings

3. Adjust the Recognition Threshold slider to balance security and convenience

## Troubleshooting

### Common Issues
1. Camera not detected
   - Check if camera is connected
   - Ensure no other application is using the camera
   - Try restarting the application

2. Poor recognition accuracy
   - Register more face embeddings
   - Ensure good lighting
   - Try different angles
   - Adjust the recognition threshold

3. False positives in anti-spoofing
   - Collect more fake face samples
   - Retrain the anti-spoofing model
   - Adjust the spoof confidence threshold

## Advanced Features

### Liveness Detection
You can enhance the system with liveness detection by:
1. Adding blink detection
2. Requiring head movement
3. Implementing mouth open detection

### Depth-based Anti-Spoofing
If you have a depth camera:
1. Use MediaPipe Depth model
2. Combine with existing anti-spoofing
3. Train a custom model using depth data

## Deployment

### Creating an Executable
1. Install PyInstaller:
   ```
   pip install pyinstaller
   ```

2. Create executable:
   ```
   pyinstaller --noconfirm --onefile main.py
   ```

3. The executable will be in the `dist` folder

### Packaging for Distribution
1. Include the following files:
   - Main executable
   - Model files (best.pt)
   - Configuration files
   - Documentation

2. Create an installer using tools like:
   - Inno Setup (Windows)
   - PyInstaller (cross-platform)
   - Briefcase (BeeWare)

## Security Considerations

1. Store face embeddings securely
2. Implement encryption for sensitive data
3. Consider adding multi-factor authentication
4. Regularly update the anti-spoofing model
5. Monitor for unusual access patterns

## Future Enhancements

1. Support for multiple users
2. Cloud-based authentication
3. Mobile app integration
4. Advanced liveness detection
5. Integration with access control systems
