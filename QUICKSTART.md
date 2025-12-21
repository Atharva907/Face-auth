
# Quick Start Guide - iPhone-like Face Authentication System

This guide provides a quick overview to get you started with the iPhone-like Face Authentication System.

## Prerequisites

- Python 3.10-3.11 (Python 3.13 may have compatibility issues)
- A working webcam
- Basic understanding of command line

## Quick Setup

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv face_auth_env

# Activate the environment
# On Windows:
face_auth_env\Scriptsctivate
# On macOS/Linux:
source face_auth_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Face Data

```bash
# Run data collection
python datacollection.py

# Press 'r' for real face, 'f' for fake face, 'm' for mask, 'g' for goggles
# Press 'q' to quit when done
```

### 3. Process Data

```bash
# Move data to central location
python move_to_all.py

# Augment data
python augment.py

# Split data for training
python splitData.py
```

### 4. Train Anti-Spoofing Model

```bash
# Train YOLO model
yolo train model=yolov8n.pt data=Dataset/splitData/data.yaml epochs=100 imgsz=320
```

### 5. Register Faces

```bash
# Register your face
python register_face.py

# Press 's' to save face embedding
# Press 'q' to quit when done
```

### 6. Run the Application

#### Option 1: GUI Application (Recommended)

```bash
# Run the GUI
python ui.py
```

- Click "Register Face" to add new faces
- Click "Authenticate" to start face recognition
- Adjust the recognition threshold as needed

#### Option 2: Command Line Application

```bash
# Run the command line version
python main.py
```

## Troubleshooting

### Common Issues

1. **"Fatal error in launcher" with pip**
   - Try: `python -m pip install <package_name>`
   - Or reinstall Python

2. **Camera not detected**
   - Check if camera is connected
   - Ensure no other app is using it
   - Try restarting the application

3. **Poor recognition accuracy**
   - Register more face embeddings
   - Ensure good lighting
   - Adjust the recognition threshold

## Next Steps

- Read the full documentation in README.md
- Check the detailed workflow in WORKFLOW.md
- Follow the installation guide in INSTALLATION.md

## Need Help?

- Check the project documentation
- Create an issue on GitHub
- Contact the development team
