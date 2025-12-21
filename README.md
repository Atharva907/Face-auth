
# iPhone-like Face Authentication System

This project implements an iPhone-like face authentication system with anti-spoofing capabilities using Python and deep learning models.

## Features

- Real-time face detection and recognition
- Anti-spoofing detection to prevent attacks using photos or videos
- Face registration for authentication
- User-friendly interface using CustomTkinter
- Support for various face conditions (with mask, goggles, etc.)

## Project Structure

```
Puja/
├── Dataset/
│   ├── all/                   # All collected and augmented images
│   └── splitData/             # Split data for training
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       └── data.yaml          # YOLO dataset configuration
├── DataCollect/               # Collected data organized by labels
├── face_embeddings/           # Saved face embeddings for recognition
├── datacollection.py          # Script to collect face data
├── move_to_all.py             # Script to move collected data
├── augment.py                 # Script to augment data
├── splitData.py               # Script to split data into train/val/test
├── register_face.py           # Script to register faces
├── face_recognition.py        # Script for face recognition
├── main.py                    # Main authentication script
├── ui.py                      # GUI application
└── requirements.txt           # Python dependencies
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Puja.git
   cd Puja
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Collection

1. Run the data collection script:
   ```
   python datacollection.py
   ```

   Use the following keys to capture images:
   - 'r' - real face
   - 'f' - fake face (printed, screen)
   - 'm' - face with mask
   - 'g' - face with goggles
   - 'n' - face without mask
   - 'q' - quit

2. Move collected data to the dataset:
   ```
   python move_to_all.py
   ```

3. Augment the data:
   ```
   python augment.py
   ```

4. Split the data into train/val/test:
   ```
   python splitData.py
   ```

### Model Training

1. Train the anti-spoofing model:
   ```
   yolo train model=yolov8n.pt data=Dataset/splitData/data.yaml epochs=100 imgsz=320
   ```

### Face Registration and Authentication

1. Register faces:
   ```
   python register_face.py
   ```
   Press 's' to save face embeddings.

2. Run the authentication system:
   ```
   python main.py
   ```

3. Or use the GUI:
   ```
   python ui.py
   ```

## How It Works

1. **Face Detection**: Uses InsightFace to detect faces in real-time.
2. **Anti-Spoofing**: Uses a YOLO model to detect if the face is real or fake.
3. **Face Recognition**: Compares face embeddings with stored embeddings using cosine similarity.
4. **Decision Making**: If the face is real and matches with stored embeddings, access is granted.

## Customization

- Adjust the recognition threshold in the UI or in the scripts.
- Add more face conditions to the dataset for better performance.
- Modify the UI to suit your needs.

## Troubleshooting

- If the camera doesn't work, check if it's being used by another application.
- If the anti-spoofing model is not found, make sure you've trained the model first.
- For performance issues, try reducing the input image size or using a GPU.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.
