
# Installation Guide for iPhone-like Face Authentication System

This guide provides detailed instructions for setting up the iPhone-like Face Authentication System on your system.

## Prerequisites

- Python 3.10-3.11 (Python 3.13 may have compatibility issues with some packages)
- A working webcam
- Git (for cloning the repository)

## Step 1: Setting Up Python Environment

### Option A: Using a Virtual Environment (Recommended)

1. Create a virtual environment:
   ```
   python -m venv face_auth_env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     face_auth_env\Scriptsctivate
     ```
   - On macOS/Linux:
     ```
     source face_auth_env/bin/activate
     ```

### Option B: Using Anaconda (Alternative)

1. Install Anaconda from https://www.anaconda.com/products/distribution
2. Create a new environment:
   ```
   conda create -n face_auth python=3.11
   ```
3. Activate the environment:
   ```
   conda activate face_auth
   ```

## Step 2: Installing Dependencies

Once your Python environment is set up, install the required packages:

### Method 1: Using pip (Standard)

1. Upgrade pip:
   ```
   python -m pip install --upgrade pip
   ```

2. Install from requirements.txt:
   ```
   pip install -r requirements.txt
   ```

### Method 2: Installing Packages Individually

If you encounter issues with requirements.txt, try installing packages individually:

1. Core dependencies:
   ```
   pip install opencv-python
   pip install numpy
   pip install Pillow
   pip install scikit-learn
   ```

2. Face recognition and detection:
   ```
   pip install insightface
   pip install onnxruntime
   ```

3. Object detection for anti-spoofing:
   ```
   pip install ultralytics
   ```

4. UI dependencies:
   ```
   pip install customtkinter
   ```

### Method 3: Using Conda (If using Anaconda)

1. Install core packages:
   ```
   conda install opencv pillow numpy scikit-learn
   ```

2. Install remaining packages with pip:
   ```
   pip install insightface onnxruntime ultralytics customtkinter
   ```

## Step 3: Verifying Installation

1. Check if all packages are installed correctly:
   ```
   python -c "import cv2; import numpy; import PIL; import sklearn; import insightface; import onnxruntime; import ultralytics; import customtkinter; print('All packages imported successfully')"
   ```

## Troubleshooting Common Issues

### Issue 1: "Fatal error in launcher" with pip

This error typically occurs when there's a mismatch between your Python installation and pip. Try these solutions:

1. Use python -m pip instead of pip:
   ```
   python -m pip install <package_name>
   ```

2. Reinstall pip:
   ```
   python -m ensurepip --default-pip
   ```

3. Reinstall Python completely if the above doesn't work.

### Issue 2: Package Compatibility with Python 3.13

Python 3.13 is very new and many packages haven't been updated for it yet. Solutions:

1. Use Python 3.11 instead (recommended)
2. Find pre-release or development versions of packages that support Python 3.13
3. Wait for package maintainers to update their packages

### Issue 3: On Windows with MINGW64

If you're using Git Bash on Windows, try these solutions:

1. Use Command Prompt or PowerShell instead:
   ```
   cmd
   ```
   or
   ```
   powershell
   ```

2. Use winpty to prefix your commands:
   ```
   winpty pip install -r requirements.txt
   ```

3. Use the full path to pip:
   ```
   /c/Users/Admin/AppData/Local/Programs/Python/Python311/Scripts/pip.exe install -r requirements.txt
   ```

## Step 4: Running the Project

Once all dependencies are installed, you can run the project:

1. Register faces:
   ```
   python register_face.py
   ```

2. Run the authentication system with UI:
   ```
   python ui.py
   ```

3. Or run the command-line version:
   ```
   python main.py
   ```

## Additional Resources

- Python official documentation: https://docs.python.org/
- OpenCV documentation: https://docs.opencv.org/
- InsightFace documentation: https://github.com/deepinsight/insightface
- YOLOv8 documentation: https://docs.ultralytics.com/
- CustomTkinter documentation: https://github.com/TomSchimansky/CustomTkinter

If you encounter any other issues, please check the project's GitHub repository for updates or create an issue with details about your problem.
