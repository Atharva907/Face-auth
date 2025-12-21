import cv2
import os
from datetime import datetime

# =========================
# CONFIG
# =========================
output_dir = "DataCollect"
os.makedirs(output_dir, exist_ok=True)
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)
counter = 0

labels = {
    'r': 'real',
    'f': 'fake',
    'm': 'mask',
    'g': 'goggles',
    'n': 'no-mask'
}

# YOLO class mapping
class_map = {
    'real': 0,
    'fake': 1,
    'mask': 2,
    'goggles': 3,
    'no-mask': 4
}

# Create label folders
for label in labels.values():
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

print("Press the following keys to capture images:")
for key, label in labels.items():
    print(f"'{key}' - {label}")
print("Press 'q' to quit")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Data Collection', frame)

    key = cv2.waitKey(1) & 0xFF
    key_char = chr(key)

    if key_char == 'q':
        break

    if key_char in labels:
        if len(faces) == 0:
            print("No face detected. Please face the camera.")
            continue

        x, y, w, h = faces[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"{timestamp}_{counter}.jpg"
        label_filename = f"{timestamp}_{counter}.txt"

        label_name = labels[key_char]

        img_path = os.path.join(output_dir, label_name, img_filename)
        label_path = os.path.join(output_dir, label_name, label_filename)

        # Save image
        cv2.imwrite(img_path, frame)

        # YOLO normalized coordinates
        img_h, img_w = frame.shape[:2]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        width = w / img_w
        height = h / img_h

        class_index = class_map[label_name]

        with open(label_path, 'w') as f:
            f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        print(f"Saved {img_filename} → {label_name}")
        counter += 1

cap.release()
cv2.destroyAllWindows()
