import customtkinter as ctk
import cv2
import numpy as np
import os
import pickle
import time
import threading
from PIL import Image, ImageTk
from insightface import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

class FaceAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("iPhone-like Face Authentication System")
        self.root.geometry("800x600")

        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")  # Options: "System", "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        # Variables
        self.embeddings = []
        self.app = None
        self.spoof_detector = None
        self.cap = None
        self.current_frame = None
        self.is_running = False
        self.recognition_threshold = 0.6
        self.spoof_confidence_threshold = 0.7

        # Setup UI
        self.setup_ui()

        # Initialize models
        self.initialize_models()

    def setup_ui(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="iPhone-like Face Authentication System", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=10)

        # Camera frame
        self.camera_frame = ctk.CTkFrame(self.main_frame)
        self.camera_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.camera_label = ctk.CTkLabel(self.camera_frame, text="Camera will appear here")
        self.camera_label.pack(fill="both", expand=True, padx=5, pady=5)

        # Control buttons frame
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(fill="x", padx=10, pady=10)

        # Buttons
        self.register_btn = ctk.CTkButton(
            self.control_frame, 
            text="Register Face", 
            command=self.register_face
        )
        self.register_btn.pack(side="left", padx=5, pady=5)

        self.authenticate_btn = ctk.CTkButton(
            self.control_frame, 
            text="Authenticate", 
            command=self.start_authentication
        )
        self.authenticate_btn.pack(side="left", padx=5, pady=5)

        self.stop_btn = ctk.CTkButton(
            self.control_frame, 
            text="Stop", 
            command=self.stop_camera
        )
        self.stop_btn.pack(side="left", padx=5, pady=5)

        self.clear_btn = ctk.CTkButton(
            self.control_frame, 
            text="Clear Saved Data", 
            command=self.clear_data
        )
        self.clear_btn.pack(side="left", padx=5, pady=5)

        # Status label
        self.status_label = ctk.CTkLabel(
            self.main_frame, 
            text="Ready", 
            font=ctk.CTkFont(size=16)
        )
        self.status_label.pack(pady=5)

        # Settings frame
        self.settings_frame = ctk.CTkFrame(self.main_frame)
        self.settings_frame.pack(fill="x", padx=10, pady=5)

        # Recognition threshold slider
        self.threshold_label = ctk.CTkLabel(self.settings_frame, text="Recognition Threshold:")
        self.threshold_label.pack(side="left", padx=5)

        self.threshold_slider = ctk.CTkSlider(
            self.settings_frame, 
            from_=0.1, 
            to=1.0, 
            number_of_steps=9,
            command=self.update_threshold
        )
        self.threshold_slider.set(self.recognition_threshold)
        self.threshold_slider.pack(side="left", padx=5, fill="x", expand=True)

        # Status frame
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.pack(fill="x", padx=10, pady=5)

        self.status_text = ctk.CTkTextbox(self.status_frame, height=100)
        self.status_text.pack(fill="both", expand=True, padx=5, pady=5)

    def initialize_models(self):
        # Directory containing face embeddings
        embeddings_dir = "face_embeddings"

        # Create directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)

        # Load all saved embeddings
        self.embeddings = []
        for file in os.listdir(embeddings_dir):
            if file.endswith('.pkl'):
                with open(os.path.join(embeddings_dir, file), 'rb') as f:
                    self.embeddings.append(pickle.load(f))

        self.update_status(f"Loaded {len(self.embeddings)} face embeddings")

        # Initialize InsightFace for face recognition
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0)  # Use CPU, set ctx_id=-1 to use GPU if available

        # Load YOLO model for anti-spoofing
        model_path = "runs/detect/train/weights/best.pt"
        if os.path.exists(model_path):
            self.spoof_detector = YOLO(model_path)
            self.update_status("Loaded anti-spoofing model")
        else:
            self.update_status("Anti-spoofing model not found. Please train the model first.")
            self.spoof_detector = None

    def update_status(self, message):
        self.status_text.insert("end", f"{message}\n")
        self.status_text.see("end")
        self.status_label.configure(text=message)

    def update_threshold(self, value):
        self.recognition_threshold = float(value)

    def register_face(self):
        self.stop_camera()
        self.update_status("Starting face registration...")

        # Open a new window for registration
        self.reg_window = ctk.CTkToplevel(self.root)
        self.reg_window.title("Register Face")
        self.reg_window.geometry("640x480")

        # Camera frame in registration window
        self.reg_camera_frame = ctk.CTkFrame(self.reg_window)
        self.reg_camera_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.reg_camera_label = ctk.CTkLabel(self.reg_camera_frame, text="Initializing camera...")
        self.reg_camera_label.pack(fill="both", expand=True, padx=5, pady=5)

        # Buttons in registration window
        self.reg_btn_frame = ctk.CTkFrame(self.reg_window)
        self.reg_btn_frame.pack(fill="x", padx=10, pady=10)

        self.capture_btn = ctk.CTkButton(
            self.reg_btn_frame, 
            text="Capture Face", 
            command=self.capture_face
        )
        self.capture_btn.pack(side="left", padx=5, pady=5)

        self.close_reg_btn = ctk.CTkButton(
            self.reg_btn_frame, 
            text="Close", 
            command=self.close_registration
        )
        self.close_reg_btn.pack(side="left", padx=5, pady=5)

        # Start camera for registration
        self.reg_cap = cv2.VideoCapture(0)
        self.is_registering = True
        self.update_registration_camera()

    def update_registration_camera(self):
        if not self.is_registering:
            return

        # Read a frame from the webcam
        ret, frame = self.reg_cap.read()
        if ret:
            # Get faces
            faces = self.app.get(frame)

            # Draw rectangles around faces
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Convert frame to PIL Image and then to ImageTk
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the label
            self.reg_camera_label.configure(image=imgtk)
            self.reg_camera_label.image = imgtk

        # Schedule the next update
        self.root.after(10, self.update_registration_camera)

    def capture_face(self):
        # Read a frame from the webcam
        ret, frame = self.reg_cap.read()
        if ret:
            # Get faces
            faces = self.app.get(frame)

            if len(faces) > 0:
                # Get the first face
                face = faces[0]

                # Get the embedding
                embedding = face.embedding

                # Save the embedding
                embeddings_dir = "face_embeddings"
                counter = len([f for f in os.listdir(embeddings_dir) if f.endswith('.pkl')])
                embedding_path = os.path.join(embeddings_dir, f"face_{counter}.pkl")
                with open(embedding_path, 'wb') as f:
                    pickle.dump(embedding, f)

                # Update the embeddings list
                self.embeddings.append(embedding)

                self.update_status(f"Saved face embedding to {embedding_path}")

                # Show success message
                self.reg_camera_label.configure(text="Face registered successfully!")

                # Close registration window after a short delay
                self.root.after(2000, self.close_registration)
            else:
                self.reg_camera_label.configure(text="No face detected. Please try again.")

    def close_registration(self):
        self.is_registering = False
        if hasattr(self, 'reg_cap') and self.reg_cap:
            self.reg_cap.release()
        if hasattr(self, 'reg_window'):
            self.reg_window.destroy()

    def start_authentication(self):
        self.stop_camera()
        self.update_status("Starting authentication...")
        self.is_running = True

        # Start camera
        self.cap = cv2.VideoCapture(0)

        # Start the authentication loop in a separate thread
        self.auth_thread = threading.Thread(target=self.authentication_loop)
        self.auth_thread.daemon = True
        self.auth_thread.start()

    def authentication_loop(self):
        # Variables for tracking
        last_recognition_time = 0
        access_granted = False
        access_denied = False

        while self.is_running:
            # Read a frame from the webcam
            ret, frame = self.cap.read()
            if not ret:
                break

            self.current_frame = frame.copy()

            # Get faces
            faces = self.app.get(frame)

            # Process each face
            for face in faces:
                bbox = face.bbox.astype(int)

                # Step 1: Anti-spoofing check
                is_real = True  # Default to real if no model is available
                if self.spoof_detector:
                    # Crop the face region for spoof detection
                    face_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    # Run spoof detection
                    results = self.spoof_detector.predict(face_region, verbose=False)

                    # Get the prediction
                    if len(results) > 0 and len(results[0].boxes.cls) > 0:
                        # Get the class with highest confidence
                        cls = int(results[0].boxes.cls[0].item())
                        conf = float(results[0].boxes.conf[0].item())

                        # Class 1 is real, 0 is fake
                        is_real = cls == 1 and conf >= self.spoof_confidence_threshold

                # Step 2: Face recognition if it's a real face
                is_match = False
                similarity_score = 0

                if is_real:
                    # Get the embedding for the current face
                    embedding = face.embedding

                    # Calculate similarity with all saved embeddings
                    similarities = []
                    for saved_embedding in self.embeddings:
                        similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
                        similarities.append(similarity)

                    # Find the maximum similarity
                    similarity_score = max(similarities) if similarities else 0

                    # Determine if it's a match
                    is_match = similarity_score >= self.recognition_threshold

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

            # Convert frame to PIL Image and then to ImageTk
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the label in the main thread
            self.root.after(0, self.update_camera_label, imgtk)

            # Sleep for a short time to reduce CPU usage
            time.sleep(0.01)

    def update_camera_label(self, imgtk):
        self.camera_label.configure(image=imgtk)
        self.camera_label.image = imgtk

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        # Reset the camera label
        self.camera_label.configure(image="", text="Camera stopped")

    def clear_data(self):
        # Ask for confirmation
        if self.embeddings:
            response = ctk.CTkInputDialog(
                text="Type 'DELETE' to confirm clearing all saved face data:",
                title="Confirm Clear Data"
            )

            if response.get_input() == "DELETE":
                # Clear embeddings in memory
                self.embeddings = []

                # Delete all saved embedding files
                embeddings_dir = "face_embeddings"
                for file in os.listdir(embeddings_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(embeddings_dir, file))

                self.update_status("All saved face data has been cleared")
        else:
            self.update_status("No saved face data to clear")

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceAuthApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()