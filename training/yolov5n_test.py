from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# Function to open file dialog and select image
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", ".jpg;.jpeg;*.png")])
    return file_path

# Load YOLO model
model = YOLO("best.pt")

# Get image path from user
image_path = select_image()

if image_path:
    # Perform prediction
    results = model.predict(source=image_path, show=True, save=True, conf=0.5)

    # Print detected classes and confidence
    for result in results:
        for box in result.boxes:
            print(f"\nPredicted Stage: {model.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}")
else:
    print("No image selected. Exiting.")