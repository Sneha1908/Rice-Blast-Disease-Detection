# 🌾 Rice Blast Disease Detection System

An AI-powered deep learning system for **early detection and stage-wise classification** of rice blast disease from rice leaf images. This precision farming tool integrates:

-  A lightweight CNN classifier (EfficientNetB0) to validate input as Rice Leaf / Non-Rice Leaf / Non-Leaf  
-  YOLOv5n model to detect infected regions and classify disease progression into **six stages** (0, 1, 3, 5, 7, 9)  
-  ESRGAN-based image enhancer for improved visibility of subtle lesions  
-  A Flutter mobile application for real-time predictions on device  
-  Flask-based backend API to handle detection and communication

The system improves accuracy and reduces delays in disease management by replacing manual inspection with automated image-based analysis.

---

## ✨ Key Features

-  **CNN-Based Leaf Classification**  
  Classifies images into: `RICE LEAF`, `NOT A RICE LEAF`, or `NOT A LEAF` using EfficientNetB0.

-  **YOLOv5-Based Stage Detection**  
  Identifies infected areas and classifies rice blast disease into six progression stages: 0, 1, 3, 5, 7, 9.

-  **Image Enhancement (ESRGAN)**  
  Improves low-resolution or blurry leaf images using super-resolution techniques.

-  **Mobile App (Flutter)**  
  Allows farmers to upload images directly from their phone and view disease stage and treatment advice in real time.

-  **Flask Backend API**  
  Handles classification, enhancement, and detection workflows.

-  **Stage-Based Recommendations**  
  Returns specific agricultural advice based on the detected stage of infection.

---

## 🏗️ System Architecture

This project follows a multi-stage pipeline integrating classification, enhancement, detection, and mobile interface modules.

### 🧩 Component Flow
User → Mobile App → Flask API → [CNN Classifier → ESRGAN Enhancer → YOLOv5 Detector] → Response (Stage + Recommendation)

### 🔁 Architecture Layers

1. **Input Layer (Flutter App)**
   - Users upload leaf images through the mobile interface.
   - Sends the image via HTTP POST to the backend API.

2. **CNN Classification (EfficientNetB0)**
   - Validates if the image is a `RICE LEAF`, `NON-RICE LEAF`, or `NON-LEAF`.
   - Rejects invalid inputs before moving to detection.

3. **Image Enhancement (Real-ESRGAN)**
   - Optional enhancement step.
   - Improves clarity of leaf texture and lesion visibility.

4. **YOLOv5 Stage Detection**
   - Runs object detection to locate disease lesions.
   - Classifies into one of six progression stages (0 to 9).
   - Annotates the image and returns stage-wise prediction with confidence.

5. **Treatment Recommendation Engine**
   - Based on detected stage, the system returns a customized advice message (e.g., fungicide dosage, irrigation suggestions).

6. **Frontend Response**
   - The app displays:
     - Predicted class
     - Stage (if applicable)
     - Confidence
     - Annotated output image
     - Recommendation
    
---

## 🛠 Technologies Used

| Category             | Tools / Libraries                                                   |
|----------------------|---------------------------------------------------------------------|
| **Model Training**   | TensorFlow, Keras, EfficientNetB0                                   |
| **Object Detection** | YOLOv5n (via Ultralytics)                                           |
| **Image Enhancement**| ESRGAN (RealESRGAN, RRDBNet, PyTorch, basicsr)                      |
| **Computer Vision**  | OpenCV, Pillow                                                      |
| **API Development**  | Flask, Werkzeug                                                     |
| **Mobile App**       | Flutter, Dart                                                       |
| **Data Handling**    | NumPy, JSON                                                         |
| **Visualization**    | Matplotlib, Seaborn                                                 |
| **Evaluation**       | scikit-learn (accuracy, precision, recall, F1 score, confusion matrix) |

---

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sneha1908/Rice-Blast-Disease-Detection.git
cd Rice-Blast-Disease-Detection
```
### 2. (Optional) Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 4. Download Trained Models

All required models are available in the [GitHub Releases](https://github.com/Sneha1908/Rice-Blast-Disease-Detection/releases/tag/v1.0) section.

| Model File                          | Purpose                                     | Download Link |
|------------------------------------|---------------------------------------------|---------------|
| `best.zip`                         | YOLOv5n model for stage-wise detection      | [Download](https://github.com/Sneha1908/Rice-Blast-Disease-Detection/releases/download/v1.0/best.zip) |
| `rice_leaf_classifier_model_final.zip` | EfficientNetB0 classifier for leaf validation | [Download](https://github.com/Sneha1908/Rice-Blast-Disease-Detection/releases/download/v1.0/rice_leaf_classifier_model_final.zip) |
| `RealESRGAN_x4plus.zip`            | Enhanced Super-resolution Generative Adversial Network Model (ESRGAN enhancer)    | [Download](https://github.com/Sneha1908/Rice-Blast-Disease-Detection/releases/download/v1.0/RealESRGAN_x4plus.zip) |

> 📁 After downloading, extract all `.zip` files into a folder named `models/` in your project directory before running the app.
 
### 5. Run the Flask Backend
```bash
python app.py
```
The API will start at:
http://localhost:5000/

   - Available endpoints:
     - POST /classify → Validates if the image is a rice leaf
     - POST /detect → Detects disease stage and returns results
     - GET /static/outputs/<filename> → Serves annotated output image
### 6. Run the Mobile App (Flutter)
- Open the rice_blast_app folder in VS Code or Android Studio
- Run flutter pub get
- Connect a physical or virtual device

Run:
```bash
flutter run
```
---

## 📷 Results

All result screenshots are available in the `/Results/` folder of this repository.

###  System Architecture

- System Design Overview  
  ![System Design](Results/System%20Design.png)

- Architecture Flow  
  ![System Architecture](Results/System%20Architecture.png)

###  Model Training & Evaluation (EfficientNetB0)

- Accuracy and Loss Plot  
  ![Accuracy & Loss](Results/EfficientNetB0%20-%20Accuracy%20nd%20Loss%20Graph.png)

- Classification Report  
  ![Classification Report](Results/EfficientNetB0%20-%20Classification%20Report.png.jpg)

- Confusion Matrix  
  ![Confusion Matrix](Results/EfficientNetB0%20-%20Confusion%20Matrix.png.jpg)

- Sample Predictions  
  ![Prediction Results](Results/EfficientNetB0%20-%20Prediction%20Results.png.jpg)

###  Image Enhancement (ESRGAN)

- Enhanced vs. Original  
  ![ESRGAN Results](Results/Image%20Enhancement%20-%20ESRGAN%20Results.png)

###  Disease Detection with YOLOv5n

- YOLO Evaluation Graphs  
  ![YOLO Graphs](Results/YOLOv5n%20-%20Evaluation%20Graphs.png)

- YOLO Evaluation Metrics  
  ![YOLO Metrics](Results/YOLOv5n%20-%20Evaluation%20Metrics.png)

- Precision-Recall Curve  
  ![YOLO PR Curve](Results/YOLOv5n%20-%20Precision%20Recall%20Curve.png)

- Prediction Output Sample 1  
  ![YOLO Result 1](Results/YOLOv5n%20-%20Prediction%20Result%201.png.jpg)

- Prediction Output Sample 2  
  ![YOLO Result 2](Results/YOLOv5n%20-%20Prediction%20Result%202.png.jpg)

###  Flutter Mobile App UI

- Real-time Disease Detection on Phone  
  ![Mobile App](Results/Mobile%20App%20Results.png)





