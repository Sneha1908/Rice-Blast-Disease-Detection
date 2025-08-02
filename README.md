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
### 4. Download Model Files
Download the trained models from models.zip file from the Releases page and extract it into the models/ folder.

   - Included models:
     - rice_leaf_classifier_model_final.keras
     - best.pt
     - RealESRGAN_x4plus.pth
       
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

All resukt screenshots are available in the `/results/` folder of this repository.

---

### 🏗️ System Architecture

- System Design Overview  
  ![System Design](screenshots/System Design.png)

- Architecture Flow  
  ![System Architecture](screenshots/System Architecture.png)

---

### 🧠 Model Training & Evaluation

- EfficientNetB0: Accuracy & Loss  
  ![Accuracy Loss](screenshots/EfficientNetB0 - Accuracy nd Loss Graph.png)

- Classification Report  
  ![Classification Report](screenshots/EfficientNetB0 - Classification Report.png)

- Confusion Matrix  
  ![Confusion Matrix](screenshots/EfficientNetB0 - Confusion Matrix.png)

- Prediction Samples  
  ![Prediction](screenshots/EfficientNetB0 - Prediction Results.png)

---

### 🖼️ Image Enhancement (ESRGAN)

- Before and After Comparison  
  ![Enhancement](screenshots/Image Enhancement - ESRGAN Results.png)

---

### 🎯 YOLOv5n Detection

- Evaluation Graphs  
  ![YOLO Graphs](screenshots/YOLOv5n - Evaluation Graphs.png)

- Evaluation Metrics Table  
  ![Metrics Table](screenshots/YOLOv5n - Evaluation Metrics.png)

- Precision-Recall Curve  
  ![PR Curve](screenshots/YOLOv5n - Precision Recall Curve.png)

- Prediction Result Samples  
  ![YOLO Prediction 1](screenshots/YOLOv5n - Prediction Result 1.png)  
  ![YOLO Prediction 2](screenshots/YOLOv5n - Prediction Result 2.png)

---

### 📱 Mobile App (Flutter)

- App UI and Results View  
  ![App UI](screenshots/Mobile App Results.png)










