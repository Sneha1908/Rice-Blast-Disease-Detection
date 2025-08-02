import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from tkinter import Tk, filedialog  # For file upload

# Import ESRGAN components
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ----- CONFIGURATION -----
classifier_model_path = 'models/rice_leaf_classifier_model_final.keras'
esrgan_model_path = 'models/RealESRGAN_x4plus.pth'  # Path to Real-ESRGAN model
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# ----- CLASS LABELS -----
class_names = ['NOT A LEAF', 'NOT A RICE LEAF', 'RICE LEAF']

remedies = {
    '0': (
        "Healthy leaf. No visible lesions.\n"
        "Maintain regular monitoring every 3–4 days. Ensure proper spacing between plants to reduce humidity. "
        "Follow integrated nutrient management to maintain crop vigor."
    ),
    '1': (
        "Initial infection with minute gray-green specks.\n"
        "Use preventive measures like foliar spray of Tricyclazole @ 0.6 g/L. "
        "Avoid excessive nitrogen fertilizers."
    ),
    '3': (
        "Spindle-shaped spots appear on leaves.\n"
        "Apply Tricyclazole 75WP @ 600 g/ha or Azoxystrobin + Tebuconazole @ 1 ml/L. "
        "Maintain AWD irrigation."
    ),
    '5': (
        "Large lesions causing leaf damage.\n"
        "Recommendation: Apply Isoprothiolane 40% EC @ 1.5 L/ha. "
        "Avoid late planting. Strengthen potassium application."
    ),
    '7': (
        "Severe infection, 40%+ leaf affected.\n"
        "Isolate infected patches. Spray systemic fungicides like Edifenphos. "
        "Improve field aeration."
    ),
    '9': (
        "Terminal stage – necrosis and yield loss.\n"
        "Discard infected plants. Treat seeds for next season."
        "Use resistant varieties like CO51."
    )
}

# ----- STEP 1: CLASSIFICATION -----
def classify_image(image_path):
    print(f"\n📷 Classifying image: {image_path}")
    model = load_model(classifier_model_path)
    
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100
    label = class_names[predicted_class]

    print(f"\nPrediction: {label} ({confidence:.2f}%)")
    return label, confidence

# ----- STEP 2: ENHANCEMENT -----
def enhance_image_with_esrgan(image_path):
    print("\n🧪 Enhancing image using Real-ESRGAN...")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=esrgan_model_path,
        model=model,
        tile=200,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Optional downscaling
    max_dim = 500
    h, w = img_rgb.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    try:
        output_img, _ = upsampler.enhance(img_rgb, outscale=4)
        enhanced_path = os.path.join(output_dir, 'enhanced.png')
        cv2.imwrite(enhanced_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        print(f"✅ Enhanced image saved to: {enhanced_path}")

        # Show original vs enhanced
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img_rgb)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        axs[1].imshow(output_img)
        axs[1].set_title("Enhanced Image (ESRGAN)")
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

        return enhanced_path
    except RuntimeError as e:
        print(f"❌ Enhancement failed: {e}")
        return None

# ----- STEP 3: YOLOv5 DETECTION -----
def detect_stage_with_yolov5(image_path):
    print("\n🔍 Detecting rice blast stage using YOLOv5...")
    model = YOLO("models/best.pt")
    results = model.predict(source=image_path, save=True, conf=0.25, project='outputs', name='predict', exist_ok=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls)
            conf = box.conf.item()
            class_name = model.names[cls_id]
            print(f"\n🔬 DETECTED STAGE : {class_name} ({conf*100:.2f}%)")
            print(f"\n🌾 RECOMMENDED ACTION:\n {remedies.get(str(class_name), 'No advice available.')}\n")

    # Match annotated output image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    matches = glob.glob(os.path.join('outputs', 'predict', base_name + ".*"))

    if matches:
        stage_image_path = matches[0]
        detected_img = PILImage.open(stage_image_path)
        plt.imshow(detected_img)
        plt.title("YOLOv5 - Detected Stage(s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"❌ Annotated output not found for {base_name}")

# ----- MAIN EXECUTION -----
if __name__ == "__main__":
    # Open a file dialog to choose an image
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(title="Select a rice leaf image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
    if not file_path:
        print("❌ No file selected. Exiting.")
        exit()

    # Step 1: Classification
    label, confidence = classify_image(file_path)

    # Step 2: Validation
    if label != 'RICE LEAF':
        print(f"\n🚫 The uploaded image is classified as: {label} ({confidence:.2f}%)")
        print("💡 Please upload a valid rice leaf image.")
        exit()

    print("✅ Valid rice leaf image. Proceeding to enhancement...")

    # Step 3: Enhancement using ESRGAN
    enhanced_path = enhance_image_with_esrgan(file_path)
    if not enhanced_path:
        print("⚠️ Enhancement failed. Proceeding with original image.")
        enhanced_path = file_path

    print("✅ Proceeding to detection...")

    # Step 4: YOLOv5 Detection
    detect_stage_with_yolov5(enhanced_path)
