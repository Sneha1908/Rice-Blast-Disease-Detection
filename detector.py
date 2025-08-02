import os
from ultralytics import YOLO
import cv2

MODEL_PATH = os.path.join('models', 'best.pt')
model = YOLO(MODEL_PATH)

STAGE_RECOMMENDATIONS = {
    '0': (
        "Stage 0: Healthy leaf. No visible lesions.\n"
        "Recommendation: Maintain regular monitoring every 3–4 days. Ensure proper spacing between plants to reduce humidity. "
        "Follow integrated nutrient management to maintain crop vigor."
    ),
    '1': (
        "Stage 1: Initial infection with minute gray-green specks.\n"
        "Recommendation: Use preventive measures like foliar spray of Tricyclazole @ 0.6 g/L. Avoid excessive nitrogen fertilizers."
    ),
    '3': (
        "Stage 3: Spindle-shaped spots appear on leaves.\n"
        "Recommendation: Apply Tricyclazole 75WP @ 600 g/ha or Azoxystrobin + Tebuconazole @ 1 ml/L. Maintain AWD irrigation."
    ),
    '5': (
        "Stage 5: Large lesions causing leaf damage.\n"
        "Recommendation: Apply Isoprothiolane 40% EC @ 1.5 L/ha. Avoid late planting. Strengthen potassium application."
    ),
    '7': (
        "Stage 7: Severe infection, 40%+ leaf affected.\n"
        "Recommendation: Isolate infected patches. Spray systemic fungicides like Edifenphos. Improve field aeration."
    ),
    '9': (
        "Stage 9: Terminal stage – necrosis and yield loss.\n"
        "Recommendation: Discard infected plants. Treat seeds for next season. Use resistant varieties like CO51."
    )
}

def detect_stage(image_path, output_dir="static/outputs"):
    results = model(image_path)[0]
    img = cv2.imread(image_path)
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({conf*100:.1f}%)"
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        detections.append({
            "stage": f"Stage {label}",
            "confidence": f"{conf*100:.2f}%",
            "recommendation": STAGE_RECOMMENDATIONS.get(label, "No recommendation available.")
        })

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"detection_{os.path.basename(image_path)}"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img)
    return f"static/outputs/{output_filename}", detections
