import os
import torch
from realesrgan import RealESRGAN
from PIL import Image

def enhance_image(input_path, output_path=None):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load image
        image = Image.open(input_path).convert("RGB")

        # Load RealESRGAN model
        model = RealESRGAN(device, scale=4)
        model.load_weights('models/RealESRGAN_x4plus.pth')  # ✅ Make sure this path is correct

        # Enhance image
        sr_image = model.predict(image)

        # Output path
        if output_path is None:
            base_name = os.path.basename(input_path)
            output_path = os.path.join('outputs', f'enhanced_{base_name}')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save enhanced image
        sr_image.save(output_path)
        return output_path

    except Exception as e:
        print(f"[❌ ERROR] Enhancement failed: {e}")
        return None
