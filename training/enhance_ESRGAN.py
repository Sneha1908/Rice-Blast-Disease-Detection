import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Set up directories
input_folder = r'E:\REVIEW_2\FINAL DATASETS\input_images'  # Specify your input folder
output_folder = r'E:\REVIEW_2\FINAL DATASETS\enhanced_images'  # Specify your output folder
os.makedirs(output_folder, exist_ok=True)

# Initialize the Real-ESRGAN model
def initialize_model():
    # RRDBNet model configuration
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = r'E:\REVIEW_2\models\RealESRGAN_x4plus.pth'
    # Load the Real-ESRGANer
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()  # Use half precision if GPU is available
    )
    return upsampler

# Enhance image
def enhance_image(input_path, output_path, upsampler):
    # Read input image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform enhancement
    with torch.no_grad():
        output, _ = upsampler.enhance(img, outscale=4)
    
    # Convert enhanced image back to BGR and save
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)
    return output

# Main workflow
def main():
    # Initialize model
    upsampler = initialize_model()

    # Process each image in the input folder
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, f"enhanced_{file_name}")
        
        # Enhance and save the image
        enhanced_image = enhance_image(input_path, output_path, upsampler)
        
        # Display original and enhanced images side-by-side
        original = cv2.imread(input_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        enhanced = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(enhanced)
        plt.title("Enhanced Image")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()

# Run the script
if __name__ == "__main__":
    main()
