# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import random

# --- CONFIGURATION ---
base_dir = 'data/extracted_chars'   # Source folder with cleaned images
output_dir = 'data/dataset'         # Target folder for the final augmented dataset
aug_factor = 10                      # Number of synthetic variations per original image

def augment_image(img):
    """Applies slight random transformations to simulate handwriting variations."""
    h, w = img.shape[:2]
    
    # 1. Rotation: Random angle between -15 and 15 degrees
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    # Using borderValue=255 to maintain a white background after rotation
    img_aug = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # 2. Zoom: Random scaling between 0.9 and 1.1
    zoom = random.uniform(0.9, 1.1)
    new_w, new_h = int(w * zoom), int(h * zoom)
    img_zoomed = cv2.resize(img_aug, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Re-center the zoomed image on a white 64x64 canvas
    final_img = np.full((h, w), 255, dtype=np.uint8)
    start_y, start_x = max(0, (h - new_h) // 2), max(0, (w - new_w) // 2)
    src_y, src_x = max(0, (new_h - h) // 2), max(0, (new_w - w) // 2)
    h_overlap, w_overlap = min(h, new_h), min(w, new_w)
    
    final_img[start_y:start_y+h_overlap, start_x:start_x+w_overlap] = \
        img_zoomed[src_y:src_y+h_overlap, src_x:src_x+w_overlap]

    # 3. Translation: Slight shift of +/- 3 pixels
    tx, ty = random.randint(-3, 3), random.randint(-3, 3)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    final_img = cv2.warpAffine(final_img, M_trans, (w, h), borderValue=255)

    return final_img

# --- EXECUTION ---
print(f"Starting data augmentation from '{base_dir}' to '{output_dir}'...")

# Create the output root directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through each letter subfolder (a, b, c...)
for letter_folder in os.listdir(base_dir):
    src_letter_path = os.path.join(base_dir, letter_folder)
    
    if not os.path.isdir(src_letter_path):
        continue
    
    # Create the corresponding letter subfolder in the output directory
    dest_letter_path = os.path.join(output_dir, letter_folder)
    os.makedirs(dest_letter_path, exist_ok=True)
    
    # List all source .png files
    original_images = [f for f in os.listdir(src_letter_path) if f.endswith('.png')]
    
    print(f"Processing letter '{letter_folder}' ({len(original_images)} source images)...")

    for img_name in original_images:
        img_full_path = os.path.join(src_letter_path, img_name)
        img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        # 1. Save the original image to the new dataset folder first
        cv2.imwrite(os.path.join(dest_letter_path, img_name), img)
        
        # 2. Generate and save 'aug_factor' augmented versions
        for i in range(aug_factor):
            img_aug = augment_image(img)
            new_name = f"{img_name.split('.')[0]}_aug_{i}.png"
            cv2.imwrite(os.path.join(dest_letter_path, new_name), img_aug)

print("-" * 30)
print(f"Finished! Final dataset available in: {output_dir}")
print("-" * 30)