# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import random

"""After removing manually the incorrect sample, we perform a data augmentation."""

# --- CONFIGURATION ---
base_dir = 'data/dataset'  # Directory containing subfolders for each letter (a, b, c...)
aug_factor = 5             # Number of new augmented images to create per original image

def augment_image(img):
    """Applies slight random transformations to simulate handwriting variations."""
    h, w = img.shape[:2]
    
    # 1. Slight random rotation (between -15 and +15 degrees)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    # Use white borderValue (255) to match the background
    img_aug = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # 2. Random Zoom (between 0.9 and 1.1)
    zoom = random.uniform(0.9, 1.1)
    new_w, new_h = int(w * zoom), int(h * zoom)
    img_zoomed = cv2.resize(img_aug, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a blank white canvas to maintain original dimensions
    final_img = np.full((h, w), 255, dtype=np.uint8)
    
    # Center the zoomed image on the destination canvas
    start_y = max(0, (h - new_h) // 2)
    start_x = max(0, (w - new_w) // 2)
    
    # Calculate crop coordinates if zoom > 1
    src_y = max(0, (new_h - h) // 2)
    src_x = max(0, (new_w - w) // 2)
    
    h_overlap = min(h, new_h)
    w_overlap = min(w, new_w)
    
    final_img[start_y:start_y+h_overlap, start_x:start_x+w_overlap] = \
        img_zoomed[src_y:src_y+h_overlap, src_x:src_x+w_overlap]

    # 3. Slight translation (shifting pixels)
    tx = random.randint(-3, 3)
    ty = random.randint(-3, 3)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    final_img = cv2.warpAffine(final_img, M_trans, (w, h), borderValue=255)

    return final_img

# --- EXECUTION ---
print("Starting data augmentation...")

# Iterate through each letter folder in the dataset
for letter_folder in os.listdir(base_dir):
    letter_path = os.path.join(base_dir, letter_folder)
    if not os.path.isdir(letter_path):
        continue
    
    # List original images (excluding those already containing 'aug' in the name)
    original_images = [f for f in os.listdir(letter_path) if f.endswith('.png') and 'aug' not in f]
    
    for img_name in original_images:
        img_full_path = os.path.join(letter_path, img_name)
        img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        # Generate 'aug_factor' number of variations
        for i in range(aug_factor):
            img_aug = augment_image(img)
            # Save with a specific suffix to distinguish from originals
            new_name = f"{img_name.split('.')[0]}_aug_{i}.png"
            cv2.imwrite(os.path.join(letter_path, new_name), img_aug)

print(f"Finished! Each letter now has approximately {len(original_images) * (aug_factor + 1)} samples.")