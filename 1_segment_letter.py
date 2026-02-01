# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
SRC_DIR = "data1/raw_data"        # Source folder containing raw photos (e.g., 'A.jpg')
DEST_DIR = "data1/extracted_chars" # Destination folder for the 21 extracted samples
TARGET_SIZE = 64                   # Final output dimensions (64x64)
NB_EXAMPLES = 21                   # Number of characters to extract per image

def segment_letter(letter):
    """Extracts 21 samples of a specific character from its raw source image."""
    img_path = os.path.join(SRC_DIR, f"{letter}.jpg")
    
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return

    # 1. Load image and convert to grayscale
    image = cv2.imread(img_path)
    if image is None:
        print(f"Read error: {img_path}")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Preprocessing & Binarization (Adaptive to handle varying lighting conditions)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 8)

    # 3. Morphology to bridge gaps in thin pen strokes (Closing operation)
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. Contour Detection
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area to select the 21 largest objects (the characters)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    candidate_letters = sorted_contours[:NB_EXAMPLES]

    # Create destination directory (e.g., data/extracted_chars/A/)
    letter_folder = os.path.join(DEST_DIR, letter)
    os.makedirs(letter_folder, exist_ok=True)

    print(f"Processing '{letter}': {len(candidate_letters)} contours identified.")

    for i, cnt in enumerate(candidate_letters):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Extract with a safety margin (padding)
        margin = 15
        y1, y2 = max(0, y-margin), min(image.shape[0], y+h+margin)
        x1, x2 = max(0, x-margin), min(image.shape[1], x+w+margin)
        roi = processed[y1:y2, x1:x2]

        if roi.size == 0: continue

        # 5. Final Formatting: 64x64 WHITE background canvas
        final_img = np.full((TARGET_SIZE, TARGET_SIZE), 255, dtype=np.uint8)
        
        # Calculate scale ratio so the character occupies approx. 75% of the frame
        h_roi, w_roi = roi.shape
        ratio = min((TARGET_SIZE-16)/w_roi, (TARGET_SIZE-16)/h_roi)
        new_w, new_h = int(w_roi * ratio), int(h_roi * ratio)
        
        if new_w > 0 and new_h > 0:
            roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Inversion: Convert from White-on-Black (processed) to BLACK-on-WHITE
            roi_final = cv2.bitwise_not(roi_resized)

            # Center the character within the 64x64 canvas
            off_y = (TARGET_SIZE - new_h) // 2
            off_x = (TARGET_SIZE - new_w) // 2
            final_img[off_y:off_y+new_h, off_x:off_x+new_w] = roi_final

            # Save the result
            output_filename = f"{letter}_{i}.png"
            cv2.imwrite(os.path.join(letter_folder, output_filename), final_img)

# --- EXECUTION ---
# List of characters to process based on your raw dataset
alphabet = "abcdefghijklmnopqrstuvwxyz"

for char in alphabet:
    segment_letter(char)

print("\nExtraction complete. Files saved in:", DEST_DIR)