# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import joblib
from skimage.feature import hog
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load Model and Scaler
# --------------------------------------------------
MODEL_PATH = "model/RF_model.pkl" 
SCALER_PATH = "model/RF_scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    clf = joblib.load(MODEL_PATH)      
    scaler = joblib.load(SCALER_PATH)  
    print("Model and Scaler loaded successfully.")
else:
    print(f"Error: Files not found at {MODEL_PATH} or {SCALER_PATH}")
    exit()

# --------------------------------------------------
# 2. Feature Extraction Logic
# --------------------------------------------------
def extract_features(img_64):
    """ 
    Extracts 162 features.
    Input image (img_64) must be grayscale (Black letter on white background).
    """
    # HOG (Must match your training parameters exactly)
    h_feat = hog(img_64, orientations=9, pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2), visualize=False)

    # Hu Moments & Geometry
    # Temporarily invert the image to locate the character contour (OpenCV expects white on black)
    _, temp_inv = cv2.threshold(img_64, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(temp_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.concatenate([h_feat, np.zeros(18)])
    
    c = max(contours, key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # Morphological features
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    asp = w / h if h > 0 else 0.0
    ext = area / (w * h) if w * h > 0 else 0.0
    sol = area / cv2.contourArea(cv2.convexHull(c)) if cv2.contourArea(cv2.convexHull(c)) > 0 else 0.0
    circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0.0
    (_, _), rad = cv2.minEnclosingCircle(c)
    rnd = 4 * area / (np.pi * (rad*2)**2) if rad > 0 else 0.0
    
    # Gradient features (Sobel)
    gx = cv2.Sobel(img_64, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_64, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    g_m, g_s = np.mean(mag), np.std(mag)
    
    geom = np.array([area, peri, circ, rnd, asp, ext, sol, g_m, g_s, 0.0, 0.0])
    return np.concatenate([h_feat, hu, geom])

# --------------------------------------------------
# 3. Smart Padding (Square 64x64 - BLACK LETTER ON WHITE)
# --------------------------------------------------
def preprocess_for_model(roi, target_size=64):
    """
    Fits the black character onto a square WHITE background.
    """
    h, w = roi.shape
    size = max(h, w)
    
    # Create a WHITE square canvas (255)
    square = np.full((size, size), 255, dtype=np.uint8)
    
    # Center the ROI within the canvas
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = roi
    
    # Final resize to model input size
    return cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)

# --------------------------------------------------
# 4. Main Word Recognition Function
# --------------------------------------------------
def read_word_robust(image_path):
    img = cv2.imread(image_path)
    if img is None: return "Image not found"
    
    # Image cleaning
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding: Extracting characters
    # cv2.THRESH_BINARY produces: black letters on white background
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 4)
    
    # For contour detection, OpenCV works best with white objects on black backgrounds
    # We create an inverted version strictly for segmentation purposes
    inv_thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(inv_thresh, kernel, iterations=1)

    # Identify individual character boundaries
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 4 and h > 8: # Filter out noise
            boxes.append((x, y, w, h))
    
    # Sort boxes from left to right for sequential reading
    boxes.sort(key=lambda b: b[0])

    word = ""
    plt.figure(figsize=(15, 4))
    
    for i, (x, y, w, h) in enumerate(boxes):
        # Extract character from the 'thresh' image (black on white)
        margin = 2
        roi = thresh[max(0, y-margin):y+h+margin, max(0, x-margin):x+w+margin]
        if roi.size == 0: continue
        
        # Format to 64x64 with white background
        formatted_letter = preprocess_for_model(roi, target_size=64)
        
        # Feature Extraction & Scaling
        feat = extract_features(formatted_letter)
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        
        # Model Prediction
        pred = clf.predict(feat_scaled)[0]
        word += pred

        # Visual debug feedback
        plt.subplot(1, len(boxes), i + 1)
        plt.imshow(formatted_letter, cmap='gray')
        plt.title(f"'{pred}'")
        plt.axis('off')

    print(f"RECOGNIZED WORD: {word}")
    plt.suptitle(f"Recognition result: {word}")
    plt.show()
    return word

# Execute Test
read_word_robust('data/test/test.jpg')