# -*- coding: utf-8 -*-
import cv2
import numpy as np
import joblib
import os

# --------------------------------------------------
# 1. Load Model and Scaler (18 features)
# --------------------------------------------------
# Ensure these files exist in your working directory
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "random_forest_scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    print("Error: Model or Scaler files not found. Please train the model first.")
    exit()

def extract_handcrafted_features(img):
    """ 
    Feature extraction identical to training phase (18 features).
    Extracts Hu Moments, Shape descriptors, Gradients, and Curvature.
    """
    # Resize to standard processing size
    img = cv2.resize(img, (32, 32))
    
    # Binarization
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0: 
        return None
    
    # Analyze the largest contour
    c = max(contours, key=cv2.contourArea)
    
    # --- Hu Moments (7 features) ---
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform to handle the large dynamic range of moments
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # --- Shape & Gradients (11 features) ---
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h > 0 else 0.0
    extent = area / (w * h) if w * h > 0 else 0.0
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    
    (_, _), radius = cv2.minEnclosingCircle(c)
    roundness = 4 * area / (np.pi * (radius*2)**2) if radius > 0 else 0.0
    
    # Gradient Magnitude
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    
    # Simplified Curvature for testing
    curvature_mean, curvature_std = 0, 0 # Default values
    try:
        c_pts = c[:,0,:].astype(float)
        dx = np.gradient(c_pts[:,0])
        dy = np.gradient(c_pts[:,1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curv = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2 + 1e-10)
        curvature_mean, curvature_std = np.mean(curv), np.std(curv)
    except: 
        pass

    # Combine all into an 18-element vector
    return np.concatenate([hu, [area, perimeter, circularity, roundness, aspect_ratio, 
                                extent, solidity, np.mean(mag), np.std(mag), 
                                curvature_mean, curvature_std]])

def read_text_from_photo(image_path):
    """
    Loads a full image, segments individual characters, predicts them,
    and reconstructs the text based on horizontal position.
    """
    # Load the image
    original_img = cv2.imread(image_path)
    if original_img is None:
        return "Error: Could not read image."
    
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Segmentation Pipeline (Preprocessing)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 8)
    
    # Morphology to bridge small gaps in handwritten strokes
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find character contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_chars = []

    for cnt in contours:
        # Filter small noise based on area
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Extract the Region of Interest (ROI) from the grayscale image
            roi = gray[y:y+h, x:x+w]
            
            # Prediction logic
            feat = extract_handcrafted_features(roi)
            if feat is not None:
                # Normalize features and predict label
                feat_scaled = scaler.transform(feat.reshape(1, -1))
                label = clf.predict(feat_scaled)[0]
                
                # Store the X position along with the label to sort text correctly
                detected_chars.append((x, label))

    # SORTING: Arrange detected characters from left to right (X position)
    detected_chars.sort(key=lambda x: x[0])
    
    # Join labels into final string
    final_text = "".join([item[1] for item in detected_chars])
    return final_text

# --- EXECUTION ---
test_image_path = "data/test_texte.jpg" # Update with your test image filename

if os.path.exists(test_image_path):
    recognized_text = read_text_from_photo(test_image_path)
    print(f"Recognized Text: {recognized_text}")
else:
    print(f"Test photo not found: {test_image_path}")