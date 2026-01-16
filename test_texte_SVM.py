# -*- coding: utf-8 -*-
import cv2
import numpy as np
import joblib
import os

# --- CONFIGURATION ---
MODEL_PATH = "svm_model.pkl"
SCALER_PATH = "svm_scaler.pkl" 
IMAGE_TEST = "data/test_texte.jpg"

# Secure Loading
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("SVM Model (11 features) and Scaler loaded successfully.")
else:
    print("Error: .pkl files not found.")
    exit()

def extract_11_features(img):
    """ STRICT extraction of the original 11 features """
    if img is None: return None
    
    # 1. Size Normalization (32x32 to match training)
    img = cv2.resize(img, (32, 32))
    
    # 2. Binarization
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return None
    c = max(contours, key=cv2.contourArea)
    
    # --- THE 11 FEATURES ---
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h > 0 else 0.0
    extent = area / (w * h) if w * h > 0 else 0.0
    hull = cv2.convexHull(c)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0.0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    (_, _), radius = cv2.minEnclosingCircle(c)
    roundness = 4 * area / (np.pi * (radius*2)**2) if radius > 0 else 0.0
    
    # Gradients
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    grad_mean = np.mean(mag)
    grad_std = np.std(mag)
    
    # Curvature
    curvature_mean, curvature_std = 0, 0
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

    # Final vector of size 11
    return np.array([
        area, perimeter, circularity, roundness, aspect_ratio, 
        extent, solidity, grad_mean, grad_std, 
        curvature_mean, curvature_std
    ])

def read_text_with_svm_11(image_path):
    """ Segments the image and uses SVM to recognize text """
    original_img = cv2.imread(image_path)
    if original_img is None: return "Image not found"
    
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 8)
    
    # Extract contours (individual letters)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 80: # Noise filter
            x, y, w, h = cv2.boundingRect(cnt)
            roi = gray[y:y+h, x:x+w]
            
            feat = extract_11_features(roi)
            if feat is not None:
                # Reshape mandatory for a single sample (1, 11)
                feat_2d = feat.reshape(1, -1)
                feat_scaled = scaler.transform(feat_2d)
                label = clf.predict(feat_scaled)[0]
                results.append((x, label))

    # Sort from left to right based on X coordinate
    results.sort(key=lambda item: item[0])
    
    return "".join([item[1] for item in results])

# --- EXECUTION ---
if os.path.exists(IMAGE_TEST):
    print("Analyzing image...")
    text = read_text_with_svm_11(IMAGE_TEST)
    print(f"Detected Text: {text}")
else:
    print(f"File {IMAGE_TEST} not found.")