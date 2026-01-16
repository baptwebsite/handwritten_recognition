# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 2026
@author: veroe

Random Forest Test Script: 
Loads the model, processes test images via a CSV file, 
and exports a comparison CSV (Image, Truth, Prediction).
"""

import numpy as np
import os
import cv2
import joblib
import csv

# --------------------------------------------------
# 1. Load Model and Scaler
# --------------------------------------------------
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "random_forest_scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Random Forest model and Scaler successfully loaded.")
else:
    print("Error: Model or Scaler file not found.")
    exit()

# --------------------------------------------------
# 2. Feature Extraction (18 features)
# --------------------------------------------------
def extract_handcrafted_features(img):
    """
    Extract a fixed-length feature vector (18 features) 
    compatible with the trained RF model.
    """
    if img is None:
        return None

    # Size normalization
    img = cv2.resize(img, (32, 32))
    
    # Binarization
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    c = max(contours, key=cv2.contourArea)
    
    # Hu Moments (7 features)
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # Shape descriptors (11 features)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    (x, y, w, h) = cv2.boundingRect(c)
    aspect_ratio = w / h if h > 0 else 0.0
    extent = area / (w * h) if w * h > 0 else 0.0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    (_, _), radius = cv2.minEnclosingCircle(c)
    diameter = 2 * radius
    roundness = 4 * area / (np.pi * diameter ** 2) if diameter > 0 else 0.0
    
    # Gradient statistics
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_mean = np.mean(magnitude)
    grad_std = np.std(magnitude)
    
    # Curvature statistics
    c_pts = c[:,0,:].astype(float)
    x_smooth = np.convolve(c_pts[:,0], np.ones(3)/3, mode='same')
    y_smooth = np.convolve(c_pts[:,1], np.ones(3)/3, mode='same')
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2 + 1e-10)
    curvature_mean = np.mean(curvature)
    curvature_std = np.std(curvature)
    
    # Final Vector
    feature_vector = np.concatenate([
        hu,
        np.array([area, perimeter, circularity, roundness,
                  aspect_ratio, extent, solidity,
                  grad_mean, grad_std, curvature_mean, curvature_std])
    ])
    
    return feature_vector

# --------------------------------------------------
# 3. Paths and Files
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Input CSV with ground truth (path, label)
INPUT_TEST_CSV = os.path.join(BASE_DIR, "test_labels.csv")
# Output CSV (image_name, label, predicted_label)
OUTPUT_PRED_CSV = "random_forest_predictions.csv"

# --------------------------------------------------
# 4. Processing and Prediction
# --------------------------------------------------
results = []
correct_count = 0

if not os.path.exists(INPUT_TEST_CSV):
    print(f"Error: Input CSV not found at {INPUT_TEST_CSV}")
    exit()

print("Processing test images...")



with open(INPUT_TEST_CSV, mode='r', encoding='utf-8') as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
        img_path = row['path'].replace('\\', '/')
        true_label = row['label'].strip()
        image_name = os.path.basename(img_path)

        # Load image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feat = extract_handcrafted_features(img)

        if feat is not None:
            # Scale and Predict
            feat_scaled = scaler.transform([feat])
            pred_label = clf.predict(feat_scaled)[0]
        else:
            pred_label = "?"

        results.append([image_name, true_label, pred_label])
        if pred_label == true_label:
            correct_count += 1

# --------------------------------------------------
# 5. Export and Summary
# --------------------------------------------------
with open(OUTPUT_PRED_CSV, mode="w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["image_name", "label", "predicted_label"])
    writer.writerows(results)

total = len(results)
accuracy = (correct_count / total) * 100 if total > 0 else 0

print("-" * 40)
print(f"Predictions saved to: {OUTPUT_PRED_CSV}")
print(f"Correct predictions: {correct_count} / {total}")
print(f"Accuracy: {accuracy:.2f}%")
print("-" * 40)