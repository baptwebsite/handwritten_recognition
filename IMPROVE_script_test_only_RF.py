# -*- coding: utf-8 -*-
"""
Created on Jan 16 2026
@author: Gemini & Veroe

Improved Random Forest Test Script (HOG + Geometric)
Processes test images via a CSV file and exports comparison results.
"""

import numpy as np
import os
import cv2
import joblib
import csv
from skimage.feature import hog

# --------------------------------------------------
# 1. Load Improved Model and Scaler
# --------------------------------------------------
# Make sure these are the ones trained with HOG!
MODEL_PATH = "IMPROVE_random_forest_model.pkl" 
SCALER_PATH = "IMPROVE_random_forest_scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Improved RF model (HOG + 18 features) and Scaler loaded.")
else:
    print(f"Error: Improved model files not found at {MODEL_PATH}")
    exit()

# --------------------------------------------------
# 2. Advanced Feature Extraction (162 features)
# --------------------------------------------------
def extract_advanced_features(img):
    """
    Extracts 162 features: 144 HOG + 7 Hu Moments + 11 Geometric descriptors.
    MUST MATCH TRAINING LOGIC.
    """
    if img is None: return None
    
    # Size normalization
    img = cv2.resize(img, (32, 32))
    
    # --- A. HOG Features (144) ---
    hog_feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=False)

    # --- B. Geometric & Hu (18) ---
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.concatenate([hog_feat, np.zeros(18)])
    
    c = max(contours, key=cv2.contourArea)
    
    # Hu Moments (7)
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # Shape descriptors (11)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    asp = w / h if h > 0 else 0.0
    ext = area / (w * h) if w * h > 0 else 0.0
    sol = area / cv2.contourArea(cv2.convexHull(c)) if cv2.contourArea(cv2.convexHull(c)) > 0 else 0.0
    circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0.0
    (_, _), rad = cv2.minEnclosingCircle(c)
    rnd = 4 * area / (np.pi * (rad*2)**2) if rad > 0 else 0.0
    
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    g_m, g_s = np.mean(mag), np.std(mag)

    try:
        c_pts = c[:, 0, :].astype(float)
        dx, dy = np.gradient(c_pts[:, 0]), np.gradient(c_pts[:, 1])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        curv = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2 + 1e-10)
        c_m, c_s = np.mean(curv), np.std(curv)
    except:
        c_m, c_s = 0.0, 0.0

    geom_vector = np.array([area, peri, circ, rnd, asp, ext, sol, g_m, g_s, c_m, c_s])
    
    # Combine HOG + Hu + Geom
    return np.concatenate([hog_feat, hu, geom_vector])

# --------------------------------------------------
# 3. Paths and Files
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TEST_CSV = os.path.join(BASE_DIR, "test_labels.csv")
OUTPUT_PRED_CSV = "improved_rf_predictions.csv"

# --------------------------------------------------
# 4. Processing and Prediction
# --------------------------------------------------
results = []
correct_count = 0

if not os.path.exists(INPUT_TEST_CSV):
    print(f"Error: Input CSV not found at {INPUT_TEST_CSV}")
    exit()

print("Processing test images with Improved RF (HOG)...")

with open(INPUT_TEST_CSV, mode='r', encoding='utf-8') as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
        img_path = row['path'].replace('\\', '/')
        true_label = row['label'].strip()
        image_name = os.path.basename(img_path)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feat = extract_advanced_features(img)

        if feat is not None:
            # Scaler needs 2D input: (1, 162)
            feat_scaled = scaler.transform(feat.reshape(1, -1))
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

print("-" * 45)
print(f"Improved Results saved to: {OUTPUT_PRED_CSV}")
print(f"Correct predictions: {correct_count} / {total}")
print(f"New Improved Accuracy: {accuracy:.2f}%")
print("-" * 45)