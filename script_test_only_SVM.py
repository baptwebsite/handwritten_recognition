# -*- coding: utf-8 -*-
"""
SVM classification of handwritten characters + export CSV

Uses the same handcrafted features as the training script.
"""

import numpy as np
import os
import cv2
import csv
import joblib

# --------------------------------------------------
# Feature extraction function (MUST MATCH TRAINING)
# --------------------------------------------------
def extract_handcrafted_features(img):
    """
    Extract a fixed-length feature vector (11 features)
    from a grayscale character image.
    """
    # Size normalization
    img = cv2.resize(img, (32, 32))

    # Binarization and contour extraction
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Keep the largest connected component (the character itself)
    c = max(contours, key=cv2.contourArea)

    # Shape descriptors
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
    diameter = 2 * radius
    roundness = 4 * area / (np.pi * diameter ** 2) if diameter > 0 else 0.0

    # Gradient statistics (Sobel filters)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_mean = np.mean(magnitude)
    grad_std = np.std(magnitude)

    # Curvature statistics
    c_pts = c[:, 0, :].astype(float)
    # Apply smoothing to reduce noise in curvature calculation
    x_smooth = np.convolve(c_pts[:, 0], np.ones(3)/3, mode="same")
    y_smooth = np.convolve(c_pts[:, 1], np.ones(3)/3, mode="same")
    
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10)
    curvature_mean = np.mean(curvature)
    curvature_std = np.std(curvature)

    # Final feature vector (dimension = 11)
    feature_vector = np.array([
        area,
        perimeter,
        circularity,
        roundness,
        aspect_ratio,
        extent,
        solidity,
        grad_mean,
        grad_std,
        curvature_mean,
        curvature_std
    ])

    return feature_vector

# --------------------------------------------------
# Paths and Load Model
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The CSV you provided (path,label)
INPUT_TEST_CSV = os.path.join(BASE_DIR, "test_labels.csv") 
# The new CSV to create
OUTPUT_PRED_CSV = "svm_predictions.csv"

if os.path.exists("svm_model.pkl") and os.path.exists("svm_scaler.pkl"):
    svm_clf = joblib.load("svm_model.pkl")
    scaler = joblib.load("svm_scaler.pkl")
else:
    print("Error: Model files not found.")
    exit()

# --------------------------------------------------
# Process and Predict
# --------------------------------------------------
results = []
correct_count = 0

if not os.path.exists(INPUT_TEST_CSV):
    print(f"Error: Input CSV not found at {INPUT_TEST_CSV}")
    exit()

print("Reading test CSV and predicting...")

with open(INPUT_TEST_CSV, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = row['path'].replace('\\', '/')
        true_label = row['label'].strip()
        
        # Get just the filename for the output CSV (e.g., test_a.png)
        image_name = os.path.basename(img_path)

        # Load and process image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feat = extract_handcrafted_features(img)

        if feat is not None:
            feat_scaled = scaler.transform([feat])
            pred_label = svm_clf.predict(feat_scaled)[0]
        else:
            pred_label = "?"

        # Track results
        results.append([image_name, true_label, pred_label])
        if pred_label == true_label:
            correct_count += 1

# --------------------------------------------------
# Save to New CSV
# --------------------------------------------------
with open(OUTPUT_PRED_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label", "predicted_label"])
    writer.writerows(results)

# --------------------------------------------------
# Final Metrics
# --------------------------------------------------
total = len(results)
accuracy = (correct_count / total) * 100 if total > 0 else 0

print("-" * 30)
print(f"Results saved to: {OUTPUT_PRED_CSV}")
print(f"Total processed: {total}")
print(f"Correct predictions: {correct_count}")
print(f"Test Accuracy: {accuracy:.2f}%")
print("-" * 30)