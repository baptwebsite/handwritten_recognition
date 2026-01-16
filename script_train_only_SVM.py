# -*- coding: utf-8 -*-
"""
Train an SVM classifier for handwritten character recognition
using handcrafted geometric, gradient, and curvature features.

IMPORTANT:
- The SAME feature extraction function must be used at training and testing time.
- The SAME features are also been used here for the Random Forest model.
"""

import numpy as np
import os
import cv2
import csv
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------
# Feature extraction function
# --------------------------------------------------
def extract_handcrafted_features(img):
    """
    Extract a fixed-length feature vector (11 features)
    from a grayscale character image.

    Returns:
        feature_vector : np.ndarray of shape (11,)
    """

    # -----------------------------
    # Size normalization
    # -----------------------------
    img = cv2.resize(img, (32, 32))

    # -----------------------------
    # Binarization and contour extraction
    # -----------------------------
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Keep the largest connected component
    c = max(contours, key=cv2.contourArea)

    # -----------------------------
    # Shape descriptors
    # -----------------------------
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

    # -----------------------------
    # Gradient-based statistics
    # -----------------------------
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    grad_mean = np.mean(magnitude)
    grad_std = np.std(magnitude)

    # -----------------------------
    # Curvature statistics
    # -----------------------------
    c_pts = c[:, 0, :].astype(float)

    x_smooth = np.convolve(c_pts[:, 0], np.ones(3)/3, mode="same")
    y_smooth = np.convolve(c_pts[:, 1], np.ones(3)/3, mode="same")

    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10)
    curvature_mean = np.mean(curvature)
    curvature_std = np.std(curvature)

    # -----------------------------
    # Final feature vector (dim = 11)
    # -----------------------------
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
# Paths and data loading
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "data/dataset")
CSV_PATH = os.path.join(IMAGES_DIR, "dataset_labels.csv")

features = []
labels = []

# --------------------------------------------------
# Read CSV and extract features
# --------------------------------------------------
with open(CSV_PATH, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # img_path = os.path.join(IMAGES_DIR, row['path'].strip())
        img_path = row["path"]
        img_path.replace('\\', '/')
        label = row['label'].strip()

        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feat = extract_handcrafted_features(img)

        if feat is not None:
            features.append(feat)
            labels.append(label)

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# --------------------------------------------------
# Feature normalization (MANDATORY for SVM)
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # to avoid dominance of certain feature (e.g area compared to curvature)

# --------------------------------------------------
# Train SVM classifier
# --------------------------------------------------
svm_clf = SVC(
    kernel="rbf", # for non linear frontier
    C=10,
    gamma="scale", # local influence of points
    class_weight="balanced", # rare classes have equivalent influence
    probability=False,
    random_state=42
)

svm_clf.fit(X_scaled, y) # here soft-margin SVM (optim) with X_scaled : normalized feature vectors and y = labels

# --------------------------------------------------
# Save model and scaler
# --------------------------------------------------
joblib.dump(svm_clf, "svm_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")


