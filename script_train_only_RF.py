# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import csv
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Feature extraction function (MODIFIED TO 18 FEATURES)
# --------------------------------------------------
def extract_handcrafted_features(img):
    """
    Extract a fixed-length feature vector (18 features):
    7 Hu Moments + 11 Shape/Gradient/Curvature descriptors.
    """
    if img is None:
        return None

    # 1. Size normalization
    img = cv2.resize(img, (32, 32))

    # 2. Binarization and contour extraction
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Keep the largest connected component
    c = max(contours, key=cv2.contourArea)

    # --- NOURVEAUTÉ : HU MOMENTS (7 features) ---
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    # Log transformation pour stabiliser les valeurs (incontournable pour Hu)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # --- SHAPE DESCRIPTORS (11 features) ---
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

    # Gradient-based statistics
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_mean = np.mean(magnitude)
    grad_std = np.std(magnitude)

    # Curvature statistics
    c_pts = c[:, 0, :].astype(float)
    x_smooth = np.convolve(c_pts[:, 0], np.ones(3) / 3, mode='same')
    y_smooth = np.convolve(c_pts[:, 1], np.ones(3) / 3, mode='same')
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10)
    curvature_mean = np.mean(curvature)
    curvature_std = np.std(curvature)

    # --- FINAL VECTOR CONCATENATION (7 + 11 = 18) ---
    feature_vector = np.concatenate([
        hu, 
        np.array([
            area, perimeter, circularity, roundness,
            aspect_ratio, extent, solidity,
            grad_mean, grad_std, curvature_mean, curvature_std
        ])
    ])

    return feature_vector
# --------------------------------------------------
# Paths and data loading
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note : Vérifiez que votre CSV contient bien les chemins complets ou relatifs corrects
CSV_PATH = os.path.join(BASE_DIR, "data/dataset/dataset_labels.csv")

features = []
labels = []

if not os.path.exists(CSV_PATH):
    print(f"Erreur: {CSV_PATH} introuvable.")
else:
    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_path = row["path"].replace('\\', '/') 
            label = row['label'].strip()

            if not os.path.exists(img_path):
                print(f"Image manquante : {img_path}") # Debug
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            feat = extract_handcrafted_features(img)

            if feat is not None:
                features.append(feat)
                labels.append(label)

# --------------------------------------------------
# Vérification AVANT conversion
# --------------------------------------------------
if len(features) == 0:
    print("ERREUR : Aucune caractéristique n'a été extraite. Vérifiez vos chemins d'images.")
else:
    # Convert to NumPy arrays
    X = np.array(features)
    y = np.array(labels)

    # Si X est 1D (un seul échantillon), on le force en 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)

    print(f"Feature matrix shape: {X.shape}")

    # --------------------------------------------------
    # Normalization & Training
    # --------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    clf.fit(X_scaled, y)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    joblib.dump(clf, "random_forest_model.pkl")
    joblib.dump(scaler, "random_forest_scaler.pkl")

    print("Modèle et Scaler sauvegardés avec succès.")