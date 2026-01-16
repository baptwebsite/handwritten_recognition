# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 2026
@author: veroe

Script pour utiliser le modèle Random Forest sur un texte manuscrit
Classification lettre par lettre
"""

import numpy as np
import os
import cv2
import joblib
import csv

# --------------------------------------------------
# Charge model scaler
# --------------------------------------------------
clf = joblib.load("random_forest_model.pkl")
scaler = joblib.load("random_forest_scaler.pkl")

# --------------------------------------------------
# 2. Fonction pour extraire les mêmes features qu'à l'entraînement
# --------------------------------------------------
def extract_handcrafted_features(img):
    """
    img: image grayscale d'une lettre (taille arbitraire)
    Retourne un vecteur de features compatible avec le modèle RF
    """
    # Normalisation
    img = cv2.resize(img, (32, 32))
    
    # Binarisation
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    c = max(contours, key=cv2.contourArea)
    
    # HU Moments
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # Shape descriptors
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    (x, y, w, h) = cv2.boundingRect(c)
    aspect_ratio = w / h if h > 0 else 0.0
    extent = area / (w * h) if w * h > 0 else 0.0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    (xc, yc), radius = cv2.minEnclosingCircle(c)
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
    
    # Feature vector
    feature_vector = np.concatenate([
        hu,
        np.array([area, perimeter, circularity, roundness,
                  aspect_ratio, extent, solidity,
                  grad_mean, grad_std, curvature_mean, curvature_std])
    ])
    
    return feature_vector

# --------------------------------------------------
# Charge images in separated files
# --------------------------------------------------
TEXT_DIR = os.path.join(os.path.dirname(__file__), "data/dataset/test")  # dossier images lettres
letters_files = sorted([f for f in os.listdir(TEXT_DIR) if f.endswith(".png")])

# --------------------------------------------------
# Character reco
# --------------------------------------------------
output_csv = "random_forest_predictions.csv"
recognized_labels = []

with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "predicted_label"])

    for f in letters_files:
        img_path = os.path.join(TEXT_DIR, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feat = extract_handcrafted_features(img)

        if feat is not None:
            feat_scaled = scaler.transform([feat])  # normalisation
            pred_label = clf.predict(feat_scaled)[0]  # class prediction
        else:
            pred_label += "?"  # no possible reco

        writer.writerow([f, pred_label])
        recognized_labels.append(pred_label)

    
