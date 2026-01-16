# -*- coding: utf-8 -*-
import cv2
import numpy as np
import joblib
import os

# 1. Charger le modèle et le scaler (18 features)
clf = joblib.load("random_forest_model.pkl")
scaler = joblib.load("random_forest_scaler.pkl")

def extract_handcrafted_features(img):
    """ Extraction identique à l'entraînement (18 features) """
    img = cv2.resize(img, (32, 32))
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return None
    c = max(contours, key=cv2.contourArea)
    
    # Hu Moments (7)
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # Shape & Gradients (11)
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
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    
    # Courbure simplifiée pour le test
    curvature_mean, curvature_std = 0, 0 # Valeurs par défaut si échec
    try:
        c_pts = c[:,0,:].astype(float)
        dx = np.gradient(c_pts[:,0])
        dy = np.gradient(c_pts[:,1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curv = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2 + 1e-10)
        curvature_mean, curvature_std = np.mean(curv), np.std(curv)
    except: pass

    return np.concatenate([hu, [area, perimeter, circularity, roundness, aspect_ratio, 
                                extent, solidity, np.mean(mag), np.std(mag), 
                                curvature_mean, curvature_std]])

def lire_photo_texte(image_path):
    # Charger l'image
    img_originale = cv2.imread(image_path)
    gray = cv2.cvtColor(img_originale, cv2.COLOR_BGR2GRAY)
    
    # Segmentation (même logique que pour vos 'a')
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 8)
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Liste pour stocker les lettres et leurs positions (x) pour les trier
    lettres_detectees = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            # Extraire la zone d'intérêt (ROI)
            roi = gray[y:y+h, x:x+w]
            
            # Prédiction
            feat = extract_handcrafted_features(roi)
            if feat is not None:
                feat_scaled = scaler.transform(feat.reshape(1, -1))
                label = clf.predict(feat_scaled)[0]
                lettres_detectees.append((x, label)) # On garde la position X pour trier

    # TRI : Trier les lettres par leur position X (de gauche à droite)
    lettres_detectees.sort(key=lambda x: x[0])
    
    # Assembler le texte
    texte_final = "".join([l[1] for l in lettres_detectees])
    return texte_final

# --- TEST ---
image_test = "data/dataset/test_texte.jpg" # Mettez ici le nom de votre photo
if os.path.exists(image_test):
    resultat = lire_photo_texte(image_test)
    print(f"Texte reconnu : {resultat}")
else:
    print("Photo de test introuvable.")