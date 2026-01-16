# -*- coding: utf-8 -*-
import cv2
import numpy as np
import joblib
import os
from skimage.feature import hog

# --- CONFIGURATION ---
MODEL_PATH = "IMPROVE_random_forest_model.pkl"
SCALER_PATH = "IMPROVE_random_forest_scaler.pkl"
TARGET_SIZE = 64  # Doit correspondre à la taille utilisée dans segment_letter

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modèle RF chargé avec succès.")
else:
    print("Erreur : Fichiers modèles introuvables.")
    exit()

def extract_advanced_features(img):
    """ Extraction identique à l'entraînement (HOG + 18 features) """
    # Le modèle attend du 32x32 pour les features (même si le canvas est 64x64)
    img_32 = cv2.resize(img, (32, 32))
    
    # HOG
    hog_feat = hog(img_32, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=False)
    
    # Handcrafted (Géométrie sur le binaire)
    _, thresh = cv2.threshold(img_32, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.concatenate([hog_feat, np.zeros(18)])
    
    c = max(contours, key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # ... (Calcul des 11 features géométriques) ...
    # Note : Assure-toi que l'ordre ici est strictement le même que pendant l'entraînement
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    asp = w/h if h>0 else 0
    ext = area/(w*h) if w*h>0 else 0
    sol = area/cv2.contourArea(cv2.convexHull(c)) if cv2.contourArea(cv2.convexHull(c))>0 else 0
    circ = 4*np.pi*area/(peri**2) if peri>0 else 0
    (_, _), r = cv2.minEnclosingCircle(c)
    rnd = 4*area/(np.pi*(r*2)**2) if r>0 else 0
    gx, gy = cv2.Sobel(img_32, cv2.CV_64F, 1, 0), cv2.Sobel(img_32, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    # Curvatures simplifiées (valeurs par défaut ou calculées)
    c_m, c_s = 0, 0 

    geom_feat = [area, peri, circ, rnd, asp, ext, sol, np.mean(mag), np.std(mag), c_m, c_s]
    return np.concatenate([hog_feat, hu, geom_feat])

def read_text_from_photo(image_path):
    image = cv2.imread(image_path)
    if image is None: return "Image introuvable"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Nettoyage et Binarisation (Comme dans ta fonction segment_letter)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 8)

    # 2. Morphologie (Kernel 5x5 comme demandé)
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 3. Détection des lettres
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_chars = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 4. Extraction du ROI avec marge (margin=15)
            margin = 15
            y1, y2 = max(0, y-margin), min(image.shape[0], y+h+margin)
            x1, x2 = max(0, x-margin), min(image.shape[1], x+w+margin)
            roi = processed[y1:y2, x1:x2]

            if roi.size == 0: continue

            # 5. Formatage final TARGET_SIZE x TARGET_SIZE sur fond BLANC
            final_img = np.full((TARGET_SIZE, TARGET_SIZE), 255, dtype=np.uint8)
            
            h_roi, w_roi = roi.shape
            ratio = min((TARGET_SIZE-16)/w_roi, (TARGET_SIZE-16)/h_roi)
            new_w, new_h = int(w_roi * ratio), int(h_roi * ratio)
            
            if new_w > 0 and new_h > 0:
                roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Inversion : blanc sur noir -> NOIR sur BLANC (comme entraînement)
                roi_final = cv2.bitwise_not(roi_resized)

                # Centrage
                off_y = (TARGET_SIZE - new_h) // 2
                off_x = (TARGET_SIZE - new_w) // 2
                final_img[off_y:off_y+new_h, off_x:off_x+new_w] = roi_final

                # 6. Prédiction
                feat = extract_advanced_features(final_img)
                if feat is not None:
                    feat_scaled = scaler.transform(feat.reshape(1, -1))
                    label = clf.predict(feat_scaled)[0]
                    detected_chars.append((x, y, label))

    # Tri par ligne puis de gauche à droite
    detected_chars.sort(key=lambda c: (c[1] // 50, c[0]))
    return "".join([c[2] for c in detected_chars])

# --- TEST ---
print(f"Texte détecté : {read_text_from_photo('data/bonjour.jpg')}")