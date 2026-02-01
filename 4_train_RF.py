# -*- coding: utf-8 -*-
"""
Train a Random Forest classifier for handwritten character recognition.
Features: 18 (7 Hu Moments + 11 Shape/Gradient/Curvature descriptors).
"""
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import csv
import joblib
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def extract_advanced_features(img):
    """
    Extracts 162 features: 
    7 Hu Moments + 11 Geometric + 144 HOG features.
    """
    if img is None: return None
    
    # Standardize size
    img = cv2.resize(img, (32, 32))
    
    # --- 1. HOG FEATURES (144 features) ---
    # orientations=9: number of direction bins
    # pixels_per_cell=(8, 8): divide 32x32 into 16 cells (4x4 grid)
    # cells_per_block=(2, 2): normalization blocks
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)

    # --- 2. GEOMETRIC & HU (18 features) ---
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Return HOG + 18 zeros if no contour is found
        return np.concatenate([hog_features, np.zeros(18)])
    
    c = max(contours, key=cv2.contourArea)
    
    # Hu Moments
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # Geometric descriptors
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    asp = w / h if h > 0 else 0.0
    ext = area / (w * h) if w * h > 0 else 0.0
    sol = area / cv2.contourArea(cv2.convexHull(c)) if cv2.contourArea(cv2.convexHull(c)) > 0 else 0.0
    circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0.0
    (_, _), rad = cv2.minEnclosingCircle(c)
    rnd = 4 * area / (np.pi * (rad*2)**2) if rad > 0 else 0.0
    
    # Gradients
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    g_m, g_s = np.mean(mag), np.std(mag)

    # Simplified Curvature
    try:
        c_pts = c[:, 0, :].astype(float)
        dx, dy = np.gradient(c_pts[:, 0]), np.gradient(c_pts[:, 1])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        curv = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2 + 1e-10)
        c_m, c_s = np.mean(curv), np.std(curv)
    except:
        c_m, c_s = 0.0, 0.0

    geom_vector = np.array([area, peri, circ, rnd, asp, ext, sol, g_m, g_s, c_m, c_s])
    
    # Final Concatenation: 144 (HOG) + 7 (HU) + 11 (GEOM) = 162
    return np.concatenate([hog_features, hu, geom_vector])

# --------------------------------------------------
# Paths and Data Loading
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")

features = []
labels = []

print("Extracting features from dataset...")

if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found.")
else:
    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_path = row["path"].replace('\\', '/')
            label = row['label'].strip()

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            feat = extract_advanced_features(img)

            if feat is not None:
                features.append(feat)
                labels.append(label)

if len(features) == 0:
    print("CRITICAL ERROR: No features extracted. Check image paths.")
else:
    X = np.array(features)
    y = np.array(labels)

    # --------------------------------------------------
    # Train/Test Split (80% Train, 20% Test)
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # Training Random Forest
    # --------------------------------------------------
    print("Training Random Forest model...")
    clf = RandomForestClassifier(
        n_estimators=500,  # Increased trees for more features
        max_depth=25,      # Allow more depth for HOG complexity
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    y_pred = clf.predict(X_test_scaled)
    
    print("\n" + "="*30)
    print(f"OVERALL ACCURACY: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("="*30)
    print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred))

    # # --------------------------------------------------
    # # Confusion Matrix Visualization
    # # --------------------------------------------------
    # fig, ax = plt.subplots(figsize=(12, 10))
    # disp = ConfusionMatrixDisplay.from_predictions(
    #     y_test, y_pred, 
    #     display_labels=clf.classes_, 
    #     cmap='Greens', 
    #     ax=ax, 
    #     xticks_rotation='vertical'
    # )
    # plt.title("Random Forest Confusion Matrix")
    # plt.show()

    # --------------------------------------------------
    # Save Model & Scaler
    # --------------------------------------------------
    joblib.dump(clf, "model/RF_model.pkl")
    joblib.dump(scaler, "model/RF_scaler.pkl")
    print("\nRandom Forest model and Scaler saved successfully.")