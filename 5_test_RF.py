# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import joblib
import csv
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# --------------------------------------------------
# 1. Load Model and Scaler
# --------------------------------------------------
MODEL_PATH = "model/RF_model.pkl" 
SCALER_PATH = "model/RF_scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler successfully loaded.")
else:
    print(f"Error: .pkl files not found. Check {MODEL_PATH} and {SCALER_PATH}")
    exit()

# --------------------------------------------------
# 2. Feature Extraction Function (Must match training)
# --------------------------------------------------
def extract_advanced_features(img):
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
    
    # Gradients
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    g_m, g_s = np.mean(mag), np.std(mag)

    # Curvature
    try:
        c_pts = c[:, 0, :].astype(float)
        dx, dy = np.gradient(c_pts[:, 0]), np.gradient(c_pts[:, 1])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        curv = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2 + 1e-10)
        c_m, c_s = np.mean(curv), np.std(curv)
    except:
        c_m, c_s = 0.0, 0.0

    geom_vector = np.array([area, peri, circ, rnd, asp, ext, sol, g_m, g_s, c_m, c_s])
    
    return np.concatenate([hog_feat, hu, geom_vector])

# --------------------------------------------------
# 3. Reproduce Training/Testing Split
# --------------------------------------------------
CSV_PATH = "dataset.csv"
all_paths = []
all_labels = []

if not os.path.exists(CSV_PATH):
    print(f"Error: CSV file not found at {CSV_PATH}")
    exit()

with open(CSV_PATH, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_paths.append(row['path'].replace('\\', '/'))
        all_labels.append(row['label'].strip())

# We use the EXACT same parameters as the training script
_, test_paths, _, y_test = train_test_split(
    all_paths, all_labels, test_size=0.20, random_state=42, stratify=all_labels
)

# --------------------------------------------------
# 4. Inference on Test Set
# --------------------------------------------------
y_pred = []
y_true_final = [] # Track labels for successfully loaded images

print(f"Extracting features for {len(test_paths)} test images...")

for img_path, true_label in zip(test_paths, y_test):
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    feat = extract_advanced_features(img)

    if feat is not None:
        # Reshape for single sample prediction
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        pred = clf.predict(feat_scaled)[0]
        y_pred.append(pred)
        y_true_final.append(true_label)

# --------------------------------------------------
# 5. Save Results to Files
# --------------------------------------------------
report = classification_report(y_true_final, y_pred)
print(report)

# Save Text Report
with open("classification_report.txt", "w") as f:
    f.write("=== RANDOM FOREST EVALUATION REPORT ===\n")
    f.write(report)

# Save Confusion Matrix Image

fig, ax = plt.subplots(figsize=(14, 11))
cmd = ConfusionMatrixDisplay.from_predictions(
    y_true_final, y_pred, display_labels=clf.classes_, 
    cmap='Greens', xticks_rotation='vertical', ax=ax
)
plt.title("Confusion Matrix - Improved RF (HOG + Geometric)")
plt.tight_layout()

# SAVE COMMAND
plt.savefig("confusion_matrix.png", dpi=300) 
print("Results saved: 'classification_report.txt' and 'confusion_matrix.png'")
plt.show()