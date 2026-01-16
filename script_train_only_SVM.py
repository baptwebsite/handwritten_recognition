# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import csv
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# --------------------------------------------------
# Feature extraction function (11 features)
# --------------------------------------------------
def extract_handcrafted_features(img):
    if img is None: return None
    img = cv2.resize(img, (32, 32))
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0: return None
    c = max(contours, key=cv2.contourArea)

    # Shape descriptors
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

    # Gradient
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    grad_mean, grad_std = np.mean(mag), np.std(mag)

    # Curvature
    try:
        c_pts = c[:, 0, :].astype(float)
        dx = np.gradient(np.convolve(c_pts[:, 0], np.ones(3)/3, mode="same"))
        dy = np.gradient(np.convolve(c_pts[:, 1], np.ones(3)/3, mode="same"))
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        curv = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10)
        curv_mean, curv_std = np.mean(curv), np.std(curv)
    except:
        curv_mean, curv_std = 0, 0

    return np.array([area, perimeter, circularity, roundness, aspect_ratio, 
                     extent, solidity, grad_mean, grad_std, curv_mean, curv_std])

# --------------------------------------------------
# Paths and Data Loading
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset_labels.csv")

features = []
labels = []

print("Extracting features...")
if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found.")
else:
    with open(CSV_PATH, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_path = row["path"].replace('\\', '/')
            current_label = row['label'].strip()

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            feat = extract_handcrafted_features(img)

            if feat is not None:
                features.append(feat)
                labels.append(current_label)

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------
X = np.array(features)
y = np.array(labels)

# We split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --------------------------------------------------
# Normalization (Fit only on Training set!)
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use training scaler for test set

# --------------------------------------------------
# Training SVM
# --------------------------------------------------
print("Training SVM model...")
svm_clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42)
svm_clf.fit(X_train_scaled, y_train)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
y_pred = svm_clf.predict(X_test_scaled)
print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# --------------------------------------------------
# Save
# --------------------------------------------------
joblib.dump(svm_clf, "svm_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")
print("\nModel and Scaler saved successfully.")


# --------------------------------------------------
# Confusiion matrix
# --------------------------------------------------

# 1. Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=svm_clf.classes_)

# 2. Setup the visualization
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_clf.classes_)

# 3. Plot the matrix
print("Displaying Confusion Matrix...")
disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical')

plt.title("Confusion Matrix: Predicted vs Actual Letters")
plt.show()