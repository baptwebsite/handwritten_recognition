import cv2
import numpy as np
import os

# --- CONFIGURATION ---
alphabet = "abcdefghijklmnopqrstuvwxyz"
src_dir = 'data/raw_data'
dest_dir = 'data/dataset'
TARGET_SIZE = 64
NB_EXEMPLES = 21


if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

def segmenter_lettre(nom_lettre):
    img_path = os.path.join(src_dir, f"{nom_lettre}.jpg")
    if not os.path.exists(img_path):
        return

    # 1. Chargement et conversion
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Prétraitement (Flou médian contre le grain)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 8)

    # 3. Soudure des morceaux (Close)
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # --- FILTRAGE DES PETITS TROUS UNIQUEMENT ---
    # RETR_CCOMP permet de distinguer les contours externes et les trous
    cnts, hierarchy = cv2.findContours(processed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is not None:
        for i, c in enumerate(cnts):
            # Si le contour a un parent, c'est un trou (inner contour)
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(c)
                # Si le trou est minuscule (ex: moins de 40 pixels), on le bouche
                if area < 40:
                    cv2.drawContours(processed, [c], -1, 255, -1)

    # 4. Sélection des 21 meilleures lettres
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    lettres_candidates = sorted_contours[:NB_EXEMPLES]

    lettre_folder = os.path.join(dest_dir, nom_lettre)
    if not os.path.exists(lettre_folder):
        os.makedirs(lettre_folder)

    for i, cnt in enumerate(lettres_candidates):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Extraction avec marge
        margin = 15
        y1, y2 = max(0, y-margin), min(image.shape[0], y+h+margin)
        x1, x2 = max(0, x-margin), min(image.shape[1], x+w+margin)
        roi = processed[y1:y2, x1:x2]

        # 5. Formatage final 64x64
        final_img = np.full((TARGET_SIZE, TARGET_SIZE), 255, dtype=np.uint8)
        
        h_roi, w_roi = roi.shape
        ratio = min((TARGET_SIZE-16)/w_roi, (TARGET_SIZE-16)/h_roi)
        new_w, new_h = int(w_roi * ratio), int(h_roi * ratio)
        roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Inversion pour Noir sur Blanc
        roi_final = cv2.bitwise_not(roi_resized)

        # Centrage
        off_y = (TARGET_SIZE - new_h) // 2
        off_x = (TARGET_SIZE - new_w) // 2
        final_img[off_y:off_y+new_h, off_x:off_x+new_w] = roi_final

        cv2.imwrite(os.path.join(lettre_folder, f"{nom_lettre}_{i}.png"), final_img)

    print(f"Lettre '{nom_lettre}' traitée : {len(lettres_candidates)} images.")

# --- LANCEMENT ---
# for char in alphabet:
#     segmenter_lettre(char)

segmenter_lettre("test")


