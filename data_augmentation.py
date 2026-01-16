import cv2
import numpy as np
import os
import random

# --- CONFIGURATION ---
base_dir = 'data/dataset'  # Votre dossier actuel contenant les dossiers a, b, c...
aug_factor = 5        # Nombre de nouvelles images à créer par image originale

def augment_image(img):
    """Applique des transformations aléatoires légères."""
    h, w = img.shape[:2]
    
    # 1. Rotation aléatoire légère (entre -15 et +15 degrés)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img_aug = cv2.warpAffine(img, M, (w, h), borderValue=255) # Fond blanc

    # 2. Zoom aléatoire (entre 0.9 et 1.1)
    zoom = random.uniform(0.9, 1.1)
    new_w, new_h = int(w * zoom), int(h * zoom)
    img_zoomed = cv2.resize(img_aug, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Recadrage ou padding pour revenir à TARGET_SIZE (64x64)
    final_img = np.full((h, w), 255, dtype=np.uint8)
    
    # On centre l'image zoomée dans le carré de destination
    start_y = max(0, (h - new_h) // 2)
    start_x = max(0, (w - new_w) // 2)
    
    # On calcule les zones de découpe si zoom > 1
    src_y = max(0, (new_h - h) // 2)
    src_x = max(0, (new_w - w) // 2)
    
    h_overlap = min(h, new_h)
    w_overlap = min(w, new_w)
    
    final_img[start_y:start_y+h_overlap, start_x:start_x+w_overlap] = \
        img_zoomed[src_y:src_y+h_overlap, src_x:src_x+w_overlap]

    # 3. Décalage léger (Translation)
    tx = random.randint(-3, 3)
    ty = random.randint(-3, 3)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    final_img = cv2.warpAffine(final_img, M_trans, (w, h), borderValue=255)

    return final_img

# --- EXÉCUTION ---
print("Début de l'augmentation des données...")

for lettre in os.listdir(base_dir):
    lettre_path = os.path.join(base_dir, lettre)
    if not os.path.isdir(lettre_path):
        continue
    
    # Liste des images originales de la lettre
    images_originales = [f for f in os.listdir(lettre_path) if f.endswith('.png') and 'aug' not in f]
    
    for img_name in images_originales:
        img_full_path = os.path.join(lettre_path, img_name)
        img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None: continue
        
        for i in range(aug_factor):
            img_aug = augment_image(img)
            # Sauvegarde avec un suffixe pour les différencier
            new_name = f"{img_name.split('.')[0]}_aug_{i}.png"
            cv2.imwrite(os.path.join(lettre_path, new_name), img_aug)

print(f"Terminé ! Chaque lettre a maintenant environ {len(images_originales) * (aug_factor + 1)} exemples.")