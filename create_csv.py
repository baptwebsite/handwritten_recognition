import os
import csv

# --- CONFIGURATION ---
dataset_dir = 'data/dataset/'
output_csv = 'prediction_labels.csv'

def generate_csv():
    data = []

    if not os.path.exists(dataset_dir):
        print(f"Erreur : Le dossier '{dataset_dir}' n'existe pas.")
        return

    # Parcourir les dossiers de chaque lettre
    for lettre in sorted(os.listdir(dataset_dir)):
        lettre_path = os.path.join(dataset_dir, lettre)

        if os.path.isdir(lettre_path):
            for filename in sorted(os.listdir(lettre_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 1. Créer le chemin complet
                    full_path = os.path.join(lettre_path, filename)
                    
                    # 2. Forcer l'usage des slashs / (standard Linux/Web)
                    clean_path = full_path.replace('\\', '/')
                    clean_path.replace("data/dataset", "")
                    data.append([clean_path, lettre])

    # Écriture du fichier CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])
        writer.writerows(data)

    print(f"Félicitations ! Le fichier '{output_csv}' a été créé avec des chemins en /.")
    if data:
        print(f"Exemple de chemin : {data[0][0]}")

if __name__ == "__main__":
    generate_csv()