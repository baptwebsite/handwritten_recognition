# -*- coding: utf-8 -*-
import os
import csv

"""
This script allow to create the csv file used for training, testing models.
"""
# --- CONFIGURATION ---
dataset_dir = 'data/dataset/'
output_csv = 'dataset_labels.csv' # Standard name for training labels

def generate_csv():
    data = []

    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: The directory '{dataset_dir}' does not exist.")
        return

    # Iterate through folders for each letter (a, b, c...)
    # sorted() ensures the CSV is organized alphabetically
    for letter in sorted(os.listdir(dataset_dir)):
        letter_path = os.path.join(dataset_dir, letter)

        # Check if the path is actually a directory
        if os.path.isdir(letter_path):
            for filename in sorted(os.listdir(letter_path)):
                # Filter for image files
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    
                    # 1. Create the full file path
                    full_path = os.path.join(letter_path, filename)
                    
                    # 2. Force the use of forward slashes / (Standard for cross-platform compatibility)
                    clean_path = full_path.replace('\\', '/')
                    
                    # 3. Store the path and the label (the folder name represents the label)
                    data.append([clean_path, letter])

    # Writing to the CSV file
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header row
            writer.writerow(['path', 'label'])
            # Write all collected data rows
            writer.writerows(data)

        print(f"Success! The file '{output_csv}' has been created with {len(data)} entries.")
        if data:
            print(f"Example entry: Path = {data[0][0]} | Label = {data[0][1]}")
            
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")

if __name__ == "__main__":
    generate_csv()