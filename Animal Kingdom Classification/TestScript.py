import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from keras.models import load_model

# === Step 1: Load the Saved Model ===
model_path = r"D:\Neural & DL\Neural Project\NN_Project\animal_vit_model.h5"
model = load_model(model_path)
print("Model loaded successfully.")

# === Step 2: Define Test Folder Path ===
test_folder = r"C:\Users\lenovo\Desktop\Wednesday"  # Path to your test images folder

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    image_resized = cv2.resize(image, target_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb / 255.0
    return image_normalized
# === Step 4: Predict on Test Images ===
predictions = []  # List to store predictions

for filename in os.listdir(test_folder):
    # Check if file is an image
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(test_folder, filename)
        try:
            # Preprocess the image
            preprocessed_image = preprocess_image(image_path)
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
            # Make prediction
            prediction = model.predict(preprocessed_image, verbose=0)  # Suppress verbose output
            predicted_class = np.argmax(prediction)  # Get class with highest probability
            # Append result as a dictionary
            # Append result as a dictionary
            predictions.append({
                'ImageID': os.path.splitext(filename)[0] + '.jpg',
                'Class': predicted_class  # Assuming classes are 1-indexed
            })
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
# === Step 5: Output Predictions ===
# Specify a valid output CSV file path
output_csv_path = "C:\\Users\\lenovo\\Desktop\\predictions.csv"  # Include the filename with .csv extension
predictions_df = pd.DataFrame(predictions)

# Save predictions to CSV
predictions_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}.")