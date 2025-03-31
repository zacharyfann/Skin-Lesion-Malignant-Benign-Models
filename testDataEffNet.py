import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd
import os

# Load fine-tuned model
model = tf.keras.models.load_model("final_finetuned_EffNet_model.keras")

# Function to predict on a single image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    prediction = model.predict(img_array)[0][0]  # Get prediction
    label = "Malignant" if prediction >= 0.5 else "Benign"
    
    print(f"Prediction: {label} ({prediction:.4f})")
    return label

# Test on your own images
test_images = ["path_to_your_test_image1.jpg", "path_to_your_test_image2.jpg"]
import os
import pandas as pd
from pathlib import Path

def create_image_dataframe(ipynb_checkpoints_path):
    """
    Create a pandas DataFrame with image paths from .ipynb_checkpoints folder.
    
    Args:
        ipynb_checkpoints_path (str): Full path to .ipynb_checkpoints directory
    
    Returns:
        pd.DataFrame: DataFrame with columns 'image_path' and 'image_name'
    """
    # List of common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif']
    
    # Collect image paths
    image_paths = []
    for root, dirs, files in os.walk(ipynb_checkpoints_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(root, file)
                image_paths.append({
                    'image_path': full_path,
                    'image_name': file
                })
    
    # Create DataFrame
    df = pd.DataFrame(image_paths)
    return df

# Specific path to .ipynb_checkpoints
ipynb_checkpoints_path = r'C:\Users\zacha\mlCancerInfo\.ipynb_checkpoints'
image_df = create_image_dataframe(ipynb_checkpoints_path)
print(image_df)
print(f"\nTotal images found: {len(image_df)}")
for img_path in image_df['image_path']:
    print(img_path)
    predict_image(img_path)
