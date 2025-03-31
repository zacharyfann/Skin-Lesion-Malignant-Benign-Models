import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load trained model
model_path = "final_finetuned_EffNet_model.keras"  # Update if needed
model = load_model(model_path)

# Load CSV data
csv_path = "C:/Users/zacha/mlCancerInfo/release_midas.xlsx"
df = pd.read_excel(csv_path)

# Define image directory and construct full file paths
image_dir = "C:/Users/zacha/mlCancerData"
df['file_path'] = df['midas_file_name'].apply(lambda x: os.path.join(image_dir, x.strip()) if pd.notna(x) else None)

# Ensure all image files exist
df = df[df['file_path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)]

# Function to classify based on majority malignant/benign
def classify_majority(row):
    impressions = [
        str(row["clinical_impression_1"]).lower(),
        str(row["clinical_impression_2"]).lower(),
        str(row["clinical_impression_3"]).lower(),
    ]
    malignant_count = sum("malignant" in impression for impression in impressions)
    benign_count = sum("benign" in impression for impression in impressions)
    return 1 if malignant_count >= benign_count and malignant_count > 0 else 0

df["classification"] = df.apply(classify_majority, axis=1)
df["classification"] = df["classification"].map({1: "malignant", 0: "benign"})

# Handle missing race values
df['midas_race'].fillna('unknown', inplace=True)

# Load test data
test_csv_path = "C:/Users/zacha/mlCancerInfo/release_midas.xlsx"

df["file_path"] = df["midas_file_name"].apply(lambda x: os.path.join(image_dir, x.strip()))
df = df[df["file_path"].apply(lambda x: os.path.exists(x))]  # Ensure images exist

# Define test image generator (rescale only, no augmentation)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create test generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df,
    x_col="file_path",
    y_col="classification",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Get true labels
y_true = test_generator.classes  

# Get predictions (raw probability scores)
y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary labels

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))
