import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load the pre-trained model
model_path = "efficientNet_model_finetuned.keras"
model = load_model(model_path)  # This ensures we continue training from the refined model

# Freeze more layers: Now, 80% of layers are frozen (previously 65%)
for layer in model.layers[:int(len(model.layers) * 0.85)]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):  # Keep BatchNorm layers trainable
        layer.trainable = False

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

# Compute sample weights based on race distribution
race_counts = df['midas_race'].value_counts()
total_samples = len(df)
race_weights = {race: total_samples / (len(race_counts) * count) for race, count in race_counts.items()}
df['sample_weight'] = df['midas_race'].map(race_weights)

# Split into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=99)
train_weights = train_df['sample_weight'].values
val_weights = val_df['sample_weight'].values

# Image settings
image_size = (224, 224)
batch_size = 32

# Data augmentation (more aggressive transformations)
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,  # Increased rotation
    width_shift_range=0.3,  # Increased width shift
    height_shift_range=0.3,  # Increased height shift
    zoom_range=0.4,  # Increased zoom range
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Expanded brightness range
    shear_range=0.3  # Increased shear
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Train DataLoader
train_data = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file_path',
    y_col='classification',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    sample_weight=train_weights
)

# Validation DataLoader
val_data = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='file_path',
    y_col='classification',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    sample_weight=val_weights
)

# Compute class weights
class_weight_values = compute_class_weight(
    class_weight='balanced',
    classes=np.array(['malignant', 'benign']),
    y=train_df['classification']
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weight_values)}

# Reduce learning rate further for fine-tuning
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=5e-6,  # Lower learning rate for even finer tuning
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('final_finetuned_EffNet_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fine-tune the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,  # Reduced epochs as per your request
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stopping]
)

# Plot Results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Training & Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training & Validation Loss")
plt.show()

# Evaluate Model
test_loss, test_accuracy = model.evaluate(val_data, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save the final fine-tuned model
model.save('final_finetuned_EffNet_model.keras')
