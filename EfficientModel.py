from turtle import ycor
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD, schedules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy


# Load the CSV file
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
train_df, val_df = train_test_split(df, test_size=0.2, random_state=70)
train_weights = train_df['sample_weight'].values
val_weights = val_df['sample_weight'].values

# Define image size and batch size
image_size = (224, 224)  # Increased from (128, 128)
batch_size = 32  # Reduce batch size to avoid memory issues

# Separate ImageDataGenerators for train and validation (prevents data leakage)
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    shear_range=0.15
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
from sklearn.utils.class_weight import compute_class_weight
class_weight_values = compute_class_weight(
    class_weight='balanced',
    classes=np.array(['malignant', 'benign']),
    y=train_df['classification']
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weight_values)}

# Load EfficientNetB3 Pretrained Model (removes top dense layers)
base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze 65% of layers
for layer in base_model.layers[:int(len(base_model.layers) * 0.65)]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):  # Keep batchnorm layers trainable
        layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)  # Reduced from 40% to 30%
x = Dense(256, activation="relu")(x)
x = Dropout(0.15)(x)  # Reduced from 20% to 15%
output = Dense(1, activation="sigmoid")(x)

# Create model
model = Model(inputs=base_model.input, outputs=output)

# Learning rate schedule
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=1e-4, decay_steps=1000, decay_rate=0.96, staircase=True
)

# Compile model with Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('efficientNet_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,  # Increased from 20
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

# Save the final trained model
model.save('efficientNet_model.keras')
