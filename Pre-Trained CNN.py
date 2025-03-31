from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your CSV and preprocess as before
csv_path = "C:/Users/zacha/mlCancerInfo/release_midas.xlsx"
df = pd.read_excel(csv_path)
df['file_path'] = df['midas_file_name'].apply(lambda x: os.path.join("C:/Users/zacha/mlCancerData", x))

def classify_majority(row):
    impressions = [
        str(row["clinical_impression_1"]).lower(),
        str(row["clinical_impression_2"]).lower(),
        str(row["clinical_impression_3"]).lower(),
    ]
    # Count occurrences of "malignant" and "benign"
    malignant_count = sum("malignant" in impression for impression in impressions)
    benign_count = sum("benign" in impression for impression in impressions)
    
    # Assign 1 for malignant if malignant_count > benign_count, otherwise 0 for benign
    return 1 if malignant_count >= benign_count else 0

df["classification"] = df.apply(classify_majority, axis=1).map({1: "malignant", 0: "benign"})

# Calculate class weights for the 'classification' column
classification_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(df['classification'].unique()),
    y=df['classification']
)
classification_weight_dict = dict(zip(df['classification'].unique(), classification_weights))
print("Classification Weights:", classification_weight_dict)

# Calculate class weights for the 'race' column
skin_tone_counts = df['midas_race'].value_counts()
total_samples = len(df)

# Calculate race-specific weights inversely proportional to class frequency
skin_tone_weights = {tone: total_samples / (len(skin_tone_counts) * count) 
                     for tone, count in skin_tone_counts.items()}
print("Race Weights:", skin_tone_weights)

# Map weights to each data sample based on 'midas_race'
df['sample_weight'] = df['midas_race'].map(skin_tone_weights)

# Split data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Split weights into train/val sets
train_weights = train_df['sample_weight'].values
val_weights = val_df['sample_weight'].values

# Define image size and batch size
image_size = (224, 224)  # Adjusted for ResNet50
batch_size = 64

# Data augmentation for underrepresented groups (more aggressive for minorities)
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40 if 'asian' in df['midas_race'].values else 20,
    width_shift_range=0.3 if 'african_american' in df['midas_race'].values else 0.2,
    height_shift_range=0.3 if 'african_american' in df['midas_race'].values else 0.2,
    zoom_range=0.4 if 'asian' in df['midas_race'].values else 0.2,
    shear_range=0.3 if 'asian' in df['midas_race'].values else 0.2,
    horizontal_flip=True,
    validation_split=0.2  # Note: This will make sure validation data is excluded from augmentation
)

# Train data generator
train_data = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file_path',
    y_col='classification',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    sample_weight=train_weights  # Apply weights for race-based augmentation
)

# Validation data generator
val_data = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='file_path',
    y_col='classification',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    sample_weight=val_weights  # Apply weights for validation
)

# Combine weights for classification and race if needed
class_weight = {
    0: classification_weight_dict["benign"],  # Benign weight
    1: classification_weight_dict["malignant"],  # Malignant weight
}

# Now, you can pass `class_weight` when you call `model.fit()` to balance class weights in the training phase.


# Load pretrained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Define the new model
model = Model(inputs=base_model.input, outputs=output)


# Compile the model
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_resnet50_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# Train the model
# Add these weights to the model training part:
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    class_weight=class_weight,  # Add class weights for training
    callbacks=[checkpoint, early_stopping, lr_scheduler]
)

# Plot training results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(val_data, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save the model
keras.saving.save_model(model,'resnet50_custom_model.keras')
