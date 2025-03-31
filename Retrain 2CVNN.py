from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dropout, Dense
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

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

# Convert to string labels for binary classification
df["classification"] = df["classification"].map({1: "malignant", 0: "benign"})

# Handle missing race values
df['midas_race'].fillna('unknown', inplace=True)

# Compute sample weights based on race distribution
race_counts = df['midas_race'].value_counts()
total_samples = len(df)
race_weights = {race: total_samples / (len(race_counts) * count) for race, count in race_counts.items()}
df['sample_weight'] = df['midas_race'].map(race_weights)

# Split into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=13)
train_weights = train_df['sample_weight'].values
val_weights = val_df['sample_weight'].values

# Define image size and batch size
image_size = (128, 128)
batch_size = 64

# ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,  
    width_shift_range=0.3,  
    height_shift_range=0.3,  
    zoom_range=0.4,  
    horizontal_flip=True,  
    brightness_range=[0.8, 1.2],  
    shear_range=0.2  
)
# Train DataLoader
train_data = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file_path',
    y_col='classification',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    sample_weight=train_weights
)

# Validation DataLoader
val_data = datagen.flow_from_dataframe(
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

# Load Pretrained Model
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import custom_object_scope

    with custom_object_scope({'Dense': Dense}):
        model = load_model('2computerVisionNN_model.keras')

    print("Model loaded successfully.")
except:
    print("Error loading model. Check if architecture matches the dataset.")
    exit()

# Freeze initial layers
for layer in model.layers[:34]:  # Adjust this based on experiment
    layer.trainable = False

for layer in model.layers[34:]:  # Fine-tune the last 12 layers
    layer.trainable = True

# Modify the last few layers for fine-tuning
x = model.layers[-3].output  # Extract last layer before FC layers
x = Dropout(0.2)(x)  # Increase dropout to 40%
x = Dense(64, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid', name='output_layer')(model.layers[-2].output)

# Rebuild model
model = Model(inputs=model.input, outputs=output_layer)

# Adjust learning rate
optimizer = Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('2computerVisionNN_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
print(f"Total number of layers: {len(model.layers)}")

# Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,  # Reduce epochs to avoid overfitting
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stopping]
)

# Plot Results
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
model.save('2computerVisionNN_model.keras')
