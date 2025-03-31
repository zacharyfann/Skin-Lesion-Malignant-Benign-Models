from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image

# Load the CSV file
csv_path = "C:/Users/zacha/mlCancerInfo/release_midas.xlsx"
df = pd.read_excel(csv_path)

# Define paths
image_dir = "C:/Users/zacha/mlCancerData"
df['file_path'] = df['midas_file_name'].apply(lambda x: os.path.join(image_dir, x.strip()))

# Function to classify row based on the majority of malignant vs benign
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
    return 1 if malignant_count >= benign_count and malignant_count > 0 else 0

# Apply the function to create a new column
df["classification"] = df.apply(classify_majority, axis=1)

# Convert classification column from integers to strings for binary classification
df["classification"] = df["classification"].map({1: "malignant", 0: "benign"})

# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define image size and batch size
image_size = (128, 128)  # Resize to 128x128 pixels
batch_size = 64  # Increase batch size

# Create the data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,            # Normalize pixel values to [0, 1]
    rotation_range=20,            # Random rotation
    width_shift_range=0.2,        # Horizontal shift
    height_shift_range=0.2,       # Vertical shift
    shear_range=0.2,              # Shear transformations
    zoom_range=0.2,               # Random zoom
    horizontal_flip=True,         # Random horizontal flips
    validation_split=0.2          # Split data into training and validation sets
)

# Load training data using flow_from_dataframe
train_data = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file_path',
    y_col='classification',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',  # For binary classification (malignant/benign)
)

# Load validation data using flow_from_dataframe
val_data = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='file_path',
    y_col='classification',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',  # For binary classification (malignant/benign)
)

# Function to validate images
def validate_images(file_paths):
    invalid_files = []
    for path in file_paths:
        try:
            img = Image.open(path)
            img.verify()  # Verify image integrity
        except Exception as e:
            invalid_files.append(path)
    return invalid_files

# Check for invalid files
invalid_files = validate_images(train_data.filenames)
print(f"Invalid files: {invalid_files}")

# Model implementation
model = keras.Sequential([
    keras.layers.Input(shape=(128, 128, 3)),
    keras.layers.Rescaling(scale=1./255),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.1),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.1),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.1),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.1),
    keras.layers.Flatten(),
    keras.layers.Dense(512),
    keras.layers.Dense(128),
    keras.layers.Dense(32),
    keras.layers.Dense(1, activation=keras.activations.sigmoid),
])

# Callbacks for early stopping and saving the best model
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)
model_checkpoint = keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, mode="min", save_freq="epoch")
callbacks = [early_stopping, model_checkpoint]

# Parameters
epochs = 50  # Increase epochs to allow better convergence

# Compile the model with Adam optimizer
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=['accuracy', 'AUC'])
model.summary()

# Train the model
history = model.fit(
    
    train_data,
    validation_data=val_data,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks
)

# Visualize Results
# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate Model on Validation Set
test_loss, test_accuracy, test_auc = model.evaluate(val_data, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Save The Model
model.save('computerVisionNN_model.keras')


# Visualize Results
import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate Model
test_loss, test_accuracy = model.evaluate(val_data, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


