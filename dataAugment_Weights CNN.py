from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Load the CSV file
csv_path = "C:/Users/zacha/mlCancerInfo/release_midas.xlsx"
df = pd.read_excel(csv_path)
# df = df.drop(columns = ['midas_record_id','Unnamed: 0','midas_iscontrol', 'midas_ethnicity', 'midas_location','midas_gender'])

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

# Display the updated DataFrame
# print(df[["file_path", "classification"]].head(40))

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Example: Adding a 'skin_tone' column to the DataFrame
# Assuming skin_tones are labeled as 'white', 'asian', 'african_american', etc.
# Ensure your dataset includes this column or preprocess it accordingly.
# Example: df['skin_tone'] = ['white', 'asian', 'african_american', ...]

# Check for all unique categories in the column
print("Unique Values in midas_race:", df['midas_race'].unique())

# Check for NaNs
if df['midas_race'].isnull().sum() > 0:
    print("Found NaN values. Replacing them with 'unknown'.")
    df['midas_race'].fillna('unknown', inplace=True)

# Double-check the value counts
skin_tone_counts = df['midas_race'].value_counts()
print("Value Counts:", skin_tone_counts)

# Total number of samples
total_samples = len(df)

# Calculate weights inversely proportional to class frequency
skin_tone_weights = {tone: total_samples / (len(skin_tone_counts) * count) 
                     for tone, count in skin_tone_counts.items()}

print("Skin Tone Weights:", skin_tone_weights)

# Map weights to each data sample
df['sample_weight'] = df['midas_race'].map(skin_tone_weights)
# Example split into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Split weights into train/val sets
train_weights = train_df['sample_weight'].values
val_weights = val_df['sample_weight'].values

# Define image size and batch size
image_size = (128, 128)  # Resize to 128x128 pixels

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
    validation_split=0.2
)

# Train data
train_data = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file_path',
    y_col='classification',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    sample_weight=train_weights  # Add weights here
)

# Validation data
val_data = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='file_path',
    y_col='classification',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    sample_weight=val_weights  # Add weights here
)

# Class weights for final model training (malignant/benign)
classes = train_df['classification'].unique()
class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.array(classes),
    y=train_df['classification']
)

class_weight_dict = dict(enumerate(class_weight))
print("Class Weights:", class_weight_dict)


# Create the neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Add, LeakyReLU, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

# Function to create a residual block
def residual_block(x, filters):
    skip = x  # Save the input as the skip connection

    # Main branch
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    
    # Match dimensions if necessary
    if skip.shape[-1] != filters:
        skip = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(0.01))(skip)

    # Add residual connection
    x = Add()([x, skip])
    x = LeakyReLU(alpha=0.1)(x)
    return x

# Input Layer
input_layer = Input(shape=(128, 128, 3))

# First Convolutional Block (No Residual Connection)
x = Conv2D(32, (5, 5), padding='same', kernel_regularizer=l2(0.01))(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Second Block (Residual)
x = residual_block(x, 64)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Third Block (Residual)
x = residual_block(x, 128)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Fourth Block (Residual)
x = residual_block(x, 256)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Fifth Block (Residual)
x = residual_block(x, 512)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Sixth Block (Optional - Residual)
x = residual_block(x, 1024)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Apply Global Average Pooling
x = GlobalAveragePooling2D()(x)

# Fully Connected Layers
x = Dense(1024, kernel_regularizer=l2(0.01))(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.3)(x)
x = Dense(64)(x)
x = LeakyReLU(alpha=0.1)(x)

# Output Layer
output_layer = Dense(1, activation='sigmoid')(x)

# Build Model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile Model
optimizer = Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model Summary
model.summary()

# Callbacks
checkpoint = ModelCheckpoint('2best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    batch_size=64,
    class_weight=class_weight_dict,  # Add class weights if needed
    callbacks=[checkpoint, early_stopping, lr_scheduler]
)

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

# Save The Model
model.save('2computerVisionNN_model.keras')  
