import pandas as pd
import os

# Load the CSV file
csv_path = "C:/Users/zacha/mlCancerInfo/release_midas.xlsx"
df = pd.read_excel(csv_path)
df = df.drop(columns = ['midas_record_id','Unnamed: 0','midas_iscontrol', 'midas_ethnicity', 'midas_location','midas_gender'])

# Define paths
image_dir = "C:/Users/zacha/mlCancerData"
df['file_path'] = df['midas_file_name'].apply(lambda x: os.path.join(image_dir, x))

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

# Display the updated DataFrame
print(df[["file_path", "classification"]].head(40))

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

invalid_files = df[~df['file_path'].apply(os.path.isfile)]
print("Invalid files:")
print(invalid_files['file_path'])




## Function to create a residual block
# def residual_block(x, filters):
#     skip = x  # Save the input as the skip connection

#     # Main branch
#     x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.1)(x)

#     x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
#     x = BatchNormalization()(x)
    
#     # Match dimensions if necessary
#     if skip.shape[-1] != filters:
#         skip = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(0.01))(skip)

#     # Add residual connection
#     x = Add()([x, skip])
#     x = LeakyReLU(alpha=0.1)(x)
#     return x

# # Input Layer
# input_layer = Input(shape=(128, 128, 3))

# # First Convolutional Block (No Residual Connection)
# x = Conv2D(32, (5, 5), padding='same', kernel_regularizer=l2(0.01))(input_layer)
# x = BatchNormalization()(x)
# x = LeakyReLU(alpha=0.1)(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)

# # Second Block (Residual)
# x = residual_block(x, 64)
# x = MaxPooling2D(pool_size=(2, 2))(x)

# # Third Block (Residual)
# x = residual_block(x, 128)
# x = MaxPooling2D(pool_size=(2, 2))(x)

# # Fourth Block (Residual)
# x = residual_block(x, 256)
# x = MaxPooling2D(pool_size=(2, 2))(x)

# # Fifth Block (Residual)
# x = residual_block(x, 512)
# x = MaxPooling2D(pool_size=(2, 2))(x)

# # Sixth Block (Optional - Residual)
# x = residual_block(x, 1024)
# x = MaxPooling2D(pool_size=(2, 2))(x)

# # Apply Global Average Pooling
# x = GlobalAveragePooling2D()(x)

# # Fully Connected Layers
# x = Dense(1024, kernel_regularizer=l2(0.01))(x)
# x = LeakyReLU(alpha=0.1)(x)
# x = Dropout(0.3)(x)
# x = Dense(64)(x)
# x = LeakyReLU(alpha=0.1)(x)

# # Output Layer
# output_layer = Dense(1, activation='sigmoid')(x)

# #Build the model
# model = Model(inputs=input_layer, outputs=output_layer)