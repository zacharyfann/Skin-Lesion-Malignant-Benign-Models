# importing modules 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Load dataset
df = pd.read_csv("C:/Users/zacha/Downloads/parkinsons (1)/parkinsons.data")

# Prepare features and target
X = df.drop(columns=['status', 'name']).values 
y = df['status'].values 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Logistic Regression model using Keras
model = keras.Sequential([
    keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train_scaled, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2,
    verbose=0
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Decision Boundary Visualization
def plot_decision_boundary(X, y, model, scaler):
    # Select two features for visualization
    plt.figure(figsize=(12, 8))
    
    # Use feature importance from the original model
    full_weights = model.layers[0].get_weights()
    weights = full_weights[0].flatten()
    feature_indices = np.argsort(np.abs(weights))[-2:]
    
    # Extract the two selected features
    feature_names = df.drop(columns=['status', 'name']).columns[feature_indices]
    X_selected = X[:, feature_indices]
    
    # Scale the selected features
    X_scaled = scaler.transform(X)[:, feature_indices]
    
    # Create a mesh grid
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Prepare the mesh points for prediction
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Create a temporary model with just these two features
    temp_model = keras.Sequential([
        keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
    ])
    
    # Manually set weights for the two selected features
    selected_weights = full_weights[0][feature_indices, :]
    selected_bias = full_weights[1]
    
    # Compile and set weights
    temp_model.compile(optimizer='adam', loss='binary_crossentropy')
    temp_model.layers[0].set_weights([selected_weights, selected_bias])
    
    # Predict for each point in the mesh
    Z = temp_model.predict(mesh_points).reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)
    
    # Scatter plot of original data points
    plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], 
                c='blue', label='Negative Cases', alpha=0.7)
    plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], 
                c='red', label='Positive Cases', alpha=0.7)
    
    plt.xlabel(f'{feature_names[0]} (Scaled)')
    plt.ylabel(f'{feature_names[1]} (Scaled)')
    plt.title('Decision Boundary with Positive and Negative Cases')
    plt.colorbar(label='Probability of Positive Class')
    plt.legend()
    plt.show()

# Plot decision boundary using the trained model
plot_decision_boundary(X, y, model, scaler)
