import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:/Users/zacha/Downloads/parkinsons (1)/parkinsons.data")

# Prepare features and target
x_orig = df.drop(columns = ['status', 'name']).values 

y_orig = df[['status']].values 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x_orig, y_orig, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_normalized, y_train)

# Make predictions
y_pred = model.predict(X_test_normalized)
y_pred_prob = model.predict_proba(X_test_normalized)[:, 1]

# Print model performance metrics
print("\nModel Performance on Test Data:")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

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

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': df.drop(columns = ['status', 'name']).columns.tolist(),
    'Importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.show()

# Scatter plot of top two most important features
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(2)['Feature'].tolist()

# Create a DataFrame from x_orig with column names
x_df = pd.DataFrame(x_orig, columns=df.drop(columns=['status', 'name']).columns)

# Add the target column to the DataFrame
x_df['status'] = y_orig

# Separate positive and negative cases
negative_cases = x_df[x_df['status'] == 0]
positive_cases = x_df[x_df['status'] == 1]

plt.scatter(negative_cases[top_features[0]], 
            negative_cases[top_features[1]], 
            label='Negative Cases', 
            color='blue', 
            alpha=0.7)
plt.scatter(positive_cases[top_features[0]], 
            positive_cases[top_features[1]], 
            label='Positive Cases', 
            color='red', 
            alpha=0.7)
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.title('Scatter Plot of Top Two Most Important Features')
plt.legend()
plt.show()
