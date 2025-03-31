from ucimlrepo import fetch_ucirepo 
import pandas as pd
# importing modules 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder 


# fetch dataset 
parkinsons = fetch_ucirepo(id=174) 
  
# data (as pandas dataframes) 
# X = parkinsons.data.features 
# y = parkinsons.data.targets 
df = pd.read_csv("C:/Users/zacha/Downloads/parkinsons (1)/parkinsons.data")
print(df.head())

# Feature Matrix 
x_orig = df.drop(columns = ['status', 'name']).values 

# Data labels 
y_orig = df[['status']].values 

print("Shape of Feature Matrix:", x_orig.shape) 
print("Shape Label Vector:", y_orig.shape) 

# Separating the data points into positive and negative classes 
x_pos = np.array([x_orig[i] for i in range(len(x_orig)) 
                              if y_orig[i] == 1]) 
x_neg = np.array([x_orig[i] for i in range(len(x_orig)) 
                              if y_orig[i] == 0]) 

# Get feature names
feature_names = df.drop(columns = ['status', 'name']).columns.tolist()

# Plot scatter for each feature
for idx, feature_name in enumerate(feature_names):
    plt.figure(figsize=(12, 6))
    
    # Create scatter for positive cases
    plt.scatter(range(len(x_pos)), x_pos[:, idx], 
               color='blue', label='Positive', alpha=0.6)
    
    # Create scatter for negative cases
    plt.scatter(range(len(x_neg)), x_neg[:, idx],
               color='red', label='Negative', alpha=0.6)
    
    plt.xlabel('Sample Index', fontsize=10)
    plt.ylabel(feature_name, fontsize=10)
    plt.title(f'Scatter Plot of {feature_name}', fontsize=12)
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Import combinations from itertools
from itertools import combinations

# Generate combinations 31-60 of 2 features
feature_combinations = list(combinations(range(16), 2))[30:60]

# Plot each combination in a separate figure
for idx, (i, j) in enumerate(feature_combinations, 31):  # Start counting from 31
    plt.figure(figsize=(10, 8))
    
    # Plot positive points
    plt.scatter(x_pos[:, i], x_pos[:, j], 
               color='blue', label='Positive', alpha=0.6)
    
    # Plot negative points
    plt.scatter(x_neg[:, i], x_neg[:, j],
               color='red', label='Negative', alpha=0.6)
    
    plt.xlabel(feature_names[i], fontsize=10)
    plt.ylabel(feature_names[j], fontsize=10)
    plt.title(f'Feature Combination {idx}\n{feature_names[i]} vs {feature_names[j]}', fontsize=12)
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
