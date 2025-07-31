"""
random_forest.py

This script benchmarks the performance of a multi-output random forest regressor on quantum network data.
It loads density matrices, quantum states, and error parameters from CSV files, combines the features,
splits the data into training and test sets, and fits a multi-output random forest model to predict the error parameters.

The script prints the R² (coefficient of determination) for each output as well as the mean R² across all outputs.
Use this as a classical machine learning baseline to compare against neural network models.
"""

# --------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

# Load data
X_density = pd.read_csv('data/density_matricies.csv', header=None).values
X_states = pd.read_csv('data/states.csv', header=None).values
y_errors = pd.read_csv('data/errors.csv', header=None).values

# Combine features
X = np.concatenate([X_density, X_states], axis=1)
T = y_errors

# Split into train/test (same as neural net: 80% train, 20% test)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
T_train, T_test = T[:split_idx], T[split_idx:]

# Fit multi-output random forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
multi_rf = MultiOutputRegressor(rf)
multi_rf.fit(X_train, T_train)

# Predict and evaluate
T_pred = multi_rf.predict(X_test)
r2_per_output = r2_score(T_test, T_pred, multioutput='raw_values')
for i, r2 in enumerate(r2_per_output):
    print(f"R² for output {i}: {r2:.4f}")
print(f"Mean R² across all outputs: {np.mean(r2_per_output):.4f}")
