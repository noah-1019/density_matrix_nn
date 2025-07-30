# Train Neural Network for Quantum Network Entropy Prediction
# Author: Noah Plant
# Date: 2023-10-01
# Description: This script trains a neural network to predict quantum network entropy based on density matrices.

# --------------------------------------------------------------
## Import necessary libraries
# --------------------------------------------------------------
import csv
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------
## Load in data from CSV file
# --------------------------------------------------------------

# Load in density matrices (Inputs)
density_matrix = pd.read_csv('data/density_matricies.csv',header=None)
# Load in states (Inputs)
states = pd.read_csv('data/states.csv',header=None)
# Load in errors (Targets)
errors = pd.read_csv('data/errors.csv',header=None)


# --------------------------------------------------------------
## Preprocess the data
# --------------------------------------------------------------

## Convert dataframes to numpy arrays
X_density = density_matrix.values
X_states = states.values
y_errors = errors.values

## Seperate the data into inputs and targets
X = np.concatenate([X_density, X_states], axis=1) 
T = y_errors
#print(f"Input shapes: {X.shape}, Target shape: {T.shape}"): # should be (1000, 40) and (1000, 9)

## Split the data into training and evaluation sets
X_train, X_test = X[:800], X[800:]
T_train, T_test = T[:800], T[800:]
#print(f"Training set shape: {X_train.shape}, {T_train.shape}"): # Should be (800, 40) and (800, 9)
#print(f"Test set shape: {X_test.shape}, {T_test.shape}"): # Should be (200, 40) and (200, 9)

# --------------------------------------------------------------
## Create the neural network model
# --------------------------------------------------------------
# Define the architecture of the neural network


model = Sequential([
    Dense(64, activation='relu', input_shape=(40,)),  # Input layer with 40 features (32 from density matrix, 8 from states)
    Dropout(0.2),                                      # Optional regularization
    Dense(128, activation='relu'),                     # Hidden layer
    Dropout(0.2),
    Dense(64, activation='relu'),                      # Hidden layer
    Dense(9, activation='linear')                      # Output layer for regression
])

# Choose the activation functions

# Input Layer: ReLU
# - ReLU activation is commonly used for input layers to introduce non-linearity

# Hidden Layers: ReLU
# Hidden Layers: ReLU
# - ReLU activation is commonly used for hidden layers to introduce non-linearity

# Output Layer: Linear (for regression)
# - Output layer uses linear activation to predict continuous values (errors)



# --------------------------------------------------------------
## Compile the model
# --------------------------------------------------------------
model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanAbsoluteError()])

# --------------------------------------------------------------
## Train the model
# --------------------------------------------------------------

# Fit the model to the training data
# Train the model and store the training history
history = model.fit(
    X_train, T_train,
    validation_data=(X_test, T_test),
    epochs=3_000,
    batch_size=64
)


# --------------------------------------------------------------
## Evaluate the model
# --------------------------------------------------------------
# Evaluate the model on the validation set
# Print the evaluation metrics

# --------------------------------------------------------------
## Save the model
# --------------------------------------------------------------


# Plot training & validation loss and MAE
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Mean Absolute Error')

plt.tight_layout()
plt.show()