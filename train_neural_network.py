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
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import io

# --------------------------------------------------------------
## Set parameters
# --------------------------------------------------------------
batch_size = 32
learning_rate = 0.003
num_epochs = 10_000
patience = 500  # Early stopping patience



#----------------------------------------------------------------
## Get Path information
#----------------------------------------------------------------
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"nn_models/nn_model_{timestamp}_epochs{num_epochs}"
model_path = f'{folder_name}/model.keras'
log_path = f'{folder_name}/training_log.txt'

print(f"Model will be saved to {model_path}")
print(f"Training logs will be saved to {log_path}")







# --------------------------------------------------------------
## Load in data from CSV file
# --------------------------------------------------------------

# Load in density matrices (Inputs)
density_matrix = pd.read_csv('data/density_matricies.csv',header=None)
# Load in states (Inputs)
states = pd.read_csv('data/states.csv',header=None)
# Load in errors (Targets)
errors = pd.read_csv('data/errors.csv',header=None)

# Load in test data
test_density_matrix = pd.read_csv('data/test_density_matricies.csv',header=None)
test_states = pd.read_csv('data/test_states.csv',header=None)
test_errors = pd.read_csv('data/test_errors.csv',header=None)


# --------------------------------------------------------------
## Preprocess the data
# --------------------------------------------------------------

## Convert dataframes to numpy arrays
X_density = density_matrix.values
X_states = states.values
y_errors = errors.values

## Convert test dataframes to numpy arrays
X_test_density = test_density_matrix.values
X_test_states = test_states.values
y_test_errors = test_errors.values


## Seperate the data into inputs and targets
X = np.concatenate([X_density, X_states], axis=1) 
T = y_errors
#print(f"Input shapes: {X.shape}, Target shape: {T.shape}"): # should be (1000, 40) and (1000, 9)

## Convert the test data into inputs and targets
X_test = np.concatenate([X_test_density, X_test_states], axis=1)
T_test = y_test_errors

## Split the data into training and evaluation sets
X_train, X_test = X[:8000], X[8000:]
T_train, T_test = T[:8000], T[8000:]
#print(f"Training set shape: {X_train.shape}, {T_train.shape}"): # Should be (8000, 40) and (8000, 9)
#print(f"Test set shape: {X_test.shape}, {T_test.shape}"): # Should be (2000, 40) and (2000, 9)

# --------------------------------------------------------------
## Create the neural network model
# --------------------------------------------------------------
# Define the architecture of the neural network


model = Sequential([
    Dense(64, activation='relu', input_shape=(40,)),  # Input layer with 40 features (32 from density matrix, 8 from states)
    BatchNormalization(),                              # Optional normalization layer
    Dropout(0.3),                                      # Optional regularization
    Dense(128, activation='relu'),                     # Hidden layer
    BatchNormalization(),                              # Optional normalization layer
    Dense(256, activation='relu'),                     # Hidden layer
    BatchNormalization(),                              # Optional normalization layer
    Dropout(0.3),                                      # Optional regularization
    Dense(128, activation='relu'),                     # Hidden layer
    BatchNormalization(),                              # Optional normalization layer
    Dropout(0.3),
    Dense(64, activation='relu'),                      # Hidden layer
    Dense(9, activation='sigmoid'),  # restricts output to (0, 1)
    Lambda(lambda x: x * 0.5)        # scales output to (0, 0.5)                   # Output layer for regression
])

# Choose the activation functions

# Input Layer: ReLU
# - ReLU activation is commonly used for input layers to introduce non-linearity

# Hidden Layers: ReLU
# Hidden Layers: ReLU
# - ReLU activation is commonly used for hidden layers to introduce non-linearity

# Output Layer: Linear (for regression)
# - Output layer uses a sigmoid activation function to restrict the output to (0, 1)
# - A Lambda layer is used to scale the output to (0, 0.5) for the error values
# - This is suitable for regression tasks where the target values are continuous 



# --------------------------------------------------------------
## Compile the model
# --------------------------------------------------------------
# Create the optimizer
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with loss function and metrics
model.compile(loss=MeanSquaredError(), optimizer=optimizer, metrics=[MeanAbsoluteError()])

# Print the model summary
model.summary()


# --------------------------------------------------------------
## Train the model
# --------------------------------------------------------------

# Fit the model to the training data
# Train the model and store the training history

early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(f'{folder_name}/best_model.keras', save_best_only=True)

history = model.fit(
    X_train, T_train,
    validation_data=(X_test, T_test),
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, model_checkpoint]
)


# --------------------------------------------------------------
## Evaluate the model
# --------------------------------------------------------------
# Evaluate the model on the validation set
# Print the evaluation metrics
T_pred = model.predict(X_test)
results = model.evaluate(X_test, T_test, verbose=1)
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# Compute R² for each output (column)
r2_per_output = r2_score(T_test, T_pred, multioutput='raw_values')
for i, r2 in enumerate(r2_per_output):
    print(f"R² for output {i}: {r2:.4f}")

# Optionally, print the mean R² across all outputs
print(f"Mean R² across all outputs: {np.mean(r2_per_output):.4f}")


# --------------------------------------------------------------
## Save the model
# --------------------------------------------------------------
# Get current date and time for the filename

model.save(model_path)
print(f"Model saved to {model_path}")

# Save the model architecture as an image
plot_model(model, to_file=f'{folder_name}/model_architecture.png', show_shapes=True, show_layer_names=True)
print(f"Model architecture saved to {folder_name}/model_architecture.png")

# save the training logs.
# Capture model summary as a string
stream = io.StringIO()
model.summary(print_fn=lambda x: stream.write(x + "\n"))
model_summary_str = stream.getvalue()
stream.close()

# Save the training & validation loss and MAE plots as an image
plot_filename = f"{folder_name}/training_plot.png"
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
plt.savefig(plot_filename)

# Add the plot filename to your log entry
log_entry = f"""
==========================
Timestamp: {timestamp}
Model file: {model_path}

Hyperparameters:
- Batch size: {batch_size}
- Learning rate: {learning_rate}
- Num epochs: {num_epochs}
- Early stopping patience: {patience}

Model Summary:
{model_summary_str}

R² per output: {r2_per_output}
Mean R²: {np.mean(r2_per_output):.4f}

Training/Validation plot: {plot_filename}
==========================

"""

# Save the log as a new file

with open(log_path, "w") as f:
    f.write(log_entry)
print(f"Training log saved to {log_path}")