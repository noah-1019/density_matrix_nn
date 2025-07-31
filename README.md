# Quantum Network Error Parameter Prediction with Machine Learning
Author: Noah Plant
Date 07/31/2025

## Project Overview
This project aims to predict quantum network error parameters using machine learning models, specifically neural networks and classical random forests. The workflow involves simulating quantum networks, reconstructing quantum states and density matrices, generating large datasets, and benchmarking machine learning models to infer error parameters from quantum measurement data.

## Goals
The goal of this project is to see if a neural network can determine a quantum network's error configuration based off of the density matrix of the qubits at the end nodes. If a neural network can map a density matrix to an error configuration it would be a viable tool for Quantum Network Tomography. However, in this project the Neural Network and random forest did not perform well, even with fine tuning of hyperparamaters and various network configurations. This indicates that the density matrix and error configuration do not have a 1 to 1 correspondance and that a more sophisticated way to represent a quantum network is required. Perhaps distilling the density matrix down to a single value such as the von neumann entropy or purity will prevent the network from getting confused. 

## Motivation
Quantum networks are susceptible to various types of noise and errors as quantum information travels through different channels. Accurately estimating the error parameters of these channels is crucial for error correction, network optimization, and reliable quantum communication. Traditional methods for error estimation can be computationally intensive or require extensive measurements. This project explores whether machine learning can efficiently infer these error parameters from simulated measurement data and reconstructed quantum states.

## Workflow
1. **Data Generation**
   - Simulate quantum networks with configurable error parameters (X, Y, Z error probabilities for each channel).
   - Generate random quantum states for two qubits.
   - Simulate the evolution of these states through a network with three noisy channels.
   - Perform projective measurements in various Pauli bases.
   - Reconstruct the two-qubit density matrix from measurement outcomes.
   - Save the generated data (density matrices, quantum states, error parameters) as CSV files for ML training.

2. **Machine Learning Pipeline**
   - **Neural Network Model**: A multi-layer Keras neural network is trained to predict the 9 error parameters (3 per channel) from the combined density matrix and state data.
   - **Random Forest Baseline**: A multi-output random forest regressor is trained on the same data to provide a classical ML benchmark.
   - **Evaluation**: Both models are evaluated using R² (coefficient of determination) for each output and the mean R² across all outputs.
   - **Output Constraints**: The neural network output is constrained to the range (0, 0.5) to match the physical range of error probabilities.

3. **Results and Logging**
   - Trained models, training logs, and evaluation metrics are saved for reproducibility.
   - Model architecture and training curves are plotted and saved for analysis.
   - All hyperparameters and results are logged for future reference.

## File Descriptions
- `helper_functions.py`: Quantum state generation, network simulation, measurement, and density matrix reconstruction utilities.
- `generate_data.py`: Parallelized data generation script. Produces CSVs for density matrices, states, and error parameters.
- `train_neural_network.py`: Loads data, preprocesses, defines, trains, and evaluates the neural network. Saves model, plots, and logs.
- `random_forest.py`: Loads data, fits a multi-output random forest, and prints R² scores for benchmarking.
- `requirements.txt`: Lists all required Python packages.
- `main_job.sh`: Example SLURM batch script for running data generation on a cluster.

## Data Format
- **Density Matrices**: Each row contains the real and imaginary parts of the 4x4 density matrix, flattened.
- **States**: Each row contains the real and imaginary parts of the two 2x1 qubit state vectors, flattened.
- **Errors**: Each row contains the 9 error parameters (3 per channel: X, Y, Z error probabilities).

## How to Use
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate data:
   ```bash
   python generate_data.py
   ```
3. Train and evaluate the neural network:
   ```bash
   python train_neural_network.py
   ```
4. Benchmark with random forest:
   ```bash
   python random_forest.py
   ```

## Conclusions
- 



## Contact
Author: Noah Plant

For questions or collaboration, please contact the author or open an issue in the repository.
