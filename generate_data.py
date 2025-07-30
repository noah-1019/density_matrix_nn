import numpy as np
import random
import itertools
import helper_functions as hf
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

data_points = 2_00  # Set to desired number of data points

def generate_one(trial):
    # Generate random quantum states
    stateA = hf.generate_state(random_state=True)
    stateB = hf.generate_state(random_state=True)
    states = [stateA, stateB]

    # Generate random error configurations
    errors = [[random.uniform(0, 0.4) for _ in range(3)] for _ in range(3)]
    expected_values = hf.expected_values(states, errors=errors, iterations=1000)
    density_matrix = hf.density_matrix(expected_values=expected_values)
    return states, density_matrix, errors

if __name__ == "__main__":
    all_states = []
    density_matricies = []
    all_errors = []

    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_one, trial) for trial in range(data_points)]
        for future in as_completed(futures):
            states, density_matrix, errors = future.result()
            all_states.append(states)
            density_matricies.append(density_matrix)
            all_errors.append(errors)
    

    # Save Data to CSV files
    with open('data/test_density_matricies.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for matrix in density_matricies:
            row = np.concatenate([matrix.real.flatten(), matrix.imag.flatten()])
            writer.writerow(row)

    with open('data/test_states.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for state in all_states:
            row = np.concatenate([state[0].real.flatten(), state[0].imag.flatten(),
                                  state[1].real.flatten(), state[1].imag.flatten()])
            writer.writerow(row)

    with open('data/test_errors.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for error in all_errors:
            row = np.concatenate(error).flatten()
            writer.writerow(row)



