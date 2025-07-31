import numpy as np
import random
import itertools

def generate_state(random_state=False):
    if random_state:
        # Generate a random quantum state
        theta = random.uniform(0, np.pi)
        phi = random.uniform(0, 2 * np.pi)
        state = np.array([[np.cos(theta / 2)], [np.exp(1j * phi) * np.sin(theta / 2)]])
        return state
    else:
        state = np.array([[1], [0]])
        return state



# Quantum Operators
def X_gate(state):
    """Applies the X gate to the given state."""
    return np.array([[0, 1], [1, 0]]) @ state
def Y_gate(state):
    """Applies the Y gate to the given state."""
    return np.array([[0, -1j], [1j, 0]]) @ state
def Z_gate(state):
    """Applies the Z gate to the given state."""
    return np.array([[1, 0], [0, -1]]) @ state
def H_gate(state):
    """Applies the H gate to the given state."""
    return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]) @ state
def S_gate(state):
    """Applies the S gate to the given state."""
    return np.array([[1, 0], [0, 1j]]) @ state

# Measurement Operators

def measure(states, bases):
    """
    Measures the joint state of two qubits in the specified Pauli bases.
    
    Parameters
    ----------
    states : list of np.ndarray
        List of two 2x1 state vectors.
    bases : list of str
        List of two basis labels, e.g., ['x', 'z'].

    Returns
    -------
    int
        The measurement outcome (+1 or -1) corresponding to the joint Pauli operator.
    """
    # Define Pauli matrices
    pauli_dict = {
        'i': np.array([[1, 0], [0, 1]]),
        'x': np.array([[0, 1], [1, 0]]),
        'y': np.array([[0, -1j], [1j, 0]]),
        'z': np.array([[1, 0], [0, -1]])
    }

    # Tensor product of the two states
    psi = np.kron(states[0], states[1])  # shape (4, 1)

    # Tensor product of the two measurement operators
    opA = pauli_dict[bases[0].lower()]
    opB = pauli_dict[bases[1].lower()]
    M = np.kron(opA, opB)  # shape (4, 4)


    # Find eigenvalues and eigenvectors of the measurement operator
    eigvals, eigvecs = np.linalg.eigh(M)

    # Project state onto eigenbasis and get probabilities
    projections = np.abs(eigvecs.conj().T @ psi).flatten()**2

    # Choose outcome according to probabilities
    outcome_idx = np.random.choice(len(eigvals), p=projections)
    outcome = eigvals[outcome_idx].real  # Should be +1 or -1

    return int(np.sign(outcome))
    



## Network Simulation Functions
def simulate_network(states, errors, measurement_bases):
    """
    Simulates the evolution and measurement of two qubits through a quantum network with three noisy 
    channels and configurable measurement bases.

    Network Configuration:
    ----------------------
        The network is structured as follows:

            [1] (Qubit A)         [1] (Qubit B)
                │                      │
        Channel 1 (errors[0])   Channel 1 (errors[0])
                │                      │
               [X] (Intermediate Node)
              /   \
             /     \
    Channel 2      Channel 3
    (errors[1])    (errors[2])
      /                \
    [2] (Qubit A)     [3] (Qubit B)
   Measurement A     Measurement B

        - Qubit A travels through Channel 1 and Channel 2 to node [2].
        - Qubit B travels through Channel 1 and Channel 3 to node [3].
        - Each channel introduces independent X, Y, Z errors with probabilities specified in the errors list.
        - At the end, each qubit is measured in a user-specified basis.

    Parameters
    ----------
    states : list of np.ndarray
        List containing the initial quantum states of the two qubits (Qubit A and Qubit B).
        Each state should be a 2x1 numpy array (column vector).

    errors : list of list or np.ndarray
        List of three error probability vectors, one for each channel:
            errors[0]: [p_X, p_Y, p_Z] for Channel 1
            errors[1]: [p_X, p_Y, p_Z] for Channel 2
            errors[2]: [p_X, p_Y, p_Z] for Channel 3
        Each vector contains the probabilities for X, Y, and Z errors, respectively.

    measurement_bases : list of str
        List of two strings specifying the measurement basis for each qubit at the output nodes:
            measurement_bases[0]: Basis for Qubit A (at node [2]), e.g., 'x', 'y', or 'z'
            measurement_bases[1]: Basis for Qubit B (at node [3]), e.g., 'x', 'y', or 'z'
        The basis is case-insensitive.

    Returns
    -------
    tuple (int, int)
        A tuple containing the measurement outcomes for Qubit A and Qubit B, respectively.
        Each outcome is either -1 or 1, corresponding to the result of the projective measurement in the specified basis.

    Raises
    ------
    ValueError
        If an unknown measurement basis is provided for either qubit.

    Example
    -------
    >>> stateA = np.array([[1], [0]])
    >>> stateB = np.array([[1], [0]])
    >>> errors = [ [0.1, 0.1, 0.1], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05] ]
    >>> simulate_network([stateA, stateB], errors, ['z', 'x'])
    (1, 0)
    """
    qubitA= states[0]
    qubitB= states[1]

    # -------------------------------------------------------------
    # Simulate Channel 1
    # -------------------------------------------------------------

    c1_error=errors[0]
    seed_number_1=np.random.rand(3)

    if seed_number_1[0] < c1_error[0]:
        qubitA = X_gate(qubitA)
        qubitB = X_gate(qubitB)
    if seed_number_1[1] < c1_error[1]:
        qubitA = Y_gate(qubitA)
        qubitB = Y_gate(qubitB)
    if seed_number_1[2] < c1_error[2]:
        qubitA = Z_gate(qubitA)
        qubitB = Z_gate(qubitB)



    # -------------------------------------------------------------
    # Simulate Channel 2 and Channel 3
    # -------------------------------------------------------------
    c2_error=errors[1]
    c3_error=errors[2]
    seed_number_1=np.random.rand(3)
    seed_number_2=np.random.rand(3)

    # Qubit A
    if seed_number_1[0] < c2_error[0]:
        qubitA = X_gate(qubitA)
    if seed_number_1[1] < c2_error[1]:
        qubitA = Y_gate(qubitA)
    if seed_number_1[2] < c2_error[2]:
        qubitA = Z_gate(qubitA)

    # Qubit B
    if seed_number_2[0] < c3_error[0]:
        qubitB = X_gate(qubitB)
    if seed_number_2[1] < c3_error[1]:
        qubitB = Y_gate(qubitB)
    if seed_number_2[2] < c3_error[2]:
        qubitB = Z_gate(qubitB)

    # -------------------------------------------------------------
    # Measurement
    # -------------------------------------------------------------
    # measurement_bases should be a list like ['x', 'z'] or ['y', 'y']
    result_a = measure([qubitA, qubitB], measurement_bases) 
    return result_a






def expected_values(states,errors,iterations):
    """
    Calculates the expected values for each measurement outcome based on the quantum states and errors.

    Parameters
    ----------
    states : list of np.ndarray
        List containing the quantum states of the two qubits.
    errors : list of list or np.ndarray
        List of three error probability vectors, one for each channel.
    iterations : int
        Number of iterations for averaging.

    Returns
    -------
    list
        List of expected values for each measurement outcome.
    """
    # Placeholder for expected values calculation
    expected_values = []
    
    # All length-2 Pauli strings as lists of two basis labels (for simulate_network)
    paulis = ['i','x', 'y', 'z']
    measurement_bases = [ [a, b] for a, b in itertools.product(paulis, repeat=2) ]



    stateA = states[0]
    stateB = states[1]
    for basis in measurement_bases:
        basis_expected_value = 0
        for i in range(iterations):
            basis_expected_value += simulate_network([stateA, stateB],errors=errors,
                                                        measurement_bases=basis)
        expected_values.append(basis_expected_value/iterations)

        
    return expected_values


def density_matrix(expected_values):
    """
    Reconstructs the two-qubit density matrix from a list of expected values for all two-qubit Pauli operator measurements.

    Parameters
    ----------
    expected_values : list of float
        List of expectation values for each two-qubit Pauli operator, ordered according to the measurement_bases
        generated by itertools.product(['i', 'x', 'y', 'z'], repeat=2).

    Returns
    -------
    np.ndarray
        The reconstructed 4x4 density matrix as a NumPy array of complex numbers.

    Notes
    -----
    The density matrix is reconstructed using the Pauli basis expansion:
        rho = (1/4) * sum_{i,j} expected_values_{i,j} * (sigma_i ⊗ sigma_j)
    where sigma_i, sigma_j ∈ {I, X, Y, Z} and the expected values are the measured expectation values
    for each corresponding Pauli product.
    """
    
    pauli_dict = {
        'i': np.eye(2),
        'x': np.array([[0, 1], [1, 0]]),
        'y': np.array([[0, -1j], [1j, 0]]),
        'z': np.array([[1, 0], [0, -1]])
    }
    paulis = ['i','x', 'y', 'z']
    measurement_bases = [ [a, b] for a, b in itertools.product(paulis, repeat=2) ]

    # Reconstruct the density matrix
    density_matrix = np.zeros((4, 4), dtype=complex)

    for i, basis in enumerate(measurement_bases):
        expected_value = expected_values[i]
        pauli_matrix = np.kron(pauli_dict[basis[0]], pauli_dict[basis[1]])
        density_matrix += (expected_value) * pauli_matrix

    density_matrix /= 2**(2) # Normalize the density matrix
    return density_matrix


# Test measure function
if __name__ == "__main__":
    # Example states
    stateA = np.array([[1], [0]])
    stateB = np.array([[1], [0]])

    # Example errors
    errors = [[0.1, 0.1, 0.1], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]

    # Example measurement bases
    measurement_bases = ['z', 'z']
    outcome=0

    #for i in range(1000):
        #outcome += measure([stateA, stateB], measurement_bases)

    # Simulate network
    outcome = simulate_network([stateA, stateB], errors, measurement_bases)
    print("Measurement outcomes:", outcome)