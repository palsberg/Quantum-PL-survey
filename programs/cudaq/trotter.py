import cudaq
from cudaq import spin
import numpy as np
import time
import sys
from typing import List

@cudaq.kernel
def trotterize(n_sites: int, t: float, K: int,
                 coefficients: List[complex], words: List[cudaq.pauli_word]):
    qubits = cudaq.qvector(n_sites)
    dt = t / K
    for k in range(K):
        for i in range(len(coefficients)):
            exp_pauli(coefficients[i].real * dt, qubits, words[i])

def create_hamiltonian_tfim(n_sites: int, coupling: float, field: float):
    """Create the TFIM Hamiltonian operator"""
    H = 0
    for i in range(0, n_sites-1):
        H += coupling * spin.z(i) * spin.z(i + 1)
    for i in range(0, n_sites):
        H += field * spin.x(i)
    return H

def create_hamiltonian_heisenberg(n_sites: int, coupling: float, field: float):
    """Create the Heisenberg Hamiltonian operator"""
    H = 0
    for i in range(0, n_sites-1):
        H += coupling * spin.x(i) * spin.x(i + 1)
        H += coupling * spin.y(i) * spin.y(i + 1)
        H += coupling * spin.z(i) * spin.z(i + 1)
    for i in range(0, n_sites):
        H += field * spin.z(i)
    return H




# Parameters
n_sites = 3
ham_type = "tfim"
coupling = 2.0
field = 0.5
K = 10
t = 1.0

# Create Hamiltonian
if ham_type == "tfim":
    H = create_hamiltonian_tfim(n_sites, coupling, field)
elif ham_type == "heis":
    H = create_hamiltonian_heisenberg(n_sites, coupling, field)
else:
    raise ValueError("Invalid Hamiltonian type. Choose 'heis' or 'tfim'.")

# Extract coefficients and words
coefficients = [term.evaluate_coefficient() for term in H]
words = [term.get_pauli_word(H.qubit_count) for term in H]

# Time evolution
start_time = time.time()
state = cudaq.get_state(trotterize, n_sites, t, K, coefficients, words)
total_time = time.time() - start_time
state = np.array(state)






# Ground truth
from reference_hamiltonians import *
if ham_type == 'tfim':
    ref_H = tfim_hamiltonian(n_sites, coupling, field)
elif ham_type == 'heis':
    ref_H = heis_xxx_hamiltonian(n_sites, coupling, field)
ref_state = time_evolve(ref_H, zero_state(n_sites), t)

def compute_fidelity(state: np.ndarray, reference: np.ndarray) -> float:
    state = normalize(state)
    reference = normalize(reference)
    return float(np.abs(np.vdot(reference, state)) ** 2)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.complex128).flatten()
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero vector")
    return vec / norm

fidelity = compute_fidelity(state, ref_state)
print(f'{fidelity=}')
