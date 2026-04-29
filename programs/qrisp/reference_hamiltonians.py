"""
Reference Hamiltonians for the NumPy-based correctness harness.

The functions below build dense matrix representations of the
Transverse-Field Ising Model (TFIM) and the Heisenberg XXX chain
with a longitudinal field, and provide utilities for simulating
time evolution on small instances.
"""

from __future__ import annotations

from functools import reduce
from typing import Sequence

import numpy as np


PAULI_I = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def kron_many(operators: Sequence[np.ndarray]) -> np.ndarray:
    """Kronecker product of an ordered sequence of single-qubit operators."""

    def _kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.kron(a, b)

    return reduce(_kron, operators)


def tfim_hamiltonian(num_sites: int, J: float, h: float) -> np.ndarray:
    r"""
    Build the TFIM Hamiltonian
    $$H_{\text{TFIM}} = J \sum_{i=1}^{n-1} Z_i Z_{i+1} + h \sum_{i=1}^{n} X_i.$$
    """
    dim = 2 ** num_sites
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(num_sites - 1):
        ops = [PAULI_I] * num_sites
        ops[i] = PAULI_Z
        ops[i + 1] = PAULI_Z
        H += J * kron_many(ops)
    for i in range(num_sites):
        ops = [PAULI_I] * num_sites
        ops[i] = PAULI_X
        H += h * kron_many(ops)
    return H


def heis_xxx_hamiltonian(num_sites: int, J: float, field: float) -> np.ndarray:
    r"""
    Build the Heisenberg XXX Hamiltonian with a longitudinal field:
    $$H_{\text{XXX}} = J \sum_{i=1}^{n-1} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    + field \sum_{i=1}^{n} Z_i.$$
    """
    dim = 2 ** num_sites
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(num_sites - 1):
        for pauli in (PAULI_X, PAULI_Y, PAULI_Z):
            ops = [PAULI_I] * num_sites
            ops[i] = pauli
            ops[i + 1] = pauli
            H += J * kron_many(ops)
    for i in range(num_sites):
        ops = [PAULI_I] * num_sites
        ops[i] = PAULI_Z
        H += field * kron_many(ops)
    return H


def time_evolution_operator(H: np.ndarray, t: float) -> np.ndarray:
    """Return the unitary exp(-i H t) using scipy-free diagonalization."""
    eigvals, eigvecs = np.linalg.eigh(H)
    phases = np.exp(-1j * eigvals * t)
    return eigvecs @ np.diag(phases) @ np.conjugate(eigvecs.T)


def zero_state(num_sites: int) -> np.ndarray:
    """Return |0...0> as a dense statevector."""
    state = np.zeros(2**num_sites, dtype=np.complex128)
    state[0] = 1.0
    return state


def time_evolve(H: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    """Return exp(-i H t) |psi0>."""
    U = time_evolution_operator(H, t)
    return U @ psi0


__all__ = [
    "tfim_hamiltonian",
    "heis_xxx_hamiltonian",
    "time_evolution_operator",
    "time_evolve",
    "zero_state",
]
