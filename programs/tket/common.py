"""Shared helpers for pytket implementations."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from pytket import Circuit
from pytket.extensions.qiskit import tk_to_qiskit
from qiskit.quantum_info import Statevector


def simulate_statevector(circuit: Circuit) -> np.ndarray:
    qc = tk_to_qiskit(circuit.copy())
    state = Statevector.from_instruction(qc).data
    return np.asarray(state, dtype=np.complex128)


def trotterize_tfim(
    num_sites: int, J: float, h: float, total_time: float, steps: int
) -> Tuple[Circuit, Sequence[int]]:
    circ = Circuit(num_sites)
    dt = total_time / steps
    for _ in range(steps):
        for i in range(num_sites - 1):
            # ZZPhase(α) implements exp(-i (π/2) α Z⊗Z); choose α so that
            # (π/2) α = J * dt, i.e. α = 2 J dt / π.
            circ.ZZPhase(2 * J * dt / np.pi, i, i + 1)
        for i in range(num_sites):
            # Rx(α) implements exp(-i (π/2) α X); choose α so that
            # (π/2) α = h * dt, i.e. α = 2 h dt / π.
            circ.Rx(2 * h * dt / np.pi, i)
    return circ, list(range(num_sites))


def trotterize_heisenberg_xxx(
    num_sites: int, J: float, field: float, total_time: float, steps: int
) -> Tuple[Circuit, Sequence[int]]:
    circ = Circuit(num_sites)
    dt = total_time / steps
    for _ in range(steps):
        for i in range(num_sites - 1):
            circ.XXPhase(2 * J * dt, i, i + 1)
            circ.YYPhase(2 * J * dt, i, i + 1)
            circ.ZZPhase(2 * J * dt, i, i + 1)
        for i in range(num_sites):
            circ.Rz(2 * field * dt, i)
    return circ, list(range(num_sites))
