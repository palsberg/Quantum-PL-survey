"""Shared helpers for PennyLane-based implementations."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pennylane as qml


def _run_qnode(num_wires: int, builder: Callable[[], None]) -> np.ndarray:
    """Execute a qnode that prepares |0…0⟩ and then runs builder()."""
    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev)
    def circuit():
        builder()
        return qml.state()

    return circuit()


def _tfim_hamiltonian(num_sites: int, J: float, h: float) -> qml.Hamiltonian:
    """Return H_TFIM as a PennyLane Hamiltonian."""
    coeffs = []
    ops = []
    for i in range(num_sites - 1):
        coeffs.append(J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(num_sites):
        coeffs.append(h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


def _heis_hamiltonian(num_sites: int, J: float, field: float) -> qml.Hamiltonian:
    """Return H_XXX as a PennyLane Hamiltonian."""
    coeffs = []
    ops = []
    for i in range(num_sites - 1):
        for op_cls in (qml.PauliX, qml.PauliY, qml.PauliZ):
            coeffs.append(J)
            ops.append(op_cls(i) @ op_cls(i + 1))
    for i in range(num_sites):
        coeffs.append(field)
        ops.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeffs, ops)


def tfim_trotter_state(num_sites: int, J: float, h: float, total_time: float, steps: int) -> np.ndarray:
    """Simulate TFIM using PennyLane's ApproxTimeEvolution template."""
    H = _tfim_hamiltonian(num_sites, J, h)

    def build():
        qml.ApproxTimeEvolution(H, time=total_time, n=steps)

    return _run_qnode(num_sites, build)


def heis_trotter_state(num_sites: int, J: float, field: float, total_time: float, steps: int) -> np.ndarray:
    """Simulate Heisenberg XXX with a field using ApproxTimeEvolution."""
    H = _heis_hamiltonian(num_sites, J, field)

    def build():
        qml.ApproxTimeEvolution(H, time=total_time, n=steps)

    return _run_qnode(num_sites, build)
