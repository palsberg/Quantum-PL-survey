"""Shared helpers for the Qiskit Hamiltonian-simulation programs."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector


def _basis_change(qc: QuantumCircuit, qubit, axis: str, inverse: bool) -> None:
    """Rotate so that Z-measurements correspond to the chosen axis."""
    axis = axis.upper()
    if axis == "X":
        qc.h(qubit)
    elif axis == "Y":
        if inverse:
            qc.s(qubit)
            qc.h(qubit)
        else:
            qc.h(qubit)
            qc.sdg(qubit)


def _apply_zz_rotation(qc: QuantumCircuit, qa, qb, theta: float) -> None:
    """Implement exp(-i * theta * Z⊗Z)."""
    qc.cx(qa, qb)
    qc.rz(2 * theta, qb)
    qc.cx(qa, qb)


def apply_two_qubit_rotation(qc: QuantumCircuit, qa, qb, theta: float, axis: str) -> None:
    """Apply exp(-i * theta * σ_axis⊗σ_axis)."""
    axis = axis.upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported axis {axis}")
    _basis_change(qc, qa, axis, inverse=False)
    _basis_change(qc, qb, axis, inverse=False)
    _apply_zz_rotation(qc, qa, qb, theta)
    _basis_change(qc, qa, axis, inverse=True)
    _basis_change(qc, qb, axis, inverse=True)


def apply_single_qubit_rotation(qc: QuantumCircuit, qubit, theta: float, axis: str) -> None:
    """Apply exp(-i * theta * σ_axis)."""
    axis = axis.upper()
    if axis == "Z":
        qc.rz(2 * theta, qubit)
    elif axis == "X":
        qc.h(qubit)
        qc.rz(2 * theta, qubit)
        qc.h(qubit)
    elif axis == "Y":
        qc.sdg(qubit)
        qc.h(qubit)
        qc.rz(2 * theta, qubit)
        qc.h(qubit)
        qc.s(qubit)
    else:
        raise ValueError(f"Unsupported axis {axis}")


def simulate_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Return the |0…0> → final statevector for the given circuit."""
    state = Statevector.from_instruction(qc)
    return np.asarray(state.data, dtype=np.complex128)


def trotterize_tfim(
    num_sites: int, J: float, h: float, total_time: float, steps: int
) -> Tuple[QuantumCircuit, Sequence[int]]:
    """Build a first-order Lie–Trotter circuit for the TFIM Hamiltonian."""
    qr = QuantumRegister(num_sites, "q")
    qc = QuantumCircuit(qr, name="tfim_trotter")
    dt = total_time / steps
    for _ in range(steps):
        for i in range(num_sites - 1):
            apply_two_qubit_rotation(qc, qr[i], qr[i + 1], J * dt, axis="Z")
        for i in range(num_sites):
            apply_single_qubit_rotation(qc, qr[i], h * dt, axis="X")
    return qc, qr


def trotterize_heisenberg_xxx(
    num_sites: int, J: float, field: float, total_time: float, steps: int
) -> Tuple[QuantumCircuit, Sequence[int]]:
    """Build a Lie–Trotter circuit for the Heisenberg XXX chain with a field."""
    qr = QuantumRegister(num_sites, "q")
    qc = QuantumCircuit(qr, name="heisenberg_trotter")
    dt = total_time / steps
    for _ in range(steps):
        for i in range(num_sites - 1):
            apply_two_qubit_rotation(qc, qr[i], qr[i + 1], J * dt, axis="X")
            apply_two_qubit_rotation(qc, qr[i], qr[i + 1], J * dt, axis="Y")
            apply_two_qubit_rotation(qc, qr[i], qr[i + 1], J * dt, axis="Z")
        for i in range(num_sites):
            apply_single_qubit_rotation(qc, qr[i], field * dt, axis="Z")
    return qc, qr

