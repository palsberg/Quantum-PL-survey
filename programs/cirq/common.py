"""
Shared helpers for the Cirq Hamiltonian-simulation programs.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cirq
import numpy as np


def _apply_basis_change(circuit: cirq.Circuit, qubit: cirq.Qid, axis: str, inverse: bool) -> None:
    """Apply basis change that maps Z to the requested Pauli axis (or inverse)."""
    if axis == "X":
        circuit.append(cirq.H(qubit))
    elif axis == "Y":
        if inverse:
            circuit.append(cirq.S(qubit))
            circuit.append(cirq.H(qubit))
        else:
            circuit.append(cirq.H(qubit))
            circuit.append(cirq.S(qubit) ** -1)


def _apply_zz_rotation(circuit: cirq.Circuit, qa: cirq.Qid, qb: cirq.Qid, theta: float) -> None:
    """Implement exp(-i * theta * Z⊗Z)."""
    circuit.append(cirq.CNOT(qa, qb))
    circuit.append(cirq.rz(2 * theta).on(qb))
    circuit.append(cirq.CNOT(qa, qb))


def apply_two_qubit_rotation(
    circuit: cirq.Circuit, qa: cirq.Qid, qb: cirq.Qid, theta: float, axis: str
) -> None:
    """Apply exp(-i * theta * σ_axis⊗σ_axis)."""
    axis = axis.upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported axis {axis}")
    _apply_basis_change(circuit, qa, axis, inverse=False)
    _apply_basis_change(circuit, qb, axis, inverse=False)
    _apply_zz_rotation(circuit, qa, qb, theta)
    _apply_basis_change(circuit, qa, axis, inverse=True)
    _apply_basis_change(circuit, qb, axis, inverse=True)


def apply_single_qubit_rotation(
    circuit: cirq.Circuit, qubit: cirq.Qid, theta: float, axis: str
) -> None:
    """Apply exp(-i * theta * σ_axis) on a single qubit."""
    axis = axis.upper()
    if axis == "Z":
        circuit.append(cirq.rz(2 * theta).on(qubit))
    elif axis == "X":
        circuit.append(cirq.H(qubit))
        circuit.append(cirq.rz(2 * theta).on(qubit))
        circuit.append(cirq.H(qubit))
    elif axis == "Y":
        circuit.append(cirq.S(qubit) ** -1)
        circuit.append(cirq.H(qubit))
        circuit.append(cirq.rz(2 * theta).on(qubit))
        circuit.append(cirq.H(qubit))
        circuit.append(cirq.S(qubit))
    else:
        raise ValueError(f"Unsupported axis {axis}")


def simulate_statevector(circuit: cirq.Circuit, qubits: Sequence[cirq.Qid]) -> np.ndarray:
    """Simulate the given circuit initialized to |0...0> and return the statevector."""
    sim = cirq.Simulator(dtype=np.complex128)
    result = sim.simulate(circuit, qubit_order=qubits)
    return np.asarray(result.final_state_vector, dtype=np.complex128)


def trotterize_tfim(
    num_sites: int, J: float, h: float, time: float, steps: int
) -> tuple[cirq.Circuit, Sequence[cirq.Qid]]:
    """Construct a Lie–Trotter circuit for the TFIM Hamiltonian."""
    qubits = cirq.LineQubit.range(num_sites)
    circuit = cirq.Circuit()
    dt = time / steps
    for _ in range(steps):
        for i in range(num_sites - 1):
            apply_two_qubit_rotation(circuit, qubits[i], qubits[i + 1], J * dt, axis="Z")
        for i in range(num_sites):
            apply_single_qubit_rotation(circuit, qubits[i], h * dt, axis="X")
    return circuit, qubits


def trotterize_heisenberg_xxx(
    num_sites: int, J: float, field: float, time: float, steps: int
) -> tuple[cirq.Circuit, Sequence[cirq.Qid]]:
    """Construct a Lie–Trotter circuit for the Heisenberg XXX Hamiltonian with a field."""
    qubits = cirq.LineQubit.range(num_sites)
    circuit = cirq.Circuit()
    dt = time / steps
    for _ in range(steps):
        for i in range(num_sites - 1):
            apply_two_qubit_rotation(circuit, qubits[i], qubits[i + 1], J * dt, axis="X")
            apply_two_qubit_rotation(circuit, qubits[i], qubits[i + 1], J * dt, axis="Y")
            apply_two_qubit_rotation(circuit, qubits[i], qubits[i + 1], J * dt, axis="Z")
        for i in range(num_sites):
            apply_single_qubit_rotation(circuit, qubits[i], field * dt, axis="Z")
    return circuit, qubits


def extract_state_from_lcu_result(
    state_vector: np.ndarray, num_system: int, num_index: int, phase_target: int = 1
) -> np.ndarray:
    """Project onto ancilla |0...0> and phase qubit |phase_target>."""
    dim_sys = 2**num_system
    dim_index = 2**num_index
    dim_phase = 2
    reshaped = state_vector.reshape((dim_sys, dim_index, dim_phase))
    slice_vec = reshaped[:, 0, phase_target]
    norm = np.linalg.norm(slice_vec)
    if norm == 0:
        raise ValueError("LCU block returned zero amplitude on the target ancilla state.")
    return slice_vec / norm
