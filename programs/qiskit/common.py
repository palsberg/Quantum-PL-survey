"""Shared helpers for the Qiskit Hamiltonian-simulation programs."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter


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


def _tfim_sparse_pauli(num_sites: int, J: float, h: float) -> SparsePauliOp:
    """Return H_TFIM as a SparsePauliOp on num_sites qubits."""
    terms = []
    for i in range(num_sites - 1):
        terms.append(("ZZ", [i, i + 1], J))
    for i in range(num_sites):
        terms.append(("X", [i], h))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=num_sites).simplify()


def _heis_sparse_pauli(num_sites: int, J: float, field: float) -> SparsePauliOp:
    """Return H_XXX as a SparsePauliOp on num_sites qubits."""
    terms = []
    for i in range(num_sites - 1):
        terms.append(("XX", [i, i + 1], J))
        terms.append(("YY", [i, i + 1], J))
        terms.append(("ZZ", [i, i + 1], J))
    for i in range(num_sites):
        terms.append(("Z", [i], field))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=num_sites).simplify()


def _trotterize_sparse_pauli(
    H: SparsePauliOp, total_time: float, steps: int, label: str
) -> Tuple[QuantumCircuit, Sequence[int]]:
    """Use Qiskit's product-formula synthesis on a Pauli-sum Hamiltonian."""
    evolution_gate = PauliEvolutionGate(H, time=total_time, label=label)
    synthesis = SuzukiTrotter(order=1, reps=steps, insert_barriers=False, preserve_order=True)
    circuit = synthesis.synthesize(evolution_gate)
    # Qubits are indexed 0..num_qubits-1 in the returned circuit.
    return circuit, list(range(H.num_qubits))


def trotterize_tfim(
    num_sites: int, J: float, h: float, total_time: float, steps: int
) -> Tuple[QuantumCircuit, Sequence[int]]:
    """Build a first-order Lie–Trotter circuit for the TFIM Hamiltonian.

    This uses Qiskit's Pauli-sum evolution primitives rather than manually
    expanding each ZZ / X rotation into basis changes and CNOT ladders.
    """
    H = _tfim_sparse_pauli(num_sites, J, h)
    return _trotterize_sparse_pauli(H, total_time, steps, label="tfim_trotter")


def trotterize_heisenberg_xxx(
    num_sites: int, J: float, field: float, total_time: float, steps: int
) -> Tuple[QuantumCircuit, Sequence[int]]:
    """Build a Lie–Trotter circuit for the Heisenberg XXX chain with a field."""
    H = _heis_sparse_pauli(num_sites, J, field)
    return _trotterize_sparse_pauli(H, total_time, steps, label="heisenberg_trotter")
