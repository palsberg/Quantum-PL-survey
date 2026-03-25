"""Shared LCU utilities for Cirq implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cirq
import numpy as np


def householder_to_state(state: np.ndarray) -> np.ndarray:
    """Return a unitary whose first column equals `state` (normalized)."""
    v = state.astype(np.complex128)
    v = v / np.linalg.norm(v)
    dim = v.size
    e0 = np.zeros(dim, dtype=np.complex128)
    e0[0] = 1.0
    if np.allclose(v, e0):
        return np.eye(dim, dtype=np.complex128)
    diff = e0 - v
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return np.eye(dim, dtype=np.complex128)
    u = diff / norm
    H = np.eye(dim, dtype=np.complex128) - 2.0 * np.outer(u, np.conjugate(u))
    return H


def prepare_gate_from_amplitudes(amps: np.ndarray) -> cirq.MatrixGate:
    """Construct a MatrixGate that maps |0...0> to amps."""
    amps = amps / np.linalg.norm(amps)
    unitary = householder_to_state(amps)
    return cirq.MatrixGate(unitary)


def iterate_bits(index: int, num_bits: int) -> List[int]:
    """Return bits of index (LSB-first)."""
    return [(index >> k) & 1 for k in range(num_bits)]


def select_mask_ops(qubits: Sequence[cirq.Qid], index: int) -> List[cirq.Operation]:
    """Produce X masks that turn |index> into |11..1> for multi-control."""
    bits = iterate_bits(index, len(qubits))[::-1]
    return [cirq.X(qubits[k]) for k, bit in enumerate(bits) if bit == 0]


def amps_from_weights(weights: Sequence[float]) -> np.ndarray:
    """Convert nonnegative weights into amplitudes for PREPARE."""
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Sum of weights must be positive.")
    amps = np.zeros(len(weights), dtype=np.complex128)
    for idx, weight in enumerate(weights):
        amps[idx] = 0.0 if weight <= 0 else np.sqrt(weight / total)
    return amps


def apply_phase_tag(
    circuit: cirq.Circuit, controls: Sequence[cirq.Qid], phase_qubit: cirq.Qid, tag: str
) -> None:
    """Apply the requested phase tag using the phase ancilla."""
    if tag == "1":
        return
    if tag == "-1":
        op = cirq.Z(phase_qubit)
    elif tag == "i":
        op = cirq.S(phase_qubit)
    elif tag == "-i":
        op = cirq.S(phase_qubit) ** -1
    else:
        raise ValueError(f"Unknown phase tag {tag}")
    circuit.append(op.controlled_by(*controls))


def apply_controlled_pauli_string(
    circuit: cirq.Circuit, controls: Sequence[cirq.Qid], system_qubits: Sequence[cirq.Qid], pauli: str
) -> None:
    """Apply a controlled Pauli string on the system."""
    for qubit, axis in zip(system_qubits, pauli):
        if axis == "I":
            continue
        gate = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}[axis]
        circuit.append(gate(qubit).controlled_by(*controls))


def simulate_lcu_block(
    circuit: cirq.Circuit, ordered_qubits: Sequence[cirq.Qid], num_system: int, num_index: int
) -> np.ndarray:
    """Run the LCU circuit and project onto ancilla |0...0>, phase |1>."""
    sim = cirq.Simulator(dtype=np.complex128)
    result = sim.simulate(circuit, qubit_order=ordered_qubits)
    state = np.asarray(result.final_state_vector, dtype=np.complex128)
    dim_sys = 2**num_system
    dim_index = 2**num_index
    dim_phase = 2
    state = state.reshape((dim_sys, dim_index, dim_phase))
    slice_vec = state[:, 0, 1]
    norm = np.linalg.norm(slice_vec)
    if norm == 0:
        raise ValueError("LCU circuit returned zero amplitude on |0...01>.")
    return slice_vec / norm


@dataclass
class LcuTerm:
    weight: float
    pauli: str
    phase: str = "1"

