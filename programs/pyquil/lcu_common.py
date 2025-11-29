"""LCU utilities for PyQuil implementations."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from pyquil import Program
from pyquil.gates import H, X, Y, Z
from pyquil.quil import DefGate

from ..common import pauli_models
from .common import simulate_statevector


PAULI_MATRICES = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    "Sdg": np.array([[1, 0], [0, -1j]], dtype=np.complex128),
}


def _householder_to_state(vector: np.ndarray) -> np.ndarray:
    """Return a unitary U such that U|0...0> = vector (normalized)."""
    v = vector.astype(np.complex128)
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
    return np.eye(dim, dtype=np.complex128) - 2.0 * np.outer(u, np.conjugate(u))


def _prepare_gate_definition(amps: np.ndarray, name: str) -> Tuple[DefGate, callable]:
    """Create a DefGate + constructor for the PREPARE unitary."""
    unitary = _householder_to_state(amps)
    def_gate = DefGate(name, unitary)
    gate = def_gate.get_constructor()
    return def_gate, gate


def _amps_from_weights(weights: Sequence[float]) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Sum of LCU weights must be positive.")
    amps = np.zeros(len(weights), dtype=np.complex128)
    for idx, weight in enumerate(weights):
        if weight > 0:
            amps[idx] = math.sqrt(weight / total)
    return amps


_CTRL_COUNTER = 0


def _controlled_matrix(base: np.ndarray, num_controls: int) -> np.ndarray:
    dim_target = base.shape[0]
    total_dim = (2**num_controls) * dim_target
    matrix = np.eye(total_dim, dtype=np.complex128)
    matrix[-dim_target:, -dim_target:] = base
    return matrix


def _apply_with_controls(
    prog: Program, base_matrix: np.ndarray, controls: Sequence[int], targets: Sequence[int]
):
    """Apply a custom unitary with optional controls."""
    global _CTRL_COUNTER
    matrix = np.asarray(base_matrix, dtype=np.complex128)
    if controls:
        matrix = _controlled_matrix(matrix, len(controls))
        qubits = list(controls) + list(targets)
    else:
        qubits = list(targets)
    def_name = f"LCU_GATE_{_CTRL_COUNTER}"
    _CTRL_COUNTER += 1
    def_gate = DefGate(def_name, matrix)
    prog += def_gate
    prog += def_gate.get_constructor()(*qubits)


def _apply_phase_tag(prog: Program, controls: Sequence[int], phase_qubit: int, tag: str):
    if tag == "1":
        return
    matrices = {
        "-1": PAULI_MATRICES["Z"],
        "i": PAULI_MATRICES["S"],
        "-i": PAULI_MATRICES["Sdg"],
    }
    matrix = matrices[tag]
    _apply_with_controls(prog, matrix, controls, [phase_qubit])


def _apply_controlled_pauli_string(
    prog: Program, controls: Sequence[int], system_qubits: Sequence[int], pauli: str
):
    for idx, axis in enumerate(pauli):
        if axis == "I":
            continue
        qubit = system_qubits[idx]
        matrix = PAULI_MATRICES[axis]
        _apply_with_controls(prog, matrix, controls, [qubit])


def _mask_index(prog: Program, index_qubits: Sequence[int], value: int):
    bits = [(value >> k) & 1 for k in range(len(index_qubits))]
    for pos, bit in enumerate(bits):
        if bit == 0:
            prog += X(index_qubits[pos])


def _simulate_and_project(prog: Program, num_system: int, num_index: int) -> np.ndarray:
    total_qubits = num_system + num_index + 1
    state = simulate_statevector(prog, total_qubits)
    dim_sys = 2**num_system
    dim_index = 2**num_index
    state = state.reshape((dim_sys, dim_index, 2))
    vec = state[:, 0, 1]
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("LCU circuit produced zero amplitude on |0^m 1>.")
    return vec / norm


def build_lcu_program(num_sites: int, weights: List[float], paulis: List[str], phases: List[str]) -> Tuple[Program, int]:
    L = len(weights)
    if L == 0:
        raise ValueError("No LCU terms provided.")
    m = max(1, int(math.ceil(math.log2(L))))
    target_len = 2**m
    identity = "I" * num_sites
    if target_len > L:
        pad = target_len - L
        weights.extend([0.0] * pad)
        paulis.extend([identity] * pad)
        phases.extend(["1"] * pad)

    amps = _amps_from_weights(weights)
    system = list(range(num_sites))
    index = list(range(num_sites, num_sites + m))
    phase_qubit = num_sites + m

    prog = Program()
    prep_name = f"PREP_{len(amps)}"
    def_gate, prep_gate = _prepare_gate_definition(amps, prep_name)
    prog += def_gate
    prog += X(phase_qubit)
    prog += prep_gate(*index)

    controls = index
    for idx_value, weight in enumerate(weights):
        _mask_index(prog, index, idx_value)
        if weight > 0:
            _apply_phase_tag(prog, controls, phase_qubit, phases[idx_value])
            if paulis[idx_value] != identity:
                _apply_controlled_pauli_string(prog, controls, system, paulis[idx_value])
        _mask_index(prog, index, idx_value)

    prog += prep_gate(*index)  # Householder is Hermitian, so equals its own inverse.
    return prog, m


def _build_lcu_state(num_sites: int, weights: List[float], paulis: List[str], phases: List[str]) -> np.ndarray:
    prog, m = build_lcu_program(num_sites, weights, paulis, phases)
    return _simulate_and_project(prog, num_sites, m)


def tfim_lcu_state(num_sites: int, J: float, h: float, total_time: float) -> np.ndarray:
    H = pauli_models.tfim_pauli_terms(num_sites, J, h)
    gamma = pauli_models.taylor_coefficients(H, total_time)
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
    return _build_lcu_state(num_sites, weights, paulis, phases)


def heis_lcu_state(num_sites: int, J: float, field: float, total_time: float) -> np.ndarray:
    H = pauli_models.heisenberg_pauli_terms(num_sites, J, field)
    gamma = pauli_models.taylor_coefficients(H, total_time)
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
    return _build_lcu_state(num_sites, weights, paulis, phases)
