"""LCU helpers in pytket"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from pytket import Circuit
from pytket.circuit import CircBox, StatePreparationBox, QControlBox
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.passes import DecomposeBoxes
from qiskit.quantum_info import Statevector

from ..common import pauli_models


def _normalized_amplitudes(weights: Sequence[float]) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("LCU weights must sum to a positive value.")
    amps = np.zeros(len(weights), dtype=np.complex128)
    for idx, weight in enumerate(weights):
        if weight > 0:
            amps[idx] = math.sqrt(weight / total)
    return amps


def _mask_index(circ: Circuit, ancilla: Sequence[int], value: int) -> None:
    bits = [(value >> k) & 1 for k in range(len(ancilla))]
    for pos, bit in enumerate(bits):
        if bit == 0:
            circ.X(ancilla[pos])


def _pauli_box(pauli: str) -> CircBox:
    sub = Circuit(len(pauli))
    for idx, axis in enumerate(pauli):
        if axis == "X":
            sub.X(idx)
        elif axis == "Y":
            sub.Y(idx)
        elif axis == "Z":
            sub.Z(idx)
    return CircBox(sub)


def _phase_box(tag: str) -> CircBox:
    circ = Circuit(1)
    if tag == "-1":
        circ.Z(0)
    elif tag == "i":
        circ.Rz(np.pi / 2, 0)
    elif tag == "-i":
        circ.Rz(-np.pi / 2, 0)
    return CircBox(circ)


def _apply_controlled_box(
    circ: Circuit, box: CircBox, controls: Sequence[int], targets: Sequence[int]
) -> None:
    """Apply `box` controlled on all `controls` being |1>.

    If `controls` is empty, apply the box unconditionally.
    """
    target_list = list(targets)
    if not controls:
        circ.add_circbox(box, target_list)
        return

    control_list = list(controls)
    n_controls = len(control_list)

    qcb = QControlBox(box, n_controls)
    qubits = control_list + target_list
    circ.add_qcontrolbox(qcb, qubits)


def _simulate_block(circ: Circuit, num_system: int, num_index: int) -> np.ndarray:
    """Simulate the full LCU block and project onto index=0, phase=1."""
    circ = circ.copy()
    # Decompose CircBox / QControlBox constructs so that the Qiskit translator
    # only sees primitive gates that it natively supports.
    DecomposeBoxes().apply(circ)
    qc = tk_to_qiskit(circ)
    state = Statevector.from_instruction(qc).data

    dim_sys = 2**num_system

    def bits(value: int, width: int) -> list[int]:
        # Least-significant-bit first, matching Qiskit's convention.
        return [(value >> k) & 1 for k in range(width)]

    success = np.zeros(dim_sys, dtype=np.complex128)
    for sys_state in range(dim_sys):
        sys_bits = bits(sys_state, num_system)
        idx_bits = [0] * num_index
        phase_bit = 1
        basis_index = 0
        shift = 0
        for bit in sys_bits + idx_bits + [phase_bit]:
            basis_index |= (bit << shift)
            shift += 1
        success[sys_state] = state[basis_index]

    norm = np.linalg.norm(success)
    if norm == 0:
        vec = np.zeros(dim_sys, dtype=np.complex128)
        vec[0] = 1.0
        return vec
    return success / norm


def _build_lcu_state(
    num_sites: int, weights: List[float], paulis: List[str], phases: List[str]
) -> np.ndarray:
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

    amps = _normalized_amplitudes(weights)

    system = list(range(num_sites))
    anc = list(range(num_sites, num_sites + m))
    phase_qubit = num_sites + m

    circ = Circuit(num_sites + m + 1)
    circ.X(phase_qubit)

    loader_state = np.asarray(amps, dtype=np.complex128)
    loader_box = StatePreparationBox(loader_state)
    loader_inv = StatePreparationBox(loader_state, is_inverse=True)
    circ.add_gate(loader_box, anc)

    for idx, weight in enumerate(weights):
        if weight <= 0:
            continue
        _mask_index(circ, anc, idx)

        if phases[idx] != "1":
            phase_box = _phase_box(phases[idx])
            _apply_controlled_box(circ, phase_box, anc, [phase_qubit])

        pauli = paulis[idx]
        if pauli != identity:
            pauli_box = _pauli_box(pauli)
            _apply_controlled_box(circ, pauli_box, anc, system)

        _mask_index(circ, anc, idx)

    circ.add_gate(loader_inv, anc)
    return _simulate_block(circ, num_sites, m)


def tfim_lcu_state(num_sites: int, J: float, h: float, total_time: float) -> np.ndarray:
    H = pauli_models.tfim_pauli_terms(num_sites, J, h)
    gamma = pauli_models.taylor_coefficients(H, total_time)
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
    return _build_lcu_state(num_sites, weights, paulis, phases)


def heis_lcu_state(
    num_sites: int, J: float, field: float, total_time: float
) -> np.ndarray:
    H = pauli_models.heisenberg_pauli_terms(num_sites, J, field)
    gamma = pauli_models.taylor_coefficients(H, total_time)
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
    return _build_lcu_state(num_sites, weights, paulis, phases)
