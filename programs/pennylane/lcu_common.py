"""LCU utilities for PennyLane implementations."""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import pennylane as qml

from ..common import pauli_models


def amps_from_weights(weights: Sequence[float]) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Sum of LCU weights must be positive.")
    amps = np.zeros(len(weights), dtype=np.complex128)
    for idx, weight in enumerate(weights):
        if weight > 0:
            amps[idx] = math.sqrt(weight / total)
    return amps


def apply_index_mask(index_wires: Sequence[int], index_value: int) -> None:
    bits = [(index_value >> k) & 1 for k in range(len(index_wires))]
    for pos, bit in enumerate(bits):
        if bit == 0:
            qml.PauliX(index_wires[pos])


def apply_phase_tag(controls: Sequence[int], phase_wire: int, tag: str) -> None:
    angle_map = {"-1": np.pi, "i": np.pi / 2, "-i": -np.pi / 2}
    if tag == "1":
        return
    angle = angle_map[tag]
    gate = qml.PhaseShift
    if controls:
        qml.ctrl(gate, control=controls)(angle, wires=phase_wire)
    else:
        gate(angle, wires=phase_wire)


def apply_controlled_pauli_string(
    controls: Sequence[int], system_wires: Sequence[int], pauli_string: str
) -> None:
    gate_map = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    for idx, axis in enumerate(pauli_string):
        if axis == "I":
            continue
        gate = gate_map[axis]
        if controls:
            qml.ctrl(gate, control=controls)(wires=system_wires[idx])
        else:
            gate(wires=system_wires[idx])


def project_statevector(state: np.ndarray, num_system: int, num_index: int) -> np.ndarray:
    dim_sys = 2**num_system
    dim_index = 2**num_index
    reshaped = state.reshape((dim_sys, dim_index, 2))
    vec = reshaped[:, 0, 1]
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("LCU circuit produced zero amplitude on |0^m 1>.")
    return vec / norm


def lcu_state(
    num_sites: int,
    weights: List[float],
    paulis: List[str],
    phases: List[str],
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

    amps = amps_from_weights(weights)
    total_wires = num_sites + m + 1
    system = list(range(num_sites))
    index = list(range(num_sites, num_sites + m))
    phase = num_sites + m

    dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(phase)
        qml.MottonenStatePreparation(amps, wires=index)
        controls = index
        for idx_value, weight in enumerate(weights):
            apply_index_mask(index, idx_value)
            if weight > 0:
                apply_phase_tag(controls, phase, phases[idx_value])
                apply_controlled_pauli_string(controls, system, paulis[idx_value])
            apply_index_mask(index, idx_value)
        qml.adjoint(qml.MottonenStatePreparation)(amps, wires=index)
        return qml.state()

    raw_state = circuit()
    return project_statevector(raw_state, num_sites, m)


def tfim_lcu_state(num_sites: int, J: float, h: float, total_time: float) -> np.ndarray:
    H = pauli_models.tfim_pauli_terms(num_sites, J, h)
    gamma = pauli_models.taylor_coefficients(H, total_time)
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
    return lcu_state(num_sites, weights, paulis, phases)


def heis_lcu_state(num_sites: int, J: float, field: float, total_time: float) -> np.ndarray:
    H = pauli_models.heisenberg_pauli_terms(num_sites, J, field)
    gamma = pauli_models.taylor_coefficients(H, total_time)
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
    return lcu_state(num_sites, weights, paulis, phases)

