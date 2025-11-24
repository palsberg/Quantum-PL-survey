"""LCU helpers for Qiskit implementations."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import SGate, SdgGate, StatePreparation, XGate, YGate, ZGate
from qiskit.quantum_info import Statevector

from ..common import pauli_models


def amps_from_weights(weights: Sequence[float]) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Sum of LCU weights must be positive.")
    amps = np.zeros(len(weights), dtype=np.complex128)
    for idx, weight in enumerate(weights):
        if weight <= 0:
            continue
        amps[idx] = math.sqrt(weight / total)
    return amps


def apply_index_mask(qc: QuantumCircuit, index_reg: Sequence, index_value: int) -> None:
    """Flip selector qubits so that |index_value> becomes |11…1>."""
    bits = [(index_value >> k) & 1 for k in range(len(index_reg))]
    for pos, bit in enumerate(bits):
        if bit == 0:
            qc.x(index_reg[pos])


def apply_phase_tag(qc: QuantumCircuit, controls: Sequence, target, tag: str) -> None:
    if tag == "1":
        return
    gate_map = {"-1": ZGate(), "i": SGate(), "-i": SdgGate()}
    if tag not in gate_map:
        raise ValueError(f"Unknown phase tag {tag}")
    gate = gate_map[tag]
    if controls:
        qc.append(gate.control(len(controls)), list(controls) + [target])
    else:
        qc.append(gate, [target])


def apply_controlled_pauli_string(
    qc: QuantumCircuit, controls: Sequence, system_reg: Sequence, pauli_string: str
) -> None:
    gate_map = {"X": XGate(), "Y": YGate(), "Z": ZGate()}
    for idx, axis in enumerate(pauli_string):
        if axis == "I":
            continue
        gate = gate_map[axis]
        if controls:
            qc.append(gate.control(len(controls)), list(controls) + [system_reg[idx]])
        else:
            qc.append(gate, [system_reg[idx]])


def simulate_block_state(qc: QuantumCircuit, num_system: int, num_index: int) -> np.ndarray:
    """Simulate and project onto ancilla |0…0>, phase |1>."""
    state = Statevector.from_instruction(qc).data
    dim_sys = 2**num_system
    dim_index = 2**num_index
    state = state.reshape((dim_sys, dim_index, 2))
    slice_vec = state[:, 0, 1]
    norm = np.linalg.norm(slice_vec)
    if norm == 0:
        raise ValueError("LCU circuit returned zero amplitude on |0^m 1>.")
    return slice_vec / norm


def build_lcu_circuit(
    num_sites: int,
    weights: List[float],
    paulis: List[str],
    phases: List[str],
) -> tuple[QuantumCircuit, int]:
    """Assemble the PREPARE† · SELECT · PREPARE circuit."""
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
    system = QuantumRegister(num_sites, "q")
    index = QuantumRegister(m, "idx")
    phase = QuantumRegister(1, "phase")
    qc = QuantumCircuit(system, index, phase, name="lcu_block")
    qc.x(phase[0])

    prepare = StatePreparation(amps)
    qc.append(prepare, index)
    controls = list(index)

    for idx_value, weight in enumerate(weights):
        apply_index_mask(qc, index, idx_value)
        if weight > 0:
            apply_phase_tag(qc, controls, phase[0], phases[idx_value])
            apply_controlled_pauli_string(qc, controls, system, paulis[idx_value])
        apply_index_mask(qc, index, idx_value)

    qc.append(prepare.inverse(), index)
    return qc, len(index)


def tfim_lcu_state(num_sites: int, J: float, h: float, total_time: float) -> np.ndarray:
    H = pauli_models.tfim_pauli_terms(num_sites, J, h)
    gamma = pauli_models.taylor_coefficients(H, total_time)
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
    circuit, m = build_lcu_circuit(num_sites, weights, paulis, phases)
    return simulate_block_state(circuit, num_sites, m)


def heis_lcu_state(num_sites: int, J: float, field: float, total_time: float) -> np.ndarray:
    H = pauli_models.heisenberg_pauli_terms(num_sites, J, field)
    gamma = pauli_models.taylor_coefficients(H, total_time)
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
    circuit, m = build_lcu_circuit(num_sites, weights, paulis, phases)
    return simulate_block_state(circuit, num_sites, m)
